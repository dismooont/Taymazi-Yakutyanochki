"""
Тесты приёма zip-архивов.

Это проверки безопасности, а не удобства: содержимое архива полностью контролируется
тем, кто его загрузил. Каждый тест здесь соответствует конкретной известной атаке
(docs/WEB_PLAN.md, раздел 7.2).
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from web.archive import ArchiveError, ArchiveLimits, extract_images, inspect


def _image_bytes(color=(120, 60, 30)) -> bytes:
    import io

    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), color).save(buffer, "JPEG")
    return buffer.getvalue()


def _make_zip(path: Path, entries: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    return path


# --------------------------------------------------------------------------
# Нормальный сценарий
# --------------------------------------------------------------------------

def test_extracts_images_flat(tmp_path):
    archive = _make_zip(tmp_path / "a.zip", {
        "photos/one.jpg": _image_bytes(),
        "photos/nested/two.jpg": _image_bytes((10, 200, 10)),
    })

    count, total = inspect(archive, ArchiveLimits())
    result = extract_images(archive, tmp_path / "out", ArchiveLimits())

    assert count == 2
    assert total > 0
    assert sorted(p.name for p in result.files) == ["one.jpg", "two.jpg"]
    assert all(p.parent == tmp_path / "out" for p in result.files)  # структура папок не сохраняется


def test_same_names_from_different_folders_do_not_overwrite(tmp_path):
    archive = _make_zip(tmp_path / "a.zip", {
        "a/photo.jpg": _image_bytes((1, 2, 3)),
        "b/photo.jpg": _image_bytes((250, 250, 250)),
    })

    result = extract_images(archive, tmp_path / "out", ArchiveLimits())

    assert len(result.files) == 2
    assert len({p.read_bytes() for p in result.files}) == 2  # оба файла целы


def test_non_images_are_skipped(tmp_path):
    archive = _make_zip(tmp_path / "a.zip", {
        "good.jpg": _image_bytes(),
        "readme.txt": b"hello",
        "script.exe": b"MZ",
    })

    result = extract_images(archive, tmp_path / "out", ArchiveLimits())

    assert [p.name for p in result.files] == ["good.jpg"]
    assert sorted(name for name, _ in result.skipped) == ["readme.txt", "script.exe"]


# --------------------------------------------------------------------------
# Атаки
# --------------------------------------------------------------------------

@pytest.mark.parametrize("evil_name", [
    "../../../etc/passwd.jpg",
    "..\\..\\windows\\system32\\evil.jpg",
    "/absolute/path/evil.jpg",
    "C:\\Windows\\evil.jpg",
])
def test_zip_slip_cannot_escape_destination(tmp_path, evil_name):
    """
    Классический zip-slip: имя члена архива уводит запись за пределы целевой папки.
    Файл обязан оказаться внутри out/ и только там.
    """
    archive = _make_zip(tmp_path / "evil.zip", {evil_name: _image_bytes()})
    destination = tmp_path / "out"

    result = extract_images(archive, destination, ArchiveLimits())

    assert len(result.files) == 1
    written = result.files[0].resolve()
    assert written.parent == destination.resolve()
    assert ".." not in str(written)
    # ничего не создано на уровень выше
    assert not (tmp_path / "etc").exists()
    assert not (tmp_path.parent / "etc").exists()


def test_zip_bomb_rejected_before_extraction(tmp_path):
    """
    Архив на несколько килобайт, разворачивающийся в сотни мегабайт. Отказ должен
    произойти до распаковки — иначе диск кончится раньше, чем сработает проверка.
    """
    big = b"\0" * (20 * 1024 * 1024)
    archive = tmp_path / "bomb.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for i in range(10):
            z.writestr(f"{i}.png", big)

    assert archive.stat().st_size < 1024 * 1024  # сам архив крошечный

    with pytest.raises(ArchiveError, match="лимит"):
        inspect(archive, ArchiveLimits(max_total_bytes=50 * 1024 * 1024))

    assert not (tmp_path / "out").exists()


def test_too_many_members_rejected(tmp_path):
    archive = _make_zip(
        tmp_path / "many.zip", {f"{i}.jpg": b"x" for i in range(50)}
    )

    with pytest.raises(ArchiveError, match="слишком много файлов"):
        inspect(archive, ArchiveLimits(max_members=10))


def test_oversized_member_rejected(tmp_path):
    archive = _make_zip(tmp_path / "big.zip", {"huge.jpg": b"\0" * (5 * 1024 * 1024)})

    with pytest.raises(ArchiveError, match="слишком большой"):
        inspect(archive, ArchiveLimits(max_member_bytes=1024 * 1024))


def test_symlink_members_are_skipped(tmp_path):
    """Распакованная символическая ссылка увела бы последующую запись мимо песочницы."""
    archive = tmp_path / "link.zip"
    with zipfile.ZipFile(archive, "w") as z:
        info = zipfile.ZipInfo("evil.jpg")
        info.external_attr = (0o120777 << 16)  # S_IFLNK
        z.writestr(info, "/etc/passwd")
        z.writestr("real.jpg", _image_bytes())

    result = extract_images(archive, tmp_path / "out", ArchiveLimits())

    assert [p.name for p in result.files] == ["real.jpg"]
    assert ("evil.jpg", "символическая ссылка") in result.skipped


def test_not_a_zip_rejected(tmp_path):
    fake = tmp_path / "fake.zip"
    fake.write_bytes(b"I am definitely not a zip file")

    with pytest.raises(ArchiveError, match="не zip-архив"):
        inspect(fake, ArchiveLimits())


def test_archive_without_images_rejected(tmp_path):
    archive = _make_zip(tmp_path / "docs.zip", {"a.txt": b"1", "b.pdf": b"2"})

    with pytest.raises(ArchiveError, match="нет изображений"):
        inspect(archive, ArchiveLimits())


def test_lying_header_cannot_exceed_limit():
    """
    file_size в заголовке zip — это заявление автора архива, а не факт: заголовок можно
    подделать, заявив килобайт и положив гигабайт. Поэтому запись на диск считает
    фактически прочитанные байты и обрывается сама, не полагаясь на заголовок.
    """
    from web.archive import _copy_limited

    source = io.BytesIO(b"\0" * (3 * 1024 * 1024))  # реальных 3 МБ
    target = io.BytesIO()

    with pytest.raises(ArchiveError, match="больше, чем заявлено"):
        _copy_limited(source, target, limit=1024)

    assert target.tell() <= 1024 + (1 << 20)  # запись оборвана, а не доведена до конца


def test_oversized_member_is_skipped_not_fatal(tmp_path):
    """Слишком большой файл внутри архива пропускается — остальные должны загрузиться."""
    archive = _make_zip(tmp_path / "mixed.zip", {
        "ok.jpg": _image_bytes(),
        "huge.jpg": b"\0" * (3 * 1024 * 1024),
    })

    result = extract_images(archive, tmp_path / "out", ArchiveLimits(max_member_bytes=1024 * 1024))

    assert [p.name for p in result.files] == ["ok.jpg"]
    assert ("huge.jpg", "слишком большой файл") in result.skipped
