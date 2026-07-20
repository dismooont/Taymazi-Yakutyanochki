"""
Генерация подписей к снимкам базы (фаза C3). Никакого планирования: команда
проходит по базе и размечает то, что ещё не размечено.

Фоновая очередь с запуском по бездействию — это C4. Сначала нужно убедиться, что
машинные подписи вообще сохраняют выигрыш, измеренный в C0 на человеческих.

Два режима.

1. Разметить базу (пишет прямо в неё):

       python scripts/generate_captions.py --root data/users/<id>/databases/<id>

   Подписи сохраняются после каждой порции: прогон на тысячу снимков идёт часами,
   и прерывание не должно обнулять сделанное.

2. Разметить снимки индекса COCO в отдельный файл (для замера):

       python scripts/generate_captions.py --index_dir index --output blip.json --limit 1000

   В саму базу COCO писать нельзя — она открыта только на чтение.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.captioner import DEFAULT_BLIP_MODEL, Captioner  # noqa: E402
from core.store import IndexStore  # noqa: E402


def human_time(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f} с"
    if seconds < 5400:
        return f"{seconds / 60:.0f} мин"
    return f"{seconds / 3600:.1f} ч"


def caption_store(args, captioner: Captioner) -> None:
    store = IndexStore.open(args.root)
    pending = store.photos_without_caption(limit=args.limit)
    covered, total = store.captions_coverage()
    print(f"База: {total} снимков, размечено {covered}, в работе {len(pending)}")
    if not pending:
        print("Размечать нечего")
        return

    started = time.perf_counter()
    done = 0
    for start in range(0, len(pending), args.chunk):
        chunk = pending[start:start + args.chunk]
        paths = [str(store.photo_path(photo)) for photo in chunk]
        texts = captioner.caption_images(paths, batch_size=args.batch)

        # Пишем после каждой порции, а не в конце: прогон идёт часами.
        store.set_caption_texts(
            {photo.photo_id: text for photo, text in zip(chunk, texts) if text},
            model=args.model,
        )
        done += len(chunk)
        elapsed = time.perf_counter() - started
        left = (len(pending) - done) * elapsed / done
        print(f"  {done}/{len(pending)}  ({elapsed / done:.1f} с/снимок, осталось ~{human_time(left)})")

    if not args.no_vectors:
        index_captions(store, args)

    covered, total = store.captions_coverage()
    print(f"Готово: размечено {covered} из {total}")
    if not store.fusion_ready():
        print("Внимание: покрытия пока не хватает, поиск останется обычным "
              "(нужна половина базы и не меньше десяти подписей)")


def index_captions(store: IndexStore, args) -> None:
    """Кодирует подписи текстовой моделью — без этого искать по ним нечем."""
    from core.captions import CaptionEncoder

    photos = [p for p in store.list_photos() if p.caption]
    if not photos:
        return
    encoder = CaptionEncoder.get(args.caption_model)
    print(f"Кодирую {len(photos)} подписей моделью {args.caption_model}...")
    vectors = encoder.encode([p.caption for p in photos])
    store.set_caption_vectors(
        {photo.photo_id: vector for photo, vector in zip(photos, vectors)},
        model=args.caption_model,
    )


def caption_index_dir(args, captioner: Captioner) -> None:
    """Разметка снимков COCO в отдельный файл — для замера, не для базы."""
    index_dir = Path(args.index_dir)
    meta = json.loads((index_dir / "images_meta.json").read_text(encoding="utf-8"))
    if args.limit:
        meta = meta[: args.limit]

    output = Path(args.output)
    done: dict[str, str] = {}
    if output.exists():
        done = json.loads(output.read_text(encoding="utf-8"))
        print(f"Продолжаю: уже размечено {len(done)}")

    pending = [item for item in meta if str(item["image_id"]) not in done]
    print(f"Снимков к разметке: {len(pending)} из {len(meta)}")

    started = time.perf_counter()
    for start in range(0, len(pending), args.chunk):
        chunk = pending[start:start + args.chunk]
        paths = []
        for item in chunk:
            path = Path(item["path"])
            paths.append(str(path if path.is_absolute() else PROJECT_ROOT / path))
        texts = captioner.caption_images(paths, batch_size=args.batch)
        for item, text in zip(chunk, texts):
            done[str(item["image_id"])] = text

        output.write_text(json.dumps(done, ensure_ascii=False, indent=1), encoding="utf-8")
        processed = start + len(chunk)
        elapsed = time.perf_counter() - started
        left = (len(pending) - processed) * elapsed / processed
        print(f"  {processed}/{len(pending)}  "
              f"({elapsed / processed:.1f} с/снимок, осталось ~{human_time(left)})")

    print(f"Готово: {len(done)} подписей в {output}")


def main():
    parser = argparse.ArgumentParser(description="Генерация подписей BLIP")
    parser.add_argument("--root", help="папка базы — подписи пишутся прямо в неё")
    parser.add_argument("--index_dir", help="папка индекса COCO — режим замера")
    parser.add_argument("--output", help="куда сложить подписи в режиме замера")
    parser.add_argument("--limit", type=int, default=None, help="сколько снимков разметить")
    # Размер пачки заметно влияет на скорость: на этой машине замерено 3,09 с на
    # снимок при 4 и 1,80 с при 8.
    parser.add_argument("--batch", type=int, default=8, help="снимков за один прогон модели")
    parser.add_argument("--chunk", type=int, default=32, help="через сколько снимков сохранять")
    parser.add_argument("--threads", type=int, default=None,
                        help="ограничить torch по числу ядер (чтобы не мешать поиску)")
    parser.add_argument("--model", default=DEFAULT_BLIP_MODEL)
    parser.add_argument("--caption_model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--no_vectors", action="store_true",
                        help="только тексты, без кодирования подписей")
    args = parser.parse_args()

    if bool(args.root) == bool(args.index_dir):
        parser.error("укажите либо --root (разметить базу), либо --index_dir (замер)")
    if args.index_dir and not args.output:
        parser.error("в режиме замера нужен --output")

    captioner = Captioner.get(args.model, num_threads=args.threads)
    if args.root:
        caption_store(args, captioner)
    else:
        caption_index_dir(args, captioner)


if __name__ == "__main__":
    main()
