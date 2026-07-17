"""
Zero-Shot Image <-> Text Search на базе CLIP + FAISS.

Что делает скрипт:
  1. build   — читает изображения + CSV с подписями, считает эмбеддинги CLIP,
               строит FAISS-индекс и сохраняет его на диск.
  2. add     — добавляет новые изображения в уже существующий индекс без
               пересчёта эмбеддингов для всего датасета заново (инкрементально).
  3. text    — ищет изображения по текстовому запросу (в т.ч. с переводом RU->EN).
  4. image   — ищет похожие изображения / подписи по картинке-запросу.

Датасет (см. ТЗ):
  images_dir/*.jpg
  captions.csv  с колонками: image_id, caption_en, caption_ru (caption_ru опционален)
  image_id должен совпадать с именем файла без расширения, например image_id=139 -> 139.jpg
  (при необходимости поправьте функцию image_id_to_filename под свой датасет).

Установка зависимостей:
  pip install torch transformers faiss-cpu pillow pandas tqdm matplotlib --break-system-packages
  # для перевода запросов (опционально, deep-translator не конфликтует с huggingface_hub):
  pip install deep-translator --break-system-packages

По умолчанию команды text/image сохраняют коллаж с найденными картинками
в PNG-файл внутри index_dir (search_result_text.png / search_result_image.png).
Отключить сохранение: добавить флаг --no_plot.

Примеры запуска:
  python clip_zero_shot_search.py build \
      --images_dir ./data/images --captions_csv ./data/captions.csv --index_dir ./index

  python clip_zero_shot_search.py add \
      --images_dir ./data/new_photos --index_dir ./index

  python clip_zero_shot_search.py text \
      --index_dir ./index --query "a dog playing in the snow" --top_k 5

  python clip_zero_shot_search.py text \
      --index_dir ./index --query "собака играет в снегу" --translate --top_k 5

  python clip_zero_shot_search.py image \
      --index_dir ./index --image_path ./query.jpg --top_k 5
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

try:
    import faiss
except ImportError:
    sys.exit("Не найден faiss. Установите: pip install faiss-cpu --break-system-packages")


MODEL_NAME = "openai/clip-vit-base-patch32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32


# --------------------------------------------------------------------------
# Утилиты
# --------------------------------------------------------------------------

def build_filename_index(images_dir: Path) -> dict:
    """Один проход по всем вложенным папкам: {имя_файла: полный_путь}."""
    index = {}
    for path in images_dir.rglob("*"):
        if path.suffix.lower() in (".jpg", ".jpeg", ".png"):
            index[path.name] = path
    return index


def image_id_to_filename(image_id, filename_index: dict) -> Path:
    """
    Ищет файл изображения по image_id в предварительно построенном
    словаре {имя_файла: путь} (см. build_filename_index).
    Формат MS COCO: image_id дополняется нулями слева до 12 цифр,
    например image_id=139 -> 000000000139.jpg
    """
    padded_id = str(image_id).zfill(12)
    candidates_names = [f"{padded_id}{ext}" for ext in (".jpg", ".jpeg", ".png")]
    candidates_names += [f"{image_id}{ext}" for ext in (".jpg", ".jpeg", ".png")]

    for name in candidates_names:
        if name in filename_index:
            return filename_index[name]

    raise FileNotFoundError(
        f"Изображение для image_id={image_id} не найдено "
        f"(искал {padded_id}.jpg и {image_id}.jpg, включая подпапки)"
    )


def normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    return vectors / norms


def load_model():
    print(f"Загрузка модели {MODEL_NAME} на {DEVICE}...")
    model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    return model, processor


def extract_features_tensor(output):
    """
    Совместимость между версиями transformers:
    - в transformers < 5.0 get_image_features/get_text_features возвращают torch.Tensor напрямую
    - в transformers >= 5.0 возвращается объект BaseModelOutputWithPooling
    Извлекаем сам тензор эмбеддинга независимо от версии.
    """
    if torch.is_tensor(output):
        return output
    for attr in ("pooler_output", "image_embeds", "text_embeds", "last_hidden_state"):
        if hasattr(output, attr):
            tensor = getattr(output, attr)
            if tensor is not None:
                # last_hidden_state имеет форму (batch, seq_len, dim) -> берём CLS-токен
                if attr == "last_hidden_state" and tensor.dim() == 3:
                    tensor = tensor[:, 0, :]
                return tensor
    raise TypeError(f"Не удалось извлечь тензор эмбеддинга из объекта типа {type(output)}")


def visualize_images(items, scores, out_path, title, captions=None):
    """
    Сохраняет коллаж из найденных изображений с их score в PNG-файл.
    items: список dict с ключом 'path' (как в images_meta.json)
    captions: опционально — список подписей под каждой картинкой (напр. caption для image->text)
    """
    # matplotlib импортируется лениво: он нужен только для CLI-визуализации,
    # а боту (bot/inference.py) не требуется — так образ Docker остаётся лёгким.
    import matplotlib
    matplotlib.use("Agg")  # без GUI: сохраняем результат в файл, не пытаемся открыть окно
    import matplotlib.pyplot as plt

    n = len(items)
    if n == 0:
        print("Нечего визуализировать — пустой список результатов.")
        return

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4.5))
    if n == 1:
        axes = [axes]

    for ax, item, score in zip(axes, items, scores):
        try:
            img = Image.open(item["path"]).convert("RGB")
            ax.imshow(img)
        except Exception as e:
            ax.text(0.5, 0.5, f"не удалось открыть\n{e}", ha="center", va="center", wrap=True)
        subtitle = f"id={item['image_id']}\nscore={score:.3f}"
        ax.set_title(subtitle, fontsize=10)
        ax.axis("off")

    if captions:
        for ax, cap in zip(axes, captions):
            ax.set_xlabel(cap, fontsize=8, wrap=True)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Коллаж с результатами сохранён: {out_path}")


# --------------------------------------------------------------------------
# Построение индекса
# --------------------------------------------------------------------------

def compute_image_embeddings(model, processor, image_paths):
    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(image_paths), BATCH_SIZE), desc="Эмбеддинги изображений"):
            batch_paths = image_paths[i:i + BATCH_SIZE]
            images = [Image.open(p).convert("RGB") for p in batch_paths]
            inputs = processor(images=images, return_tensors="pt").to(DEVICE)
            feats = extract_features_tensor(model.get_image_features(**inputs))
            embeddings.append(feats.cpu().numpy())
    return normalize(np.concatenate(embeddings, axis=0)).astype("float32")


def compute_text_embeddings(model, processor, texts):
    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Эмбеддинги текстов"):
            batch = texts[i:i + BATCH_SIZE]
            inputs = processor(
                text=batch, return_tensors="pt", padding=True, truncation=True
            ).to(DEVICE)
            feats = extract_features_tensor(model.get_text_features(**inputs))
            embeddings.append(feats.cpu().numpy())
    return normalize(np.concatenate(embeddings, axis=0)).astype("float32")


def build_index(images_dir: str, captions_csv: str, index_dir: str):
    images_dir = Path(images_dir)
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(captions_csv)
    required_cols = {"image_id", "caption_en"}
    if not required_cols.issubset(df.columns):
        sys.exit(f"CSV должен содержать колонки {required_cols}, найдено: {list(df.columns)}")

    # Уникальные изображения (по одному эмбеддингу на картинку)
    unique_ids = df["image_id"].drop_duplicates().tolist()
    print(f"Индексирую файлы в {images_dir} (включая подпапки)...")
    filename_index = build_filename_index(images_dir)
    print(f"Найдено файлов изображений: {len(filename_index)}")

    image_paths, valid_ids = [], []
    for image_id in unique_ids:
        try:
            image_paths.append(image_id_to_filename(image_id, filename_index))
            valid_ids.append(image_id)
        except FileNotFoundError as e:
            print(f"[пропуск] {e}")

    df = df[df["image_id"].isin(valid_ids)].reset_index(drop=True)

    model, processor = load_model()

    # --- эмбеддинги картинок (одна запись на уникальный image_id) ---
    image_embs = compute_image_embeddings(model, processor, image_paths)
    faiss.normalize_L2(image_embs)
    image_index = faiss.IndexFlatIP(image_embs.shape[1])
    image_index.add(image_embs)
    faiss.write_index(image_index, str(index_dir / "images.index"))

    with open(index_dir / "images_meta.json", "w", encoding="utf-8") as f:
        json.dump(
            [{"image_id": str(i), "path": str(p)} for i, p in zip(valid_ids, image_paths)],
            f, ensure_ascii=False, indent=2,
        )

    # --- эмбеддинги подписей (по одной записи на каждую подпись, 5 на картинку) ---
    captions = df["caption_en"].astype(str).tolist()
    text_embs = compute_text_embeddings(model, processor, captions)
    faiss.normalize_L2(text_embs)
    text_index = faiss.IndexFlatIP(text_embs.shape[1])
    text_index.add(text_embs)
    faiss.write_index(text_index, str(index_dir / "captions.index"))

    with open(index_dir / "captions_meta.json", "w", encoding="utf-8") as f:
        json.dump(
            [{"image_id": str(row.image_id), "caption": row.caption_en} for row in df.itertuples()],
            f, ensure_ascii=False, indent=2,
        )

    print(f"Готово: {len(valid_ids)} изображений, {len(captions)} подписей -> {index_dir}")


def normalize_id(value) -> str:
    """
    Приводит числовой id к каноничному виду без ведущих нулей (как хранится
    после чтения CSV через pandas), чтобы '000000000139' и '139' считались
    одним и тем же image_id при дедупликации. Нечисловые id (например,
    Telegram file_id) возвращаются как есть.
    """
    s = str(value)
    return str(int(s)) if s.isdigit() else s


def add_to_index(images_dir: str, index_dir: str, captions_csv: str = None):
    """
    Добавляет новые изображения в уже существующий индекс (без пересчёта всего с нуля).
    Изображения, чей image_id уже присутствует в индексе, пропускаются.

    Режим 1 (без --captions_csv): image_id берётся из имени файла (без расширения).
        Изображение добавляется только в images.index (доступно для image->image
        и text->image поиска, но не появится в результатах image->captions поиска).
        Это основной режим для сценария "пользователь прислал фото боту".

    Режим 2 (с --captions_csv): та же логика, что и в build — image_id и подписи
        берутся из CSV, обновляются оба индекса (images.index и captions.index).
    """
    images_dir = Path(images_dir)
    index_dir = Path(index_dir)

    images_index_path = index_dir / "images.index"
    images_meta_path = index_dir / "images_meta.json"
    if not images_index_path.exists() or not images_meta_path.exists():
        sys.exit(
            f"В {index_dir} не найден существующий индекс (images.index/images_meta.json). "
            f"Сначала выполните команду 'build'."
        )

    image_index, image_meta = load_index(str(index_dir), "images")
    existing_ids = {normalize_id(item["image_id"]) for item in image_meta}

    print(f"Индексирую файлы в {images_dir} (включая подпапки)...")
    filename_index = build_filename_index(images_dir)
    print(f"Найдено файлов изображений: {len(filename_index)}")

    if captions_csv:
        df = pd.read_csv(captions_csv)
        required_cols = {"image_id", "caption_en"}
        if not required_cols.issubset(df.columns):
            sys.exit(f"CSV должен содержать колонки {required_cols}, найдено: {list(df.columns)}")
        candidate_ids = df["image_id"].drop_duplicates().tolist()
    else:
        # image_id = имя файла без расширения
        candidate_ids = [Path(name).stem for name in filename_index.keys()]
        df = None

    new_paths, new_ids = [], []
    for image_id in candidate_ids:
        norm_id = normalize_id(image_id)
        if norm_id in existing_ids:
            continue  # уже в индексе (сравнение с учётом zero-padding) — пропускаем
        try:
            path = image_id_to_filename(image_id, filename_index)
        except FileNotFoundError as e:
            print(f"[пропуск] {e}")
            continue
        new_paths.append(path)
        new_ids.append(norm_id)
        existing_ids.add(norm_id)  # защита от дублей и внутри самого нового набора файлов

    if not new_paths:
        print("Новых изображений для добавления не найдено (все уже есть в индексе или файлы не найдены).")
        return

    model, processor = load_model()

    print(f"Добавляю {len(new_paths)} новых изображений...")
    new_embs = compute_image_embeddings(model, processor, new_paths)
    faiss.normalize_L2(new_embs)
    image_index.add(new_embs)
    faiss.write_index(image_index, str(images_index_path))

    image_meta.extend([{"image_id": i, "path": str(p)} for i, p in zip(new_ids, new_paths)])
    with open(images_meta_path, "w", encoding="utf-8") as f:
        json.dump(image_meta, f, ensure_ascii=False, indent=2)

    print(f"Готово: добавлено {len(new_paths)} изображений. Всего в индексе: {len(image_meta)}")

    # --- опционально: обновляем индекс подписей теми же новыми image_id ---
    if df is not None:
        captions_index_path = index_dir / "captions.index"
        captions_meta_path = index_dir / "captions_meta.json"
        if captions_index_path.exists() and captions_meta_path.exists():
            cap_index, cap_meta = load_index(str(index_dir), "captions")
        else:
            cap_index, cap_meta = None, []

        new_ids_set = set(new_ids)
        new_captions_df = df[df["image_id"].astype(str).isin(new_ids_set)]
        captions = new_captions_df["caption_en"].astype(str).tolist()

        if captions:
            cap_embs = compute_text_embeddings(model, processor, captions)
            faiss.normalize_L2(cap_embs)
            if cap_index is None:
                cap_index = faiss.IndexFlatIP(cap_embs.shape[1])
            cap_index.add(cap_embs)
            faiss.write_index(cap_index, str(captions_index_path))

            cap_meta.extend([
                {"image_id": str(row.image_id), "caption": row.caption_en}
                for row in new_captions_df.itertuples()
            ])
            with open(captions_meta_path, "w", encoding="utf-8") as f:
                json.dump(cap_meta, f, ensure_ascii=False, indent=2)

            print(f"Добавлено {len(captions)} новых подписей. Всего подписей в индексе: {len(cap_meta)}")


# --------------------------------------------------------------------------
# Поиск
# --------------------------------------------------------------------------

def load_index(index_dir: str, kind: str):
    index_dir = Path(index_dir)
    index = faiss.read_index(str(index_dir / f"{kind}.index"))
    with open(index_dir / f"{kind}_meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    return index, meta


TRANSLATE_CACHE_FILE = "translate_cache.json"


def translate_ru_to_en(text: str, cache_path: str) -> str:
    """Переводит запрос RU->EN с кэшированием (см. ограничения по API в ТЗ)."""
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            cache = json.load(f)
    if text in cache:
        return cache[text]

    try:
        from deep_translator import GoogleTranslator
        translated = GoogleTranslator(source="ru", target="en").translate(text)
        if not translated:
            raise ValueError("Переводчик вернул пустую строку")
    except Exception as e:
        print(f"[перевод не удался, используется исходный текст без кэширования] {e}")
        return text  # НЕ кэшируем неудачный перевод, чтобы попытаться снова в следующий раз

    cache[text] = translated
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    return translated


def search_by_text(index_dir: str, query: str, top_k: int, translate: bool, save_plot: bool = True):
    original_query = query
    if translate:
        cache_path = str(Path(index_dir) / TRANSLATE_CACHE_FILE)
        query = translate_ru_to_en(query, cache_path)
        print(f"Переведённый запрос: {query}")

    model, processor = load_model()
    query_emb = compute_text_embeddings(model, processor, [query])
    faiss.normalize_L2(query_emb)

    index, meta = load_index(index_dir, "images")
    scores, indices = index.search(query_emb, top_k)

    print(f"\nТоп-{top_k} изображений по запросу: \"{query}\"")
    found_items, found_scores = [], []
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
        item = meta[idx]
        print(f"{rank}. image_id={item['image_id']}  score={score:.4f}  path={item['path']}")
        found_items.append(item)
        found_scores.append(score)

    if save_plot:
        out_path = str(Path(index_dir) / "search_result_text.png")
        visualize_images(found_items, found_scores, out_path, f'Запрос: "{original_query}" -> "{query}"')


def search_by_image(index_dir: str, image_path: str, top_k: int, save_plot: bool = True):
    model, processor = load_model()
    image = Image.open(image_path).convert("RGB")
    with torch.no_grad():
        inputs = processor(images=[image], return_tensors="pt").to(DEVICE)
        query_emb = extract_features_tensor(model.get_image_features(**inputs)).cpu().numpy().astype("float32")
    faiss.normalize_L2(query_emb)

    # похожие изображения
    img_index, img_meta = load_index(index_dir, "images")
    scores, indices = img_index.search(query_emb, top_k)
    print(f"\nТоп-{top_k} похожих изображений:")
    found_items, found_scores = [], []
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
        item = img_meta[idx]
        print(f"{rank}. image_id={item['image_id']}  score={score:.4f}  path={item['path']}")
        found_items.append(item)
        found_scores.append(score)

    if save_plot:
        out_path = str(Path(index_dir) / "search_result_image.png")
        visualize_images(
            found_items, found_scores, out_path,
            f"Похожие изображения на запрос: {Path(image_path).name}",
        )

    # ближайшие подписи (image -> text retrieval)
    cap_index, cap_meta = load_index(index_dir, "captions")
    scores, indices = cap_index.search(query_emb, top_k)
    print(f"\nТоп-{top_k} релевантных подписей:")
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
        item = cap_meta[idx]
        print(f"{rank}. image_id={item['image_id']}  score={score:.4f}  caption=\"{item['caption']}\"")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Zero-Shot Image/Text Search на CLIP")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_build = subparsers.add_parser("build", help="построить FAISS-индекс с нуля")
    p_build.add_argument("--images_dir", required=True)
    p_build.add_argument("--captions_csv", required=True)
    p_build.add_argument("--index_dir", required=True)

    p_add = subparsers.add_parser("add", help="добавить новые изображения в существующий индекс")
    p_add.add_argument("--images_dir", required=True, help="папка с новыми изображениями")
    p_add.add_argument("--index_dir", required=True, help="папка с уже существующим индексом (см. build)")
    p_add.add_argument("--captions_csv", default=None,
                        help="опционально: CSV с image_id,caption_en для новых фото. "
                             "Если не указан — image_id берётся из имени файла, подписи не индексируются.")

    p_text = subparsers.add_parser("text", help="поиск изображений по тексту")
    p_text.add_argument("--index_dir", required=True)
    p_text.add_argument("--query", required=True)
    p_text.add_argument("--top_k", type=int, default=5)
    p_text.add_argument("--translate", action="store_true", help="перевести запрос RU->EN")
    p_text.add_argument("--no_plot", action="store_true", help="не сохранять коллаж с результатами")

    p_image = subparsers.add_parser("image", help="поиск по изображению-запросу")
    p_image.add_argument("--index_dir", required=True)
    p_image.add_argument("--image_path", required=True)
    p_image.add_argument("--top_k", type=int, default=5)
    p_image.add_argument("--no_plot", action="store_true", help="не сохранять коллаж с результатами")

    args = parser.parse_args()

    if args.command == "build":
        build_index(args.images_dir, args.captions_csv, args.index_dir)
    elif args.command == "add":
        add_to_index(args.images_dir, args.index_dir, args.captions_csv)
    elif args.command == "text":
        search_by_text(args.index_dir, args.query, args.top_k, args.translate, save_plot=not args.no_plot)
    elif args.command == "image":
        search_by_image(args.index_dir, args.image_path, args.top_k, save_plot=not args.no_plot)


if __name__ == "__main__":
    main()
