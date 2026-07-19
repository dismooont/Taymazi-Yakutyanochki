"""
Подсчёт метрик качества retrieval (Recall@1/5/10) для построенного индекса.

Считаются обе стандартные для MS COCO метрики (см. ТЗ, п. 1.6):
  * text -> image  — по подписи ищем изображение, проверяем, что нужная картинка
                     попала в топ-K результатов;
  * image -> text  — по изображению ищем подписи, проверяем, что среди топ-K есть
                     хотя бы одна подпись этого же изображения.

Скрипт НЕ пересчитывает эмбеддинги: он читает уже готовые векторы прямо из
FAISS-индексов (`images.index` / `captions.index`), поэтому работает за секунды
и не требует загрузки CLIP. Модель грузится только при флаге `--latency`, когда
нужно измерить полное время ответа на запрос (ТЗ: < 1 с на CPU).

Изображения, добавленные без подписей (например, присланные боту фото), в качестве
запросов не используются, но остаются в индексе как дистракторы — так оценка честнее.

Использование:
    python scripts/eval_recall.py --index_dir index
    python scripts/eval_recall.py --index_dir index --limit 1000 --latency
    python scripts/eval_recall.py --index_dir index --output index/recall.json
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

# --- переиспользуем утилиты CLI-скрипта (load_index, normalize_id и пр.) ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import clip_zero_shot_search as czss  # noqa: E402

DEFAULT_KS = (1, 5, 10)


def reconstruct_all(index) -> np.ndarray:
    """
    Достаёт все векторы из FAISS-индекса в виде матрицы (n, dim).
    Векторы в индексе уже L2-нормализованы (см. build_index), поэтому
    пересчитывать эмбеддинги CLIP не нужно.
    """
    n = index.ntotal
    try:
        return index.reconstruct_n(0, n)
    except RuntimeError:
        # запасной путь для типов индексов без пакетного reconstruct_n
        return np.vstack([index.reconstruct(i) for i in range(n)])


def recall_at_k(hit_ranks: list, ks=DEFAULT_KS, total: int = None) -> dict:
    """
    hit_ranks: для каждого запроса — позиция (0-based) первого правильного
    результата, либо None, если правильного результата нет в выдаче.
    """
    total = total if total is not None else len(hit_ranks)
    if total == 0:
        return {k: float("nan") for k in ks}
    return {
        k: sum(1 for r in hit_ranks if r is not None and r < k) / total
        for k in ks
    }


def eval_text_to_image(caption_embs, caption_ids, images_meta, ks, batch_size, index):
    """По каждой подписи ищем изображения; правильный ответ — image_id этой подписи."""
    max_k = max(ks)
    row_ids = [czss.normalize_id(item["image_id"]) for item in images_meta]

    hit_ranks = []
    for start in range(0, len(caption_embs), batch_size):
        batch = caption_embs[start:start + batch_size]
        _, indices = index.search(batch, max_k)
        for offset, row in enumerate(indices):
            gt = caption_ids[start + offset]
            rank = next((r for r, idx in enumerate(row) if row_ids[idx] == gt), None)
            hit_ranks.append(rank)

    return recall_at_k(hit_ranks, ks)


def eval_image_to_text(image_embs, image_ids, captions_meta, ks, batch_size, index):
    """
    По каждому изображению ищем подписи; правильный ответ — любая из подписей
    этого изображения (в MS COCO их пять).
    """
    max_k = max(ks)
    row_ids = [czss.normalize_id(item["image_id"]) for item in captions_meta]
    ids_with_captions = set(row_ids)

    # изображения без подписей (фото от пользователей) запросами не делаем,
    # но из индекса не убираем — они остаются дистракторами
    query_rows = [i for i, image_id in enumerate(image_ids) if image_id in ids_with_captions]
    skipped = len(image_ids) - len(query_rows)

    hit_ranks = []
    for start in range(0, len(query_rows), batch_size):
        rows = query_rows[start:start + batch_size]
        batch = image_embs[rows]
        _, indices = index.search(batch, max_k)
        for offset, row in enumerate(indices):
            gt = image_ids[rows[offset]]
            rank = next((r for r, idx in enumerate(row) if row_ids[idx] == gt), None)
            hit_ranks.append(rank)

    return recall_at_k(hit_ranks, ks), len(query_rows), skipped


def measure_latency(index_dir: str, queries: list, images_index, top_k: int = 5) -> dict:
    """
    Измеряет полное время ответа на текстовый запрос: энкодер CLIP + поиск FAISS
    (ТЗ, п. 1.6 — среднее время не должно превышать 1 с на CPU).
    Загрузка модели в замер не входит: в боте она резидентна (см. README, раздел 9.1).
    """
    model, processor = czss.load_model()

    # на одиночных запросах прогресс-бар tqdm только засоряет вывод — глушим его
    # (после рефакторинга под core/ это делается флагом, а не подменой czss.tqdm)
    encode = lambda text: czss.compute_text_embeddings(  # noqa: E731
        model, processor, [text], show_progress=False
    )

    # первый прогон прогревает ленивую инициализацию torch — в статистику не идёт
    encode(queries[0])

    encode_times, search_times = [], []
    for query in queries:
        t0 = time.perf_counter()
        emb = encode(query)  # эмбеддинг уже нормализован (core.model.l2_normalize)
        t1 = time.perf_counter()
        images_index.search(emb, top_k)
        t2 = time.perf_counter()
        encode_times.append(t1 - t0)
        search_times.append(t2 - t1)

    total = [e + s for e, s in zip(encode_times, search_times)]
    return {
        "queries": len(queries),
        "device": czss.DEVICE,
        "encode_mean_s": float(np.mean(encode_times)),
        "search_mean_s": float(np.mean(search_times)),
        "total_mean_s": float(np.mean(total)),
        "total_p95_s": float(np.percentile(total, 95)),
        "total_max_s": float(np.max(total)),
    }


def format_recall(name: str, recall: dict, n_queries: int) -> str:
    parts = "  ".join(f"Recall@{k} = {v * 100:6.2f} %" for k, v in sorted(recall.items()))
    return f"{name:<16} ({n_queries:>6} запросов):  {parts}"


def main():
    parser = argparse.ArgumentParser(description="Recall@K для CLIP zero-shot индекса")
    parser.add_argument("--index_dir", required=True, help="папка с индексом (см. команду build)")
    parser.add_argument("--ks", default="1,5,10", help="значения K через запятую (по умолчанию 1,5,10)")
    parser.add_argument("--limit", type=int, default=None,
                        help="оценить на случайной подвыборке запросов (для быстрой проверки)")
    parser.add_argument("--seed", type=int, default=42, help="seed для --limit")
    parser.add_argument("--batch_size", type=int, default=256, help="размер батча при поиске в FAISS")
    parser.add_argument("--latency", action="store_true",
                        help="дополнительно замерить время ответа (грузит CLIP)")
    parser.add_argument("--latency_queries", type=int, default=20,
                        help="сколько запросов использовать для замера времени")
    parser.add_argument("--output", default=None, help="сохранить результаты в JSON-файл")
    args = parser.parse_args()

    ks = tuple(sorted(int(k) for k in args.ks.split(",")))
    index_dir = Path(args.index_dir)

    if not (index_dir / "captions.index").exists():
        sys.exit(
            f"В {index_dir} нет captions.index — метрики считать не по чему. "
            f"Постройте индекс с --captions_csv (команда build)."
        )

    images_index, images_meta = czss.load_index(str(index_dir), "images")
    captions_index, captions_meta = czss.load_index(str(index_dir), "captions")
    print(f"Индекс: {images_index.ntotal} изображений, {captions_index.ntotal} подписей")

    image_embs = reconstruct_all(images_index)
    caption_embs = reconstruct_all(captions_index)
    image_ids = [czss.normalize_id(item["image_id"]) for item in images_meta]
    caption_ids = [czss.normalize_id(item["image_id"]) for item in captions_meta]

    if args.limit and args.limit < len(caption_embs):
        rng = random.Random(args.seed)
        rows = sorted(rng.sample(range(len(caption_embs)), args.limit))
        t2i_embs = caption_embs[rows]
        t2i_ids = [caption_ids[i] for i in rows]
        print(f"Подвыборка для text->image: {args.limit} подписей (seed={args.seed})")
    else:
        t2i_embs, t2i_ids = caption_embs, caption_ids

    print("\nСчитаю text -> image...")
    t2i = eval_text_to_image(t2i_embs, t2i_ids, images_meta, ks, args.batch_size, images_index)

    print("Считаю image -> text...")
    i2t, n_i2t, skipped = eval_image_to_text(
        image_embs, image_ids, captions_meta, ks, args.batch_size, captions_index
    )

    print("\n" + "=" * 72)
    print(format_recall("text -> image", t2i, len(t2i_ids)))
    print(format_recall("image -> text", i2t, n_i2t))
    print("=" * 72)
    if skipped:
        print(f"(пропущено {skipped} изображений без подписей — использованы только как дистракторы)")
    print(
        "Ориентир из ТЗ для zero-shot CLIP на полном MS COCO: "
        "Recall@1 ~ 58 %, Recall@5 ~ 81,5 %, Recall@10 ~ 88,1 %"
    )

    report = {
        "index_dir": str(index_dir),
        "images_indexed": int(images_index.ntotal),
        "captions_indexed": int(captions_index.ntotal),
        "model": czss.MODEL_NAME,
        "text_to_image": {f"recall@{k}": v for k, v in sorted(t2i.items())},
        "text_to_image_queries": len(t2i_ids),
        "image_to_text": {f"recall@{k}": v for k, v in sorted(i2t.items())},
        "image_to_text_queries": n_i2t,
    }

    if args.latency:
        print("\nЗамер времени ответа (загружаю модель)...")
        sample = [captions_meta[i]["caption"] for i in range(min(args.latency_queries, len(captions_meta)))]
        latency = measure_latency(str(index_dir), sample, images_index)
        report["latency"] = latency
        print(
            f"Среднее время запроса: {latency['total_mean_s'] * 1000:.0f} мс "
            f"(энкодер {latency['encode_mean_s'] * 1000:.0f} мс + FAISS "
            f"{latency['search_mean_s'] * 1000:.1f} мс), p95 = "
            f"{latency['total_p95_s'] * 1000:.0f} мс, device={latency['device']}"
        )
        limit_ok = "уложились" if latency["total_mean_s"] < 1.0 else "НЕ уложились"
        print(f"Требование ТЗ (< 1 с на запрос): {limit_ok}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\nОтчёт сохранён: {args.output}")


if __name__ == "__main__":
    main()
