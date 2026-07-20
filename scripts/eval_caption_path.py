"""
Фаза C0: стоит ли вообще искать через подписи (SBERT) в дополнение к CLIP.

Замысел замера. Подписи MS COCO написаны людьми, то есть заведомо лучше всего,
что сгенерирует BLIP. Значит, они дают ВЕРХНЮЮ ГРАНИЦУ качества каптион-пути:
если поиск по идеальным подписям не выигрывает у CLIP, то по машинным не выиграет
тем более, и городить BLIP незачем.

Как устроено сравнение. У каждого снимка COCO пять подписей. Первую берём как
запрос, остальные кладём в индекс как документы. Так подпись-запрос никогда не
совпадает с собственным документом — иначе SBERT нашёл бы сам себя с оценкой 1.0
и замер не значил бы ничего.

Три системы на одном и том же наборе запросов и одном и том же пуле снимков:
  clip   — вектор запроса (CLIP-текст) против векторов изображений;
  sbert  — вектор запроса (SBERT) против векторов чужих подписей, оценка снимка
           берётся как максимум по его подписям;
  hybrid — взвешенная сумма оценок двух путей, вес подбирается перебором.

Векторы CLIP не пересчитываются: они уже лежат в captions.index и images.index.

Использование:
    python scripts/eval_caption_path.py --index_dir index
    python scripts/eval_caption_path.py --index_dir index --limit 500
"""

import argparse
import json
import re
import sys
import time
from collections import OrderedDict, defaultdict
from pathlib import Path

import faiss
import numpy as np

DEFAULT_KS = (1, 5, 10)
SBERT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Классы запросов, на которых CLIP слаб по литературе. Средний Recall выигрыш
# на них размывает, поэтому считаем отдельно.
RELATIONAL = re.compile(
    r"\b(left of|right of|behind|in front of|next to|on top of|underneath|"
    r"beneath|above|below|beside|between|inside|outside)\b",
    re.I,
)
COUNTING = re.compile(
    r"\b(two|three|four|five|six|seven|eight|nine|ten|couple|several|\d+)\b", re.I
)
NEGATION = re.compile(r"\b(without|no|not|empty|none|nobody)\b", re.I)


def load_index(index_dir: Path, name: str):
    index = faiss.read_index(str(index_dir / f"{name}.index"))
    meta = json.loads((index_dir / f"{name}_meta.json").read_text(encoding="utf-8"))
    return index, meta


def reconstruct_all(index) -> np.ndarray:
    try:
        return index.reconstruct_n(0, index.ntotal)
    except RuntimeError:
        return np.vstack([index.reconstruct(i) for i in range(index.ntotal)])


def build_split(captions_meta: list, docs_per_image: int | None = None) -> tuple[list, list]:
    """
    Делит подписи на запросы и документы: первая подпись снимка — запрос,
    остальные — документы. Снимки с единственной подписью выбрасываем целиком:
    для них у каптион-пути не было бы ни одного документа, и сравнение
    оказалось бы нечестным в пользу CLIP.

    docs_per_image ограничивает число документов на снимок. Это не тонкая
    настройка, а главная проверка замысла: BLIP сгенерирует РОВНО ОДНУ подпись,
    а у COCO их четыре, и максимум по четырём подписям даёт каптион-пути четыре
    попытки вместо одной. Без этого ограничения замер льстит идее.
    """
    by_image = defaultdict(list)
    for row, item in enumerate(captions_meta):
        by_image[str(item["image_id"])].append((row, item["caption"]))

    queries, documents = [], []
    for image_id, items in by_image.items():
        if len(items) < 2:
            continue
        head, tail = items[0], items[1:]
        if docs_per_image is not None:
            tail = tail[:docs_per_image]
        queries.append({"image_id": image_id, "row": head[0], "text": head[1]})
        for row, text in tail:
            documents.append({"image_id": image_id, "row": row, "text": text})
    return queries, documents


def ranks_from_scores(scores: np.ndarray, gt_columns: np.ndarray) -> np.ndarray:
    """
    Позиция правильного снимка в выдаче (0-based) для каждой строки оценок.
    Считается без полной сортировки: ранг — это число кандидатов, обошедших
    правильный ответ по оценке.
    """
    gt_scores = scores[np.arange(len(scores)), gt_columns][:, None]
    return (scores > gt_scores).sum(axis=1)


def recall_at_k(ranks: np.ndarray, ks=DEFAULT_KS) -> dict:
    if len(ranks) == 0:
        return {k: float("nan") for k in ks}
    return {k: float((ranks < k).mean()) for k in ks}


def slice_masks(queries: list) -> "OrderedDict[str, np.ndarray]":
    texts = [q["text"] for q in queries]
    lengths = np.array([len(t.split()) for t in texts])
    long_cut = np.percentile(lengths, 75)
    short_cut = np.percentile(lengths, 25)

    masks = OrderedDict()
    masks["все запросы"] = np.ones(len(texts), dtype=bool)
    masks[f"длинные (>{long_cut:.0f} слов)"] = lengths > long_cut
    masks[f"короткие (<{short_cut:.0f} слов)"] = lengths < short_cut
    masks["с отношениями"] = np.array([bool(RELATIONAL.search(t)) for t in texts])
    masks["со счётом"] = np.array([bool(COUNTING.search(t)) for t in texts])
    masks["с отрицанием"] = np.array([bool(NEGATION.search(t)) for t in texts])
    return masks


def main():
    parser = argparse.ArgumentParser(description="C0: польза поиска через подписи")
    parser.add_argument("--index_dir", default="index")
    parser.add_argument("--limit", type=int, default=None,
                        help="взять только первые N запросов (быстрая проверка)")
    parser.add_argument("--model", default=SBERT_MODEL)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--docs_per_image", type=int, default=None,
                        help="сколько подписей на снимок класть в индекс; 1 = как у BLIP")
    parser.add_argument("--output", default=None)
    parser.add_argument("--cache", default=None, help="куда класть эмбеддинги SBERT")
    args = parser.parse_args()

    index_dir = Path(args.index_dir)
    ks = DEFAULT_KS

    images_index, images_meta = load_index(index_dir, "images")
    captions_index, captions_meta = load_index(index_dir, "captions")
    print(f"Индекс: {images_index.ntotal} снимков, {captions_index.ntotal} подписей")

    queries, documents = build_split(captions_meta, args.docs_per_image)
    if args.limit:
        queries = queries[: args.limit]
    per_image = args.docs_per_image or "все"
    print(f"Запросов: {len(queries)}, документов-подписей: {len(documents)} "
          f"(на снимок: {per_image})")

    # Пул кандидатов — только снимки, у которых есть хотя бы одна подпись-документ.
    # Обе системы ищут в одном и том же множестве, иначе сравнивать нечего.
    pool_ids = sorted({d["image_id"] for d in documents})
    pool_position = {image_id: i for i, image_id in enumerate(pool_ids)}
    image_row = {str(item["image_id"]): row for row, item in enumerate(images_meta)}

    missing = [i for i in pool_ids if i not in image_row]
    if missing:
        sys.exit(f"В images_meta нет снимков: {missing[:5]} — индекс несогласован")

    gt_columns = np.array([pool_position[q["image_id"]] for q in queries])
    print(f"Пул кандидатов: {len(pool_ids)} снимков")

    # --- путь 1: CLIP -------------------------------------------------------
    print("\nCLIP: беру готовые векторы из индексов...")
    caption_vectors = reconstruct_all(captions_index)
    image_vectors = reconstruct_all(images_index)

    query_clip = caption_vectors[[q["row"] for q in queries]]
    pool_clip = image_vectors[[image_row[i] for i in pool_ids]]
    clip_scores = query_clip @ pool_clip.T
    print(f"  матрица оценок: {clip_scores.shape}")

    # --- путь 2: SBERT ------------------------------------------------------
    from sentence_transformers import SentenceTransformer

    print(f"\nSBERT: {args.model}")
    t0 = time.perf_counter()
    model = SentenceTransformer(args.model)
    print(f"  модель загружена за {time.perf_counter() - t0:.1f} с")

    # В кэш кладём номера строк вместе с векторами: набор документов зависит от
    # --docs_per_image, и молча переиспользовать чужой кэш означало бы сравнивать
    # запросы с подписями от другого прогона.
    cache = Path(args.cache) if args.cache else None
    doc_texts = [d["text"] for d in documents]
    doc_rows = np.array([d["row"] for d in documents])
    if cache and cache.exists():
        stored = np.load(cache)
        if np.array_equal(stored["rows"], doc_rows):
            doc_sbert = stored["vectors"]
            print(f"  эмбеддинги документов взяты из кэша: {cache}")
        else:
            print("  кэш не подходит под этот набор документов — считаю заново")
            doc_sbert = None
    else:
        doc_sbert = None

    if doc_sbert is None:
        t0 = time.perf_counter()
        doc_sbert = model.encode(
            doc_texts, batch_size=args.batch, normalize_embeddings=True,
            show_progress_bar=True, convert_to_numpy=True,
        ).astype(np.float32)
        print(f"  {len(doc_texts)} подписей закодировано за {time.perf_counter() - t0:.0f} с")
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache, vectors=doc_sbert, rows=doc_rows)

    query_sbert = model.encode(
        [q["text"] for q in queries], batch_size=args.batch,
        normalize_embeddings=True, show_progress_bar=True, convert_to_numpy=True,
    ).astype(np.float32)

    # Оценка снимка = максимум по его подписям. Документы заранее выстроены в
    # порядке пула, поэтому максимум по снимку берётся групповой свёрткой
    # reduceat: поэлементное np.maximum.at на таком объёме работает минутами.
    order = np.argsort([pool_position[d["image_id"]] for d in documents], kind="stable")
    doc_sbert = doc_sbert[order]
    sorted_columns = np.array([pool_position[documents[i]["image_id"]] for i in order])
    starts = np.flatnonzero(np.r_[True, sorted_columns[1:] != sorted_columns[:-1]])
    if len(starts) != len(pool_ids):
        sys.exit("Не у каждого снимка пула есть подпись-документ — замер был бы кривым")

    # Полная матрица запрос x подпись заняла бы сотни мегабайт, поэтому по частям.
    sbert_scores = np.empty((len(queries), len(pool_ids)), dtype=np.float32)
    step = 512
    for start in range(0, len(queries), step):
        chunk = query_sbert[start:start + step] @ doc_sbert.T
        sbert_scores[start:start + step] = np.maximum.reduceat(chunk, starts, axis=1)
    print(f"  матрица оценок: {sbert_scores.shape}")

    # --- путь 3: гибрид -----------------------------------------------------
    # Оценки обеих систем — косинусы, но с разным разбросом: у CLIP они плотно
    # сидят около 0.2-0.3, у SBERT растянуты. Складывать их как есть означало бы
    # отдать вес тому, у кого шире шкала, поэтому приводим каждую строку к
    # нулевому среднему и единичному разбросу.
    def zscore(matrix):
        mean = matrix.mean(axis=1, keepdims=True)
        std = matrix.std(axis=1, keepdims=True) + 1e-9
        return (matrix - mean) / std

    clip_z, sbert_z = zscore(clip_scores), zscore(sbert_scores)

    # Вес подбирается на одной половине запросов, отчёт считается на другой.
    # Иначе получилось бы, что параметр настроен по тем же данным, на которых
    # потом объявляется результат, — а это завышает выигрыш гибрида просто так.
    half = len(queries) // 2
    tune_idx = np.arange(half)
    eval_idx = np.arange(half, len(queries))

    print("\nПодбираю вес гибрида (на первой половине запросов)...")
    best_alpha, best_recall = None, -1.0
    sweep = {}
    for alpha in [round(a * 0.1, 1) for a in range(11)]:
        fused = alpha * clip_z[tune_idx] + (1 - alpha) * sbert_z[tune_idx]
        r5 = recall_at_k(ranks_from_scores(fused, gt_columns[tune_idx]), ks)[5]
        sweep[alpha] = r5
        if r5 > best_recall:
            best_alpha, best_recall = alpha, r5
    print("  Recall@5 по весу CLIP: " + "  ".join(f"{a}:{v * 100:.1f}" for a, v in sweep.items()))
    print(f"  лучший вес CLIP = {best_alpha} (подобран на {half} запросах)")
    print(f"  отчёт ниже — на отложенных {len(eval_idx)} запросах")

    gt_eval = gt_columns[eval_idx]
    systems = OrderedDict()
    systems["clip"] = ranks_from_scores(clip_scores[eval_idx], gt_eval)
    systems["sbert"] = ranks_from_scores(sbert_scores[eval_idx], gt_eval)
    systems[f"hybrid a={best_alpha}"] = ranks_from_scores(
        best_alpha * clip_z[eval_idx] + (1 - best_alpha) * sbert_z[eval_idx], gt_eval
    )

    # --- отчёт --------------------------------------------------------------
    masks = slice_masks([queries[i] for i in eval_idx])
    print("\n" + "=" * 78)
    print(f"{'срез':<26}{'система':<16}{'N':>6}   R@1     R@5     R@10")
    print("-" * 78)
    report = {}
    for label, mask in masks.items():
        report[label] = {"n": int(mask.sum())}
        for name, ranks in systems.items():
            recall = recall_at_k(ranks[mask], ks)
            report[label][name] = {f"recall@{k}": v for k, v in recall.items()}
            print(f"{label:<26}{name:<16}{int(mask.sum()):>6}"
                  f"{recall[1] * 100:>7.2f}{recall[5] * 100:>8.2f}{recall[10] * 100:>8.2f}")
        print("-" * 78)

    if args.output:
        payload = {
            "model_sbert": args.model,
            "queries": len(queries),
            "documents": len(documents),
            "pool": len(pool_ids),
            "best_alpha_clip": best_alpha,
            "alpha_sweep_recall5": sweep,
            "slices": report,
        }
        Path(args.output).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Отчёт сохранён: {args.output}")


if __name__ == "__main__":
    main()
