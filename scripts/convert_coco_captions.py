"""
Конвертирует стандартный аннотационный файл MS COCO (captions_val2017.json
или captions_train2017.json) в плоский CSV с колонками image_id, caption_en,
который ожидает clip_zero_shot_search.py на входе команды `build`.

Использование:
    python scripts/convert_coco_captions.py \
        --input data/annotations/captions_val2017.json \
        --output data/captions.csv
"""

import argparse
import json

import pandas as pd


def convert(input_path: str, output_path: str):
    with open(input_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    df = pd.DataFrame(coco["annotations"])[["image_id", "caption"]]
    df = df.rename(columns={"caption": "caption_en"})
    df.to_csv(output_path, index=False)

    print(f"Сохранено {len(df)} подписей для {df['image_id'].nunique()} уникальных изображений")
    print(f"-> {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Конвертация COCO captions.json -> captions.csv")
    parser.add_argument("--input", required=True, help="путь к captions_val2017.json / captions_train2017.json")
    parser.add_argument("--output", required=True, help="путь для сохранения captions.csv")
    args = parser.parse_args()
    convert(args.input, args.output)


if __name__ == "__main__":
    main()
