#!/usr/bin/env bash
# Скачивает и распаковывает MS COCO val2017 (изображения + аннотации).
# Использование:
#   bash scripts/download_coco_val2017.sh [папка_назначения]
# По умолчанию папка назначения — ./data

set -e

DEST_DIR="${1:-data}"
mkdir -p "$DEST_DIR/images"

echo "Скачивание изображений val2017 (~1 ГБ)..."
wget -q --show-progress -O /tmp/val2017.zip http://images.cocodataset.org/zips/val2017.zip

echo "Скачивание аннотаций (~240 МБ)..."
wget -q --show-progress -O /tmp/annotations_trainval2017.zip \
    http://images.cocodataset.org/annotations/annotations_trainval2017.zip

echo "Распаковка изображений..."
unzip -q /tmp/val2017.zip -d "$DEST_DIR/images"

echo "Распаковка аннотаций..."
unzip -q /tmp/annotations_trainval2017.zip -d "$DEST_DIR"

rm -f /tmp/val2017.zip /tmp/annotations_trainval2017.zip

echo ""
echo "Готово. Структура:"
echo "  $DEST_DIR/images/val2017/*.jpg"
echo "  $DEST_DIR/annotations/captions_val2017.json"
echo ""
echo "Дальше сконвертируйте аннотации в CSV:"
echo "  python scripts/convert_coco_captions.py \\"
echo "      --input $DEST_DIR/annotations/captions_val2017.json \\"
echo "      --output $DEST_DIR/captions.csv"
