#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="${1:-configs/baseline.yaml}"
PYTHON="${PYTHON:-python}"

if [[ ! -f "$CONFIG" ]]; then
    echo "Конфиг не найден: $CONFIG" >&2
    exit 1
fi

for f in data/train.csv data/test.csv; do
    if [[ ! -f "$f" ]]; then
        echo "Нет файла $f: скачайте данные соревнования nlp-getting-started в папку data/" >&2
        exit 1
    fi
done

echo "=== [1/2] Кросс-валидация и проверка гипотезы ($CONFIG) ==="
"$PYTHON" src/evaluate.py --config "$CONFIG"

echo
echo "=== [2/2] Обучение финальной модели и сабмит ==="
"$PYTHON" src/train.py --config "$CONFIG"

echo
echo "Готово: метрики в results/, модель и submission.csv в папке train.out из конфига."
