# CLIP Zero-Shot Image ⇄ Text Search

Прототип семантического поиска "картинка по тексту" и "текст/похожие картинки по картинке"
на базе CLIP (`openai/clip-vit-base-patch32`) и FAISS, на датасете MS COCO. Без дообучения —
используются готовые предобученные веса CLIP (zero-shot).

Поддерживается перевод текстовых запросов с русского на английский (CLIP обучен на английских
подписях, поэтому русский запрос сначала переводится).

Система поддерживает **инкрементальное добавление новых изображений** в уже построенный индекс
(команда `add`) — без пересчёта эмбеддингов для всего датасета заново. Это ключевое требование
для сценария "пользователь присылает своё фото" (в том числе для будущего Telegram-бота, см.
раздел 9) — индекс растёт по мере поступления новых изображений, а не пересобирается с нуля
каждый раз.

---

## 1. Минимальные системные требования

| Параметр            | Минимум                                   | Рекомендуется                          |
|----------------------|--------------------------------------------|-----------------------------------------|
| ОС                   | Windows 10/11, macOS 12+, Linux (любой современный дистрибутив) | — |
| Python               | 3.9                                        | 3.10–3.11                               |
| RAM                  | 8 ГБ                                       | 16 ГБ (для датасета в несколько тысяч изображений) |
| Диск                 | ~4 ГБ свободно (веса модели + подмножество COCO + индекс) | 10+ ГБ, если брать больший датасет |
| GPU                  | Не обязателен (работает на CPU)            | NVIDIA GPU с CUDA, ≥4 ГБ VRAM — ускоряет построение эмбеддингов в 5–10 раз |
| Интернет             | Нужен при первом запуске (скачивание весов модели ~600 МБ и/или датасета COCO) | — |

**Важно про Python 3.13**: на момент написания README некоторые версии `torch`/`faiss-cpu`
ещё не имеют стабильных бинарных сборок под 3.13 на всех платформах. Если возникают проблемы
с установкой зависимостей — используйте Python 3.10 или 3.11.

**Про размер данных**: полное подмножество val2017 MS COCO — это ~5000 изображений (~1 ГБ архив
изображений + ~240 МБ аннотаций). Для быстрых экспериментов можно ограничиться меньшим
подмножеством (см. раздел 5).

---

## 2. Структура проекта

```
clip_zero_shot_search/
├── .idea/                          # конфигурация PyCharm (интерпретатор, run-конфигурации)
│   └── runConfigurations/          # готовые конфигурации запуска: build / text / image / add
├── src/
│   └── clip_zero_shot_search.py    # основной CLI-скрипт (build / add / text / image)
├── bot/                            # Telegram-бот (см. раздел 9)
│   ├── inference.py                # резидентная обёртка SearchEngine (модель+индекс в памяти)
│   ├── bot.py                      # бот на aiogram, long polling
│   └── requirements.txt            # зависимости бота (без matplotlib; torch — из Dockerfile)
├── scripts/
│   ├── convert_coco_captions.py    # конвертация captions_val2017.json -> captions.csv
│   ├── download_coco_val2017.sh    # скачивание датасета val2017 (Linux/macOS/Colab)
│   └── download_coco_val2017.ps1   # то же под Windows (PowerShell)
├── notebooks/
│   └── colab_reference.ipynb       # исходный Colab-ноутбук (референс, история разработки)
├── data/
│   ├── images/                     # сюда кладутся *.jpg (пусто в архиве, см. раздел 5)
│   ├── user_photos/                # присланные боту фото (создаётся автоматически)
│   └── captions.csv                # CSV: image_id, caption_en (создаётся скриптом конвертации)
├── index/                          # сюда сохраняется построенный FAISS-индекс (пусто в архиве)
├── Dockerfile                      # моно-контейнер: бот + CLIP-инференс (раздел 9)
├── docker-compose.yml              # запуск бота с volume для index/data
├── .env.example                    # шаблон .env (TELEGRAM_BOT_TOKEN)
├── requirements.txt                # зависимости CLI-скрипта
├── .gitignore
└── README.md
```

---

## 3. Установка (локально, вне Colab)

### 3.1. Клонируйте/распакуйте проект и создайте виртуальное окружение

```bash
cd clip_zero_shot_search
python3 -m venv .venv

# Linux / macOS
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

### 3.2. Установите зависимости

```bash
pip install -r requirements.txt
```

Если у вас NVIDIA GPU и нужна GPU-версия PyTorch — сначала поставьте `torch` отдельно
по инструкции с [pytorch.org](https://pytorch.org/get-started/locally/) под вашу версию CUDA,
и только потом остальные зависимости из `requirements.txt` (без переустановки torch).

### 3.3. Откройте проект в PyCharm

`File → Open` → выбрать папку `clip_zero_shot_search`. PyCharm подхватит `.idea/`, но
интерпретатор нужно будет указать вручную при первом открытии:

`File → Settings → Project → Python Interpreter → Add Interpreter → Existing`
→ указать `.venv/bin/python` (или `.venv\Scripts\python.exe` на Windows), созданный на шаге 3.1.

После этого в выпадающем списке Run Configurations (верхняя панель) появятся три готовые
конфигурации:
- **1. build_index** — построение индекса
- **2. text_search** — пример поиска по тексту
- **3. image_search** — пример поиска по картинке

Их параметры (`--query`, `--image_path` и т.д.) можно редактировать через
`Run → Edit Configurations...`.

---

## 4. Установка и запуск в Google Colab

Актуальный, проверенный порядок ячеек (все "грабли" ниже уже учтены и исправлены в
`src/clip_zero_shot_search.py`):

```python
# 1. Зависимости (без googletrans! см. раздел 6 "Известные проблемы")
!pip install transformers faiss-cpu pillow pandas tqdm matplotlib deep-translator --quiet
```

```python
# 2. Загрузить скрипт
from google.colab import files
uploaded = files.upload()   # выбрать src/clip_zero_shot_search.py
```

```python
# 3. Скачать датасет (или загрузить свой архив — см. раздел 5)
!wget -q http://images.cocodataset.org/zips/val2017.zip
!wget -q http://images.cocodataset.org/annotations/annotations_trainval2017.zip
!unzip -q val2017.zip -d /content/data/images
!unzip -q annotations_trainval2017.zip -d /content/data
```

```python
# 4. Конвертировать аннотации COCO в captions.csv
import json
import pandas as pd

with open('/content/data/annotations/captions_val2017.json') as f:
    coco = json.load(f)

df = pd.DataFrame(coco['annotations'])[['image_id', 'caption']]
df = df.rename(columns={'caption': 'caption_en'})
df.to_csv('/content/data/captions.csv', index=False)
```

```python
# 5. Построить индекс
!python clip_zero_shot_search.py build \
    --images_dir /content/data/images \
    --captions_csv /content/data/captions.csv \
    --index_dir /content/index
```

```python
# 6. Поиск по тексту (с переводом RU->EN)
!python clip_zero_shot_search.py text \
    --index_dir /content/index \
    --query "собака в снегу" \
    --translate \
    --top_k 5

from IPython.display import Image as IPImage, display
display(IPImage('/content/index/search_result_text.png'))
```

```python
# 7. Поиск по картинке (сначала загрузить query.jpg через files.upload())
!python clip_zero_shot_search.py image \
    --index_dir /content/index \
    --image_path /content/query.jpg \
    --top_k 5

display(IPImage('/content/index/search_result_image.png'))
```

---

## 5. Подготовка датасета

Скрипту `build` нужны:
1. Папка с изображениями (`.jpg`/`.jpeg`/`.png`), допускаются вложенные подпапки
   (например `images/val2017/*.jpg` — скрипт ищет рекурсивно).
2. CSV-файл с колонками `image_id`, `caption_en` (опционально `caption_ru`).
   Один `image_id` может встречаться в нескольких строках (в MS COCO — 5 подписей на картинку).

**Формат имён файлов**: скрипт понимает стандартный для MS COCO формат с ведущими нулями
до 12 цифр (`image_id=139` → `000000000139.jpg`), а также имена файлов без паддинга
(`139.jpg`) как запасной вариант.

### Вариант А — скачать официальный MS COCO val2017

Linux / macOS / Colab:
```bash
bash scripts/download_coco_val2017.sh data
python scripts/convert_coco_captions.py \
    --input data/annotations/captions_val2017.json \
    --output data/captions.csv
```

Windows (PowerShell) — скрипт с докачкой и автоповтором при обрыве соединения:
```powershell
powershell -ExecutionPolicy Bypass -File scripts\download_coco_val2017.ps1 -DestDir data
.\.venv\Scripts\python.exe scripts\convert_coco_captions.py `
    --input data\annotations\captions_val2017.json `
    --output data\captions.csv
```

### Вариант Б — своё подмножество
Просто положите изображения в `data/images/` и подготовьте `data/captions.csv` вручную
или скриптом с той же логикой, что в `scripts/convert_coco_captions.py`.

---

## 6. Использование (после установки и подготовки данных)

```bash
# Построение индекса (один раз; пересоздавать нужно только если датасет поменялся)
python src/clip_zero_shot_search.py build \
    --images_dir data/images \
    --captions_csv data/captions.csv \
    --index_dir index

# Поиск изображений по английскому тексту
python src/clip_zero_shot_search.py text \
    --index_dir index \
    --query "a dog playing in the snow" \
    --top_k 5

# Поиск изображений по русскому тексту (с автопереводом)
python src/clip_zero_shot_search.py text \
    --index_dir index \
    --query "собака играет в снегу" \
    --translate \
    --top_k 5

# Поиск похожих изображений и релевантных подписей по картинке-запросу
python src/clip_zero_shot_search.py image \
    --index_dir index \
    --image_path path/to/query.jpg \
    --top_k 5
```

После каждого поиска сохраняется коллаж с найденными картинками:
`index/search_result_text.png` или `index/search_result_image.png`
(отключается флагом `--no_plot`).

### 6.1. Добавление новых изображений в уже существующий индекс

Команда `add` добавляет новые фото в индекс, не пересчитывая эмбеддинги для уже
проиндексированных изображений — важно как при регулярном пополнении датасета, так и как
основа для будущего Telegram-бота (раздел 9), где пользователь может прислать своё фото.

**Режим 1 — без CSV (типичный сценарий "пользователь прислал произвольное фото"):**
`image_id` берётся из имени файла без расширения. Подходит для фото без подписей —
они попадут в `images.index` (доступны для поиска "похожие картинки" и "картинка по тексту"),
но не появятся в результатах "картинка → подписи", так как подписи для них не заданы.

```bash
python src/clip_zero_shot_search.py add \
    --images_dir data/new_photos \
    --index_dir index
```

**Режим 2 — с CSV (пополнение датасета новыми размеченными изображениями):**
работает как `build`, но добавляет только новые записи поверх существующего индекса,
обновляя и `images.index`, и `captions.index`.

```bash
python src/clip_zero_shot_search.py add \
    --images_dir data/new_photos \
    --captions_csv data/new_captions.csv \
    --index_dir index
```

Повторный запуск `add` с теми же файлами безопасен — уже проиндексированные `image_id`
(с учётом ведущих нулей в имени файла, например `000000000139.jpg` и `139` считаются
одним и тем же id) автоматически пропускаются, дубликаты не создаются.

---

## 7. Известные проблемы и их решения (из истории разработки)

Эти проблемы уже исправлены в текущей версии `src/clip_zero_shot_search.py`, но фиксирую их
здесь на случай похожих ошибок в будущем (например, при обновлении зависимостей):

| Проблема | Причина | Решение (уже в коде) |
|---|---|---|
| `AttributeError: module 'httpx' has no attribute 'TimeoutException'` | `googletrans==4.0.0rc1` требует старый `httpx==0.13.3`, конфликтует с `huggingface_hub` | Заменили перевод на `deep-translator` (не пинит старый httpx). **Не устанавливайте `googletrans`.** |
| Перевод возвращал тот же русский текст ("рандомные картинки" в результатах) | Неудачный перевод кэшировался как fallback, и следующие вызовы читали испорченный кэш | Функция `translate_ru_to_en` больше не кэширует неудачные попытки |
| `ValueError: need at least one array to concatenate` при `build` | `image_id_to_filename` не находила файлы — либо неверный формат имени, либо изображения лежат во вложенной подпапке (`images/val2017/*.jpg`) | Поиск теперь рекурсивный (по всем подпапкам) и понимает zero-padded формат имён MS COCO |
| `AttributeError: 'BaseModelOutputWithPooling' object has no attribute 'cpu'` | В `transformers >= 5.0` изменилось поведение `get_image_features`/`get_text_features` — возвращают объект, а не тензор напрямую | Добавлена функция `extract_features_tensor`, совместимая с обеими версиями API |
| Долгий поиск файлов на большом датасете | `rglob` вызывался заново на каждый `image_id` | Дерево папок теперь сканируется один раз (`build_filename_index`), дальше поиск по словарю |
| Команда `add` добавляла дубликаты, если новый файл назван с ведущими нулями (`000000000139.jpg`), а в индексе тот же снимок хранится как `image_id="139"` (без паддинга, как приходит из CSV) | Сравнение id "как есть" не учитывало разный формат представления одного и того же числового id | Добавлена функция `normalize_id` — числовые id сравниваются без ведущих нулей независимо от формата исходного имени файла |

Если после обновления зависимостей `pip install -r requirements.txt` появляются новые
конфликты — самый надёжный первый шаг: создать чистое виртуальное окружение
(`python -m venv .venv` заново) и установить зависимости с нуля, не поверх старого окружения.

---

## 8. Возможные следующие шаги

- **Telegram-бот в Docker** — ✅ реализован (`bot/`, `Dockerfile`, `docker-compose.yml`);
  архитектура и инструкции запуска — в разделе 9.
- **Fine-tuning CLIP** на своём подмножестве COCO (contrastive/InfoNCE loss) для повышения
  Recall@k выше zero-shot базового уровня — см. обсуждение архитектуры и функции потерь
  в истории проекта / сопроводительном документе.
- Замена `IndexFlatIP` в FAISS на приближённый индекс (`IndexIVFFlat`, `IndexHNSWFlat`) при
  росте датасета выше ~100k изображений — точный поиск станет медленным.
- Подсчёт метрик Recall@1/5/10 на валидационном подмножестве для количественной оценки качества
  (см. пример кода в истории чата — сравнение найденных `image_id` с эталонными).
- Веб-интерфейс (Streamlit/Gradio) поверх текущего CLI для демонстрации на защите проекта.
- Кэш эмбеддингов текста — сейчас пересчитывается при каждом запуске `text`, для многократных
  запросов на неизменном индексе можно закэшировать модель между вызовами (например, через
  Python API вместо перезапуска процесса на каждый запрос).

---

## 9. Telegram-бот и Docker

Текущий пайплайн обёрнут в Telegram-бота и упакован в Docker так, чтобы разворачивать
можно было на любом хостинге без привязки к конкретному облаку. Ниже — архитектура,
инструкции запуска и обоснование принятых решений.

### 9.0. Быстрый запуск

Предполагается, что индекс уже построен (раздел 6) — папка `index/` не пустая.

1. Получите токен у [@BotFather](https://t.me/BotFather) и создайте `.env`:
   ```bash
   cp .env.example .env      # Windows: copy .env.example .env
   # впишите TELEGRAM_BOT_TOKEN=...
   ```

2. **Локально** (для отладки; модель берётся из кэша HuggingFace):
   ```bash
   pip install -r bot/requirements.txt      # aiogram и пр. (torch уже стоит из requirements.txt)
   python -m bot.bot
   ```
   Если провайдер режет `api.telegram.org` — задайте в `.env` `TELEGRAM_PROXY=http://127.0.0.1:10809`
   (бот будет ходить в Telegram через прокси). Чтобы `transformers` не лез в сеть за уже
   скачанной моделью, запускайте с `HF_HUB_OFFLINE=1` (Windows PowerShell: `$env:HF_HUB_OFFLINE=1`).

3. **В Docker** (рекомендуется для деплоя):
   ```bash
   docker compose up -d --build
   docker compose logs -f bot                # смотреть логи
   ```
   Веса CLIP кэшируются в образ на этапе сборки — контейнер стартует офлайн.
   Индекс и изображения монтируются как volume (см. 9.5), поэтому присланные фото
   и рост индекса переживают пересборку контейнера.

Что умеет бот:
- **текст** (в т.ч. по-русски — авто-перевод RU→EN) → присылает подходящие картинки;
- **фото** → сохраняет в `data/user_photos/`, инкрементально добавляет в индекс
  (`SearchEngine.add_image`, без пересборки) и присылает похожие изображения + релевантные подписи.

### 9.1. Архитектура: моно-контейнер vs разделение на сервисы

**Вариант А — всё в одном контейнере (выбран, реализован).**
Бот и CLIP-инференс живут в одном образе. В отличие от CLI-скрипта, где модель
загружается заново на каждый вызов процесса, бот держит модель и FAISS-индекс
**резидентно в памяти** — загружает один раз при старте и переиспользует на каждое
сообщение. Это вынесено в класс `SearchEngine` (`bot/inference.py`), вызываемый напрямую
из кода бота через Python API (не через `subprocess`/CLI) — иначе каждый ответ занимал бы
секунды только на повторную загрузку весов. `SearchEngine` переиспользует низкоуровневые
функции CLI-скрипта (эмбеддинги, перевод, разбор `image_id`), не дублируя логику.

- Плюсы: один `docker run`, минимум инфраструктуры, идеально для CPU-инференса на CLIP ViT-B/32.
- Минусы: нельзя масштабировать бота и инференс независимо друг от друга.

**Вариант Б — два сервиса через `docker-compose` (если бот вырастет).**
`bot` (только Telegram API + очередь сообщений) ↔ `inference` (FastAPI-сервис с CLIP +
FAISS, отдаёт эмбеддинги/результаты поиска по HTTP). Оправдан, если инференс нужно вынести
на отдельную машину с GPU или переиспользовать API для чего-то ещё (например, веб-демо).

Для объёма и характера задачи (CPU-инференс на маленькой модели, отклик на один запрос —
доли секунды) выбран **вариант А** как первый шаг; переход к варианту Б — по мере роста
нагрузки.

### 9.2. Long polling vs webhook

- **Long polling** — бот сам стучится в Telegram API, не нужен ни открытый порт, ни белый
  IP, ни SSL-сертификат, ни домен. Работает вообще где угодно, даже за NAT.
- **Webhook** — Telegram стучится к боту, требует публичный HTTPS-адрес (домен + сертификат
  либо туннель типа ngrok/Cloudflare Tunnel). Ниже задержка, но больше инфраструктуры.

**Выбор: long polling** — соответствует цели "разворачивать где угодно" без домена и прокси;
разница в задержке (десятки миллисекунд) не критична для этой задачи.

### 9.3. Структура (фактическая)

```
clip_zero_shot_search/
├── bot/
│   ├── bot.py              # aiogram 3.x, long polling
│   ├── inference.py        # SearchEngine: модель и индекс грузятся один раз при старте
│   └── requirements.txt    # aiogram, python-dotenv, transformers, faiss-cpu (torch — из Dockerfile)
├── src/clip_zero_shot_search.py   # переиспользуется как модуль (build/add/search-функции)
├── index/                  # готовый FAISS-индекс — монтируется как volume
├── data/images/            # изображения COCO — монтируются (бот отдаёт файлы по путям из индекса)
├── data/user_photos/       # присланные пользователем фото — монтируются
├── Dockerfile
├── docker-compose.yml
├── .env.example            # шаблон; скопировать в .env и вписать токен
└── .env                    # TELEGRAM_BOT_TOKEN (в .gitignore, не коммитить!)
```

### 9.4. Dockerfile (вариант А)

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# torch CPU-only сборка — образ в разы легче, чем с CUDA-зависимостями
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY bot/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY bot/ bot/

ENV PYTHONUNBUFFERED=1
CMD ["python", "bot/bot.py"]
```

### 9.5. docker-compose.yml

Индекс монтируется как volume, а не запекается в образ — так `add` (в том числе вызванный
ботом при получении нового фото от пользователя) сразу сохраняется на диск хоста и переживает
пересборку контейнера.

```yaml
services:
  bot:
    build: .
    restart: unless-stopped
    env_file: .env
    volumes:
      - ./index:/app/index
      - ./data/user_photos:/app/data/user_photos   # сюда бот сохраняет присланные фото
```

Деплой на любом сервере с Docker сводится к:
```bash
docker compose up -d --build
```

### 9.6. Как в эту схему ложится команда `add`

Ради этого команда `add` (раздел 6.1) и была спроектирована как переиспользуемая логика,
а не только как CLI-обёртка: когда пользователь присылает боту фото, бот сохраняет файл в
`data/user_photos/`, вызывает `SearchEngine.add_image()` напрямую (модель уже загружена в
памяти процесса — не через `subprocess`, чтобы не платить за повторную загрузку весов) и
сразу отвечает на запрос "найди похожие" уже с учётом только что добавленного фото.
`image_id` для таких фото берётся из Telegram `file_unique_id` (режим 1 из раздела 6.1, без
CSV с подписями); повторная отправка того же фото не создаёт дубликат в индексе.
Тяжёлый CPU-инференс бот выполняет в отдельном потоке (`asyncio.to_thread`), чтобы не
блокировать event loop.

### 9.7. Варианты хостинга готового Docker-образа

| Вариант | Стоимость | Сложность | Комментарий |
|---|---|---|---|
| Свой VPS (Timeweb, Selectel, REG.RU, Hetzner) | ~200–500 руб/мес | Низкая | `docker compose up -d`, самый предсказуемый вариант |
| Yandex Cloud / VK Cloud | Есть гранты для студентов | Низкая-средняя | Если нужно "российское облако" в отчёте по проекту |
| Railway / Render / Fly.io | Бесплатный тир, потом ~$5/мес | Низкая | Пуш Docker-образа, разворачивают сами; удобно для демо на защите |
| Домашний сервер / Raspberry Pi | Бесплатно (кроме электричества) | Средняя | С long polling работает без проброса портов |

Для демонстрации на защите проекта оптимален **Railway** (бесплатно, разворачивается за
несколько минут) либо дешёвый VPS, если нужна постоянная доступность после защиты.

---

## 10. Лицензия датасета и модели

- MS COCO: изображения и аннотации распространяются под лицензией Creative Commons
  Attribution 4.0 — см. [cocodataset.org/#termsofuse](https://cocodataset.org/#termsofuse).
- CLIP (`openai/clip-vit-base-patch32`): веса распространяются Hugging Face / OpenAI,
  см. карточку модели на [huggingface.co/openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32).
