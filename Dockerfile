# Моно-контейнер: Telegram-бот + резидентный CLIP-инференс (README, раздел 9.4).
FROM python:3.11-slim

WORKDIR /app

# Прокси для сборки за "закрытой" сетью (необязательно). Прокидывается из
# docker-compose args; на обычном сервере с прямым интернетом оставить пустым.
ARG HTTP_PROXY=""
ARG HTTPS_PROXY=""
ENV HTTP_PROXY=${HTTP_PROXY} HTTPS_PROXY=${HTTPS_PROXY} \
    http_proxy=${HTTP_PROXY} https_proxy=${HTTPS_PROXY}

# torch CPU-only сборка — образ в разы легче, чем с CUDA-зависимостями.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY bot/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# core/ обязателен: после выделения ядра (docs/WEB_PLAN.md, этап M0) src/ и bot/
# импортируют его, и без этой строки контейнер падает на старте с ImportError.
COPY core/ core/
COPY src/ src/
COPY bot/ bot/

# Заранее кэшируем веса CLIP в образ, чтобы контейнер стартовал офлайн и быстро.
RUN python -c "from transformers import CLIPModel, CLIPProcessor; \
    CLIPModel.from_pretrained('openai/clip-vit-base-patch32'); \
    CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')"

# Прокси нужен был только на этапе сборки — на рантайме сбрасываем.
ENV HTTP_PROXY="" HTTPS_PROXY="" http_proxy="" https_proxy="" \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    INDEX_DIR=/app/index \
    USER_PHOTOS_DIR=/app/data/user_photos

CMD ["python", "-m", "bot.bot"]
