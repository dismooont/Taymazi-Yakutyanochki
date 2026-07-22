"""
Генерация изображения через YandexART (aistudio.yandex.ru), когда поиск ничего
не нашёл. REST API асинхронный: запрос ставит операцию в очередь, результат —
отдельным опросом (см. docs аистудии, раздел "Generating an image").

folder_id и api_key должны соответствовать одному и тому же сервисному аккаунту
— иначе запрос отклоняется на первом же шаге с понятной ошибкой (проверено
эмпирически: сервис сам называет в тексте ошибки настоящий folder_id ключа).
"""

from __future__ import annotations

import base64
import time
from dataclasses import dataclass

import requests

GENERATION_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/imageGenerationAsync"
OPERATION_URL = "https://operation.api.cloud.yandex.net/operations/{operation_id}"

REQUEST_TIMEOUT = 20.0
POLL_INTERVAL = 2.0
# Было 60 с — на практике сеть до operation.api.cloud.yandex.net временами
# держит read timeout по несколько раз подряд (проверено: 12 неудачных
# опросов подряд, 13-й прошёл). При REQUEST_TIMEOUT=20 с старое значение
# давало от силы 2-3 попытки, прежде чем сдаться. 180 с — тот же запас,
# что реально потребовался на практике, плюс небольшой резерв.
POLL_TIMEOUT = 180.0


class YandexArtError(RuntimeError):
    """Запрос отклонён или не завершился вовремя — вызывающий код решает, что делать дальше."""


@dataclass(frozen=True)
class GeneratedImage:
    content: bytes
    content_type: str = "image/jpeg"  # YandexART всегда отдаёт JPEG


def generate_image(prompt: str, *, api_key: str, folder_id: str, seed: int = 0) -> GeneratedImage:
    """
    Синхронный вызов: подходит для обработчика FastAPI (sync def уходит в
    threadpool сам) и для CLI/скриптов. Генерация в замерах занимала секунды —
    ждать её внутри запроса приемлемо, отдельная очередь не нужна.
    """
    headers = {"Authorization": f"Api-Key {api_key}", "Content-Type": "application/json"}
    body = {
        "modelUri": f"art://{folder_id}/yandex-art/latest",
        # seed=0 у YandexART — не "случайно", а конкретное фиксированное значение,
        # поэтому по умолчанию берём хеш запроса: одинаковый текст даёт одну и ту
        # же картинку, а не новую при каждом повторе.
        "generationOptions": {
            "seed": str(seed or (abs(hash(prompt)) % (2**31))),
            "aspectRatio": {"widthRatio": "1", "heightRatio": "1"},
        },
        "messages": [{"weight": "1", "text": prompt}],
    }

    try:
        response = requests.post(GENERATION_URL, headers=headers, json=body, timeout=REQUEST_TIMEOUT)
    except requests.RequestException as e:
        raise YandexArtError(f"Запрос к YandexART не удался: {e}") from e

    if not response.ok:
        raise YandexArtError(f"YandexART отклонил запрос ({response.status_code}): {response.text[:300]}")

    operation_id = response.json().get("id")
    if not operation_id:
        raise YandexArtError("YandexART не вернул id операции")

    return _poll(operation_id, headers)


def _poll(operation_id: str, headers: dict) -> GeneratedImage:
    """
    Разовый сетевой сбой одного опроса не должен хоронить всю генерацию: сама
    операция на стороне Yandex продолжает считаться, и следующий опрос вполне
    может её застать готовой. Сдаёмся только когда истёк общий POLL_TIMEOUT.
    """
    deadline = time.monotonic() + POLL_TIMEOUT
    url = OPERATION_URL.format(operation_id=operation_id)
    last_error: Exception | None = None

    while time.monotonic() < deadline:
        try:
            response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        except requests.RequestException as e:
            last_error = e
            time.sleep(POLL_INTERVAL)
            continue

        if not response.ok:
            last_error = YandexArtError(f"Опрос статуса генерации отклонён ({response.status_code})")
            time.sleep(POLL_INTERVAL)
            continue

        data = response.json()
        if data.get("error"):
            raise YandexArtError(f"YandexART сообщил об ошибке: {data['error']}")
        if data.get("done"):
            image_b64 = data.get("response", {}).get("image")
            if not image_b64:
                raise YandexArtError("Операция завершена, но снимка в ответе нет")
            return GeneratedImage(content=base64.b64decode(image_b64))

        time.sleep(POLL_INTERVAL)

    suffix = f": {last_error}" if last_error else ""
    raise YandexArtError(f"Генерация не завершилась за {POLL_TIMEOUT:.0f} с{suffix}")
