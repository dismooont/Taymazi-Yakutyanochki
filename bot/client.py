"""
Клиент API для бота.

Бот ничего не знает ни про CLIP, ни про FAISS, ни про файлы индекса: он умеет только
разговаривать с API по HTTP. Поэтому его образ не тянет torch и весит 250 МБ
вместо двух гигабайт, а модель в системе существует ровно в одном экземпляре.
"""

from __future__ import annotations

from typing import Any

import httpx


class ApiError(RuntimeError):
    """Ошибка API с текстом, пригодным для показа в чате."""

    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class SearchApi:
    def __init__(self, base_url: str, service_token: str, timeout: float = 120.0):
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={"X-Service-Token": service_token},
            timeout=timeout,
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def _request(self, method: str, path: str, **kwargs) -> Any:
        response = await self._client.request(method, path, **kwargs)
        if response.status_code >= 400:
            detail = f"Ошибка {response.status_code}"
            try:
                body = response.json()
                if isinstance(body.get("detail"), str):
                    detail = body["detail"]
            except Exception:
                pass
            raise ApiError(response.status_code, detail)
        if response.headers.get("content-type", "").startswith("application/json"):
            return response.json()
        return response.content

    # ------------------------------------------------------------------

    async def start_chat(self, chat_id: int, telegram_user_id: int, display_name: str,
                         title: str) -> dict:
        return await self._request(
            "POST", f"/api/bot/chats/{chat_id}/start",
            data={
                "telegram_user_id": str(telegram_user_id),
                "display_name": display_name,
                "title": title,
            },
        )

    async def chat_info(self, chat_id: int) -> dict | None:
        try:
            return await self._request("GET", f"/api/bot/chats/{chat_id}")
        except ApiError as e:
            if e.status_code == 404:
                return None
            raise

    async def remember_member(self, chat_id: int, telegram_user_id: int, display_name: str) -> None:
        await self._request(
            "POST", f"/api/bot/chats/{chat_id}/members",
            data={"telegram_user_id": str(telegram_user_id), "display_name": display_name},
        )

    async def add_photo(self, chat_id: int, filename: str, content: bytes,
                        telegram_user_id: int, display_name: str) -> dict:
        return await self._request(
            "POST", f"/api/bot/chats/{chat_id}/photos",
            files={"file": (filename, content, "image/jpeg")},
            data={"telegram_user_id": str(telegram_user_id), "display_name": display_name},
        )

    async def search(self, chat_id: int, query: str, top_k: int = 5) -> dict:
        return await self._request(
            "POST", f"/api/bot/chats/{chat_id}/search",
            json={"query": query, "top_k": top_k, "translate": True},
        )

    async def similar(self, chat_id: int, photo_id: str, top_k: int = 5) -> dict:
        return await self._request(
            "GET", f"/api/bot/chats/{chat_id}/photos/{photo_id}/similar",
            params={"top_k": top_k},
        )

    async def photo_bytes(self, chat_id: int, photo_id: str) -> bytes:
        return await self._request("GET", f"/api/bot/chats/{chat_id}/photos/{photo_id}/file")

    # --- общая демо-база MS COCO (только чтение, доступна из любого чата) ---

    async def demo_info(self) -> dict | None:
        try:
            return await self._request("GET", "/api/bot/demo")
        except ApiError as e:
            if e.status_code == 404:
                return None
            raise

    async def search_demo(self, query: str, top_k: int = 5) -> dict:
        return await self._request(
            "POST", "/api/bot/demo/search",
            json={"query": query, "top_k": top_k, "translate": True},
        )

    async def demo_photo_bytes(self, photo_id: str) -> bytes:
        return await self._request("GET", f"/api/bot/demo/photos/{photo_id}/file")
