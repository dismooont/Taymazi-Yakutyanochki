"""Статус фоновых задач: опрос прогресса и отмена."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from web import db
from web.deps import CurrentUser
from web.jobs import job_queue
from web.schemas import JobOut

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


def _owned_job(job_id: str, user: dict) -> dict:
    job = db.get_job(job_id)
    if job is None or job["user_id"] != user["id"]:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Задача не найдена")
    return job


@router.get("", response_model=list[JobOut])
def list_jobs(user: CurrentUser, database_id: str | None = None, limit: int = 20) -> list[JobOut]:
    """
    История задач. Нужна фронту после перезагрузки страницы: пользователь мог уйти
    во время индексации и вернуться, прогресс должен найтись сам.
    """
    rows = db.list_jobs(user["id"], database_id=database_id, limit=max(1, min(limit, 100)))
    return [JobOut.from_row(row, db.queue_position(row["id"])) for row in rows]


@router.get("/{job_id}", response_model=JobOut)
def get_job(job_id: str, user: CurrentUser) -> JobOut:
    job = _owned_job(job_id, user)
    return JobOut.from_row(job, db.queue_position(job_id))


@router.post("/{job_id}/cancel", response_model=JobOut)
def cancel_job(job_id: str, user: CurrentUser) -> JobOut:
    job = _owned_job(job_id, user)
    if job["status"] not in db.ACTIVE_JOB_STATUSES:
        raise HTTPException(status.HTTP_409_CONFLICT, "Задача уже завершена")

    job_queue.cancel(job_id)
    # статус меняет воркер, когда дойдёт до ближайшей проверки: отменять «сразу»
    # означало бы бросать индекс в середине записи
    return JobOut.from_row(db.get_job(job_id), db.queue_position(job_id))
