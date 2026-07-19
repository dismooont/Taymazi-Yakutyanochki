"""
Административные команды (запускать на сервере, где лежит data/app.db).

Самостоятельного восстановления пароля в приложении нет: письма со сбросом требуют SMTP
и подтверждения адреса (docs/WEB_PLAN.md, раздел 5). Поэтому сброс — здесь.

    python -m web.admin list-users
    python -m web.admin reset-password --login ivan --password <новый>
    python -m web.admin close-sessions --login ivan
"""

from __future__ import annotations

import argparse
import sys

from web import db
from web.config import get_settings
from web.security import AuthError, hash_password, normalize_login, validate_password


def cmd_list_users(_args) -> int:
    users = []
    with db.connect() as conn:
        rows = conn.execute(
            "SELECT u.id, u.login, u.display_name, u.created_at, u.last_seen_at,"
            " (SELECT COUNT(*) FROM databases d WHERE d.user_id = u.id) AS databases"
            " FROM users u ORDER BY u.created_at"
        ).fetchall()
        users = [dict(row) for row in rows]

    if not users:
        print("Пользователей нет")
        return 0
    print(f"{'логин':<20} {'баз':>4}  {'создан':<20} {'был':<20}")
    for user in users:
        print(f"{(user['login'] or '—'):<20} {user['databases']:>4}  "
              f"{user['created_at']:<20} {user['last_seen_at'] or '—':<20}")
    return 0


def cmd_reset_password(args) -> int:
    user = db.get_user_by_login(normalize_login(args.login))
    if user is None:
        print(f"Нет пользователя с логином {args.login}", file=sys.stderr)
        return 1
    try:
        validate_password(args.password, get_settings().min_password_length)
    except AuthError as e:
        print(str(e), file=sys.stderr)
        return 1

    db.set_password_hash(user["id"], hash_password(args.password))
    closed = db.delete_user_sessions(user["id"])
    print(f"Пароль изменён. Завершено сессий: {closed}")
    return 0


def cmd_close_sessions(args) -> int:
    user = db.get_user_by_login(normalize_login(args.login))
    if user is None:
        print(f"Нет пользователя с логином {args.login}", file=sys.stderr)
        return 1
    print(f"Завершено сессий: {db.delete_user_sessions(user['id'])}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Администрирование веб-приложения")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list-users", help="список пользователей")

    p_reset = sub.add_parser("reset-password", help="задать пользователю новый пароль")
    p_reset.add_argument("--login", required=True)
    p_reset.add_argument("--password", required=True)

    p_close = sub.add_parser("close-sessions", help="завершить все сессии пользователя")
    p_close.add_argument("--login", required=True)

    args = parser.parse_args()
    db.init_db()
    handlers = {
        "list-users": cmd_list_users,
        "reset-password": cmd_reset_password,
        "close-sessions": cmd_close_sessions,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
