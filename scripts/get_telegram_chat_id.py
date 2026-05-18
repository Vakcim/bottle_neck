from __future__ import annotations

import os
import requests
from dotenv import load_dotenv


def main():
    load_dotenv()

    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set in .env")

    url = f"https://api.telegram.org/bot{token}/getUpdates"
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    data = response.json()

    if not data.get("result"):
        print("Нет сообщений. Напиши боту /start в Telegram и запусти скрипт ещё раз.")
        return

    for update in data["result"]:
        message = update.get("message") or update.get("edited_message")
        if not message:
            continue

        chat = message.get("chat", {})
        print("chat_id:", chat.get("id"))
        print("type:", chat.get("type"))
        print("username:", chat.get("username"))
        print("first_name:", chat.get("first_name"))
        print("---")


if __name__ == "__main__":
    main()
