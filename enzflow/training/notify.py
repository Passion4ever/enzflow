"""Training notification via Telegram or Email.

Sensitive credentials are read from environment variables, not config files.

    TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
    EMAIL_SMTP_HOST, EMAIL_SMTP_PORT, EMAIL_SENDER, EMAIL_PASSWORD, EMAIL_RECEIVER
"""

from __future__ import annotations

import os
import smtplib
import urllib.parse
import urllib.request
from email.mime.text import MIMEText


def send_telegram(bot_token: str, chat_id: str, message: str) -> bool:
    """Send a Telegram message. Returns True on success."""
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        data = urllib.parse.urlencode({
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "Markdown",
        }).encode()
        req = urllib.request.Request(url, data=data)
        urllib.request.urlopen(req, timeout=10)
        return True
    except Exception as e:
        print(f"Telegram notification failed: {e}")
        return False


def send_email(
    smtp_host: str,
    smtp_port: int,
    sender: str,
    password: str,
    receiver: str,
    subject: str,
    body: str,
) -> bool:
    """Send an email via SMTP. Returns True on success."""
    try:
        msg = MIMEText(body, "plain", "utf-8")
        msg["Subject"] = subject
        msg["From"] = sender
        msg["To"] = receiver
        with smtplib.SMTP_SSL(smtp_host, smtp_port, timeout=10) as s:
            s.login(sender, password)
            s.send_message(msg)
        return True
    except Exception as e:
        print(f"Email notification failed: {e}")
        return False


def notify(cfg: dict, message: str, subject: str = "enzflow") -> None:
    """Send notification based on environment variables.

    Telegram: set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID.
    Email: set EMAIL_SMTP_HOST, EMAIL_SENDER, EMAIL_PASSWORD, EMAIL_RECEIVER.
    """
    # Telegram
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if bot_token and chat_id:
        send_telegram(bot_token, chat_id, message)

    # Email
    smtp_host = os.environ.get("EMAIL_SMTP_HOST")
    sender = os.environ.get("EMAIL_SENDER")
    password = os.environ.get("EMAIL_PASSWORD")
    receiver = os.environ.get("EMAIL_RECEIVER")
    if smtp_host and sender and password and receiver:
        smtp_port = int(os.environ.get("EMAIL_SMTP_PORT", "465"))
        send_email(smtp_host, smtp_port, sender, password, receiver, subject, message)
