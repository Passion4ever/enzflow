"""Training notification via Telegram or Email."""

from __future__ import annotations

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
    """Send notification based on config.

    Checks cfg for 'telegram' and/or 'email' sections.
    """
    tg = cfg.get("telegram")
    if tg and tg.get("bot_token") and tg.get("chat_id"):
        send_telegram(tg["bot_token"], str(tg["chat_id"]), message)

    em = cfg.get("email")
    if em and em.get("smtp_host") and em.get("receiver"):
        send_email(
            smtp_host=em["smtp_host"],
            smtp_port=em.get("smtp_port", 465),
            sender=em["sender"],
            password=em["password"],
            receiver=em["receiver"],
            subject=subject,
            body=message,
        )
