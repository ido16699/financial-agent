"""Notifier protocol and implementations.

Notifier is a simple protocol with a single method:
    send(subject: str, html: str, text: str) -> None

Implementations:
  ConsoleNotifier  — prints to stdout (default / --dry-run)
  SMTPNotifier     — sends email via SMTP (production)
"""

from __future__ import annotations

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Protocol

from stochsignal.config import settings
from stochsignal.logging_utils import get_logger

log = get_logger(__name__)


class Notifier(Protocol):
    def send(self, subject: str, html: str, text: str) -> None: ...


class ConsoleNotifier:
    """Prints the digest to stdout — used for --dry-run."""

    def send(self, subject: str, html: str, text: str) -> None:
        print("=" * 70)
        print(f"SUBJECT: {subject}")
        print("=" * 70)
        print(text)
        print("=" * 70)


class SMTPNotifier:
    """Sends the digest via SMTP (TLS).

    Reads credentials from settings (populated via .env):
      SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, DIGEST_RECIPIENTS
    """

    def send(self, subject: str, html: str, text: str) -> None:
        recipients = settings.digest_recipients
        if not recipients:
            log.error("No DIGEST_RECIPIENTS configured — cannot send email.")
            return

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = settings.smtp_user
        msg["To"] = ", ".join(recipients)

        msg.attach(MIMEText(text, "plain"))
        msg.attach(MIMEText(html, "html"))

        log.info("Sending digest to: %s", recipients)
        try:
            with smtplib.SMTP(settings.smtp_host, settings.smtp_port) as server:
                server.ehlo()
                server.starttls()
                server.login(settings.smtp_user, settings.smtp_password)
                server.sendmail(settings.smtp_user, recipients, msg.as_string())
            log.info("Digest sent successfully.")
        except Exception as exc:
            log.error("SMTP send failed: %s", exc)
            raise
