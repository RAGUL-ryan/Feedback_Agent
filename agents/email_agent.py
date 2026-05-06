"""
agents/email_agent.py  (Gmail SMTP version — no SendGrid needed)
─────────────────────────────────────────────────────────────────────────────
Sends emails using Gmail free SMTP. No sign up, no admin, no credit card.

Setup (5 minutes):
  1. myaccount.google.com → Security → enable 2-Step Verification
  2. Security → App passwords → Create → name: "Feedback Agent"
  3. Copy the 16-char password Google gives you
  4. Add to .env:
       GMAIL_ADDRESS=your.gmail@gmail.com
       GMAIL_APP_PASSWORD=abcdefghijklmnop
       EMAIL_FROM_NAME=Feedback Agent

If not configured → emails skip silently. Pipeline never crashes.
"""

from __future__ import annotations
import os, smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from dotenv import load_dotenv

load_dotenv()

GMAIL_ADDRESS      = os.getenv("GMAIL_ADDRESS", "")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD", "").replace(" ", "")
EMAIL_FROM_NAME    = os.getenv("EMAIL_FROM_NAME", "Feedback Agent")


def _is_configured() -> bool:
    return bool(GMAIL_ADDRESS and GMAIL_APP_PASSWORD)


def _send(to_email: str, to_name: str, subject: str, body_html: str) -> bool:
    if not _is_configured():
        print("[EMAIL] Gmail not configured in .env — skipping.")
        return False
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = f"{EMAIL_FROM_NAME} <{GMAIL_ADDRESS}>"
        msg["To"]      = f"{to_name} <{to_email}>"
        msg.attach(MIMEText(body_html, "html"))

        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.ehlo()
            server.starttls()
            server.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
            server.sendmail(GMAIL_ADDRESS, to_email, msg.as_string())

        print(f"[EMAIL] ✓ Sent to {to_email}")
        return True

    except smtplib.SMTPAuthenticationError:
        print("[EMAIL] Gmail auth failed — check GMAIL_APP_PASSWORD in .env")
        print("[EMAIL] Use the App Password (16 chars), NOT your Gmail login password")
        return False
    except Exception as exc:
        print(f"[EMAIL] Send failed (non-fatal): {exc}")
        return False


def send_escalation_email(
    customer_email: str,
    customer_name: str,
    ticket_id: str,
    feedback: str,
    sector: str = "support",
) -> bool:
    subject = f"We've received your case — Ticket #{ticket_id}"
    preview = feedback[:120] + ("..." if len(feedback) > 120 else "")
    body = f"""
    <div style="font-family:Segoe UI,sans-serif;max-width:560px;margin:auto;padding:32px;background:#f9f9f9;border-radius:12px;">
      <h2 style="color:#1a1a1a;margin-bottom:8px;">We've received your feedback</h2>
      <p style="color:#555;font-size:15px;line-height:1.7;">Hi {customer_name},</p>
      <p style="color:#555;font-size:15px;line-height:1.7;">Your case has been escalated to our human support team. A team member will contact you shortly.</p>
      <div style="background:#fff;border:1px solid #e0e0e0;border-radius:8px;padding:16px;margin:20px 0;">
        <p style="font-size:13px;color:#999;margin:0 0 4px;">Your ticket reference</p>
        <p style="font-size:22px;font-weight:700;color:#6c63ff;margin:0;">#{ticket_id}</p>
      </div>
      <p style="color:#888;font-size:13px;">Your message: <em>"{preview}"</em></p>
      <hr style="border:none;border-top:1px solid #eee;margin:24px 0;">
      <p style="color:#bbb;font-size:12px;text-align:center;">Automated notification from {EMAIL_FROM_NAME}. Do not reply.</p>
    </div>"""
    return _send(customer_email, customer_name, subject, body)


def send_reply_email(
    customer_email: str,
    customer_name: str,
    ai_reply: str,
    feedback: str,
) -> bool:
    subject = "Your feedback — here's our response"
    preview = feedback[:120] + ("..." if len(feedback) > 120 else "")
    body = f"""
    <div style="font-family:Segoe UI,sans-serif;max-width:560px;margin:auto;padding:32px;background:#f9f9f9;border-radius:12px;">
      <h2 style="color:#1a1a1a;margin-bottom:8px;">Here's our response</h2>
      <p style="color:#555;font-size:15px;line-height:1.7;">Hi {customer_name},</p>
      <p style="color:#888;font-size:13px;margin-bottom:16px;">Regarding: <em>"{preview}"</em></p>
      <div style="background:#f0fff6;border:1px solid #b2f0d0;border-radius:8px;padding:18px;font-size:15px;color:#1a1a1a;line-height:1.7;">{ai_reply}</div>
      <hr style="border:none;border-top:1px solid #eee;margin:24px 0;">
      <p style="color:#bbb;font-size:12px;text-align:center;">Automated response from {EMAIL_FROM_NAME}.</p>
    </div>"""
    return _send(customer_email, customer_name, subject, body)