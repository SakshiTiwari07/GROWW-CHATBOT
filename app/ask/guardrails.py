"""Refuse questions that contain personal identifiers before retrieval or the LLM."""

from __future__ import annotations

import re
from dataclasses import dataclass

PAN = re.compile(r"\b[A-Za-z]{5}[0-9]{4}[A-Za-z]\b")
AADHAAR = re.compile(r"\b\d{4}\s\d{4}\s\d{4}\b|\b\d{12}\b")
EMAIL = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
PHONE = re.compile(r"(?:\+91[\s-]?)?[6-9]\d{9}\b")
ACCOUNT = re.compile(r"\b(?:account|a/c|acct)\b.{0,20}\b\d{9,18}\b", re.IGNORECASE)
OTP = re.compile(r"\b(?:otp|pin)\b.{0,20}\b\d{4,8}\b", re.IGNORECASE)


@dataclass(frozen=True)
class PiiResult:
    blocked: bool


def screen_pii(question: str) -> PiiResult:
    blocked = any(
        pattern.search(question)
        for pattern in (PAN, AADHAAR, EMAIL, PHONE, ACCOUNT, OTP)
    )
    return PiiResult(blocked=blocked)
