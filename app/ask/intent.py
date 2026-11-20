"""Decide whether a question is factual or must be refused before retrieval."""

from __future__ import annotations

import re

ADVICE = (
    "should i buy",
    "should i sell",
    "recommend",
    "which is better",
    "best fund",
    "portfolio",
    "allocate",
    "worth buying",
)
PERFORMANCE = (
    "compare returns",
    "best return",
    "higher return",
    "calculate return",
    "cagr",
    "rank",
    "outperform",
)
OTHER_AMCS = (
    "sbi",
    "icici",
    "axis",
    "nippon",
    "kotak",
    "mirae",
    "parag parikh",
    "uti",
    "dsp",
    "franklin",
    "aditya birla",
    "motilal",
    "edelweiss",
    "hsbc",
    "canara robeco",
    "invesco",
    "whiteoak",
    "quant ",
)


def classify_intent(question: str) -> str:
    text = question.lower()
    if any(phrase in text for phrase in ADVICE):
        return "advice"
    if any(phrase in text for phrase in PERFORMANCE):
        return "performance"
    if any(re.search(rf"\b{re.escape(name.strip())}\b", text) for name in OTHER_AMCS):
        return "out_of_scope"
    return "factual"
