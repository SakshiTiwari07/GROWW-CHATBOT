"""POST /ask returns the retrieval contract."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.api import app
from app.ask.service import AskResponse


class AskApiTests(unittest.TestCase):
    def test_home_page_lists_the_scope_and_examples(self) -> None:
        response = TestClient(app).get("/")
        self.assertEqual(response.status_code, 200)
        page = response.text
        self.assertIn("HDFC Mutual Fund Facts", page)
        self.assertIn("AMC overview plus Large Cap, Flexi Cap, ELSS, and Mid Cap.", page)
        self.assertIn("Facts-only. No investment advice.", page)
        self.assertIn("What is the expense ratio of HDFC Large Cap Fund Direct Growth?", page)
        self.assertIn("What is the ELSS lock-in?", page)
        self.assertIn("What is the minimum SIP for HDFC Flexi Cap Direct Plan Growth?", page)
        self.assertNotIn("password", page.lower())
        self.assertNotIn("type=\"email\"", page)

    def test_empty_question_is_rejected(self) -> None:
        response = TestClient(app).post("/ask", json={"question": "   "})
        self.assertEqual(response.status_code, 400)

    def test_post_ask_returns_the_answer_contract(self) -> None:
        canned = AskResponse(
            answer="The expense ratio is 0.99%. Source: https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth\nLast updated from sources: 2026-11-20",
            citation_url="https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth",
            last_updated="2026-11-20",
            refused=False,
        )
        with patch("app.ask.service.ask", return_value=canned) as ask_mock:
            response = TestClient(app).post(
                "/ask",
                json={"question": " What is the expense ratio of HDFC Large Cap Fund Direct Growth? "},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            set(response.json()),
            {"answer", "citation_url", "last_updated", "refused"},
        )
        self.assertFalse(response.json()["refused"])
        ask_mock.assert_called_once()
        self.assertEqual(
            ask_mock.call_args.args[0],
            "What is the expense ratio of HDFC Large Cap Fund Direct Growth?",
        )


if __name__ == "__main__":
    unittest.main()
