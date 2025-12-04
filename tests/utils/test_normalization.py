from __future__ import annotations

import unittest

import pandas as pd

from DILIGENT.server.utils.services.text.normalization import (
    coerce_text,
    normalize_drug_name,
    normalize_token,
    normalize_whitespace,
)


class NormalizationTests(unittest.TestCase):
    # ------------------------------------------------------------------
    def test_coerce_text_strips_and_handles_empty_values(self) -> None:
        self.assertEqual(coerce_text("  Hepa  rin  "), "Hepa  rin")
        self.assertIsNone(coerce_text("   "))

    # ------------------------------------------------------------------
    def test_coerce_text_handles_non_string_values(self) -> None:
        self.assertEqual(coerce_text(123), "123")
        self.assertEqual(coerce_text(4.5), "4.5")

    # ------------------------------------------------------------------
    def test_coerce_text_handles_pandas_missing_markers(self) -> None:
        self.assertIsNone(coerce_text(pd.NA))
        self.assertIsNone(coerce_text(None))

    # ------------------------------------------------------------------
    def test_normalize_whitespace(self) -> None:
        self.assertEqual(
            normalize_whitespace("  Foo \n Bar \t Baz  "),
            "Foo Bar Baz",
        )
        self.assertEqual(normalize_whitespace(""), "")

    # ------------------------------------------------------------------
    def test_normalize_drug_name(self) -> None:
        self.assertEqual(
            normalize_drug_name("Ácétyl-CoA 500mg! (test)"),
            "acetyl coa 500mg test",
        )

    # ------------------------------------------------------------------
    def test_normalize_token(self) -> None:
        self.assertEqual(normalize_token("Dose,"), "dose")
        self.assertEqual(normalize_token("MG..;;"), "mg")


if __name__ == "__main__":
    unittest.main()
