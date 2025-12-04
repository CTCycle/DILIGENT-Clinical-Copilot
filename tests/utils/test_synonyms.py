from __future__ import annotations

import unittest

from DILIGENT.server.utils.services.text.synonyms import (
    extract_synonym_strings,
    parse_synonym_list,
    split_synonym_variants,
    try_parse_json,
)


class SynonymHelpersTests(unittest.TestCase):
    # ------------------------------------------------------------------
    def test_extract_synonym_strings_handles_nested_structures(self) -> None:
        payload = {
            "primary": ["Alpha", {"child": "Beta"}],
            "secondary": ("Gamma", ["Delta", {"deep": "Epsilon"}]),
        }
        extracted = extract_synonym_strings(payload)
        self.assertCountEqual(
            extracted,
            ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"],
        )

    # ------------------------------------------------------------------
    def test_extract_synonym_strings_parses_json_text(self) -> None:
        serialized = '["One", {"nested": ["Two"]}]'
        extracted = extract_synonym_strings(serialized)
        self.assertEqual(extracted, ["One", "Two"])

    # ------------------------------------------------------------------
    def test_parse_synonym_list_filters_empty_entries(self) -> None:
        values = ["Alpha", "  ", None, "Beta"]
        parsed = parse_synonym_list(values)
        self.assertEqual(parsed, ["Alpha", "Beta"])

    # ------------------------------------------------------------------
    def test_split_synonym_variants_handles_delimiters(self) -> None:
        variants = split_synonym_variants("Drug A; Drug B/Drug C,Drug D")
        self.assertEqual(variants, ["Drug A", "Drug B", "Drug C", "Drug D"])

    # ------------------------------------------------------------------
    def test_try_parse_json_handles_invalid_payload(self) -> None:
        self.assertIsNone(try_parse_json("not json"))


if __name__ == "__main__":
    unittest.main()
