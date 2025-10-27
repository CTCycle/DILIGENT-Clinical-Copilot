from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import patch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from DILIGENT.app.utils.updater.rxnav import RxNavClient, RxNavDrugCatalogBuilder


###############################################################################
class RxNavClientFetchTermsTests(unittest.TestCase):
    ###########################################################################
    def test_fetch_drug_terms_extracts_unique_names(self) -> None:
        payload = {
            "drugGroup": {
                "conceptGroup": [
                    {
                        "conceptProperties": [
                            {
                                "name": (
                                    "0.4 ML abatacept 125 MG/ML Prefilled Syringe, "
                                    "0.7 ML abatacept 125 MG/ML Prefilled Syringe [Orencia]"
                                ),
                                "synonym": (
                                    "1 ML abatacept 125 MG/ML Auto-Injector, "
                                    "1 ML abatacept 125 MG/ML Auto-Injector [Orencia]"
                                ),
                            }
                        ]
                    }
                ]
            }
        }

        client = RxNavClient(enabled=False)
        with patch.object(RxNavClient, "request", return_value=payload):
            terms = client.fetch_drug_terms("Abatacept")

        self.assertSetEqual({"Abatacept", "Orencia"}, set(terms))

    ###########################################################################
    def test_fetch_drug_terms_falls_back_to_raw_name(self) -> None:
        client = RxNavClient(enabled=False)
        with patch.object(RxNavClient, "request", return_value=None):
            terms = client.fetch_drug_terms("abatacept")

        self.assertListEqual(["Abatacept"], terms)


###############################################################################
class RxNavDrugCatalogBuilderTests(unittest.TestCase):
    ###########################################################################
    def test_sanitize_concept_collects_synonyms(self) -> None:
        mapping = {
            "primary drug": ["Primary Drug", "Alternate Alias", "BrandTerm"],
        }
        by_rxcui = {"123": ["Synonym From Id", "BrandTerm"]}

        class StubClient(RxNavClient):
            def __init__(self) -> None:
                super().__init__(enabled=False)

            def fetch_drug_terms(self, raw_name: str) -> list[str]:
                return mapping.get(raw_name.casefold(), [])

            def fetch_rxcui_synonyms(self, rxcui: str) -> list[str]:
                return by_rxcui.get(rxcui, [])

        builder = RxNavDrugCatalogBuilder(rx_client=StubClient())
        concept = {"fullName": "Primary Drug [BrandTerm]", "rxcui": "123", "termType": "IN"}

        payload = builder.sanitize_concept(concept)

        self.assertIsNotNone(payload)
        assert payload is not None
        self.assertEqual(payload["brand_names"], ["BrandTerm"])
        self.assertEqual(payload["synonyms"], ["Alternate Alias", "Synonym From Id"])


if __name__ == "__main__":
    unittest.main()
