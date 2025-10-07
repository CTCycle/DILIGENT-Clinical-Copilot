from __future__ import annotations

import unittest
from unittest.mock import patch

from DILIGENT.app.utils.services.retrieval import RxNavClient


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
        with patch.object(RxNavClient, "_request", return_value=payload):
            terms = client.fetch_drug_terms("Abatacept")

        self.assertSetEqual({"Abatacept", "Orencia"}, set(terms))

    ###########################################################################
    def test_fetch_drug_terms_falls_back_to_raw_name(self) -> None:
        client = RxNavClient(enabled=False)
        with patch.object(RxNavClient, "_request", return_value=None):
            terms = client.fetch_drug_terms("abatacept")

        self.assertListEqual(["Abatacept"], terms)


if __name__ == "__main__":
    unittest.main()
