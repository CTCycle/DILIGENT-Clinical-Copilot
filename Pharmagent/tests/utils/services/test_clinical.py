import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Pharmagent.app import constants as pharm_constants

fext_module = types.ModuleType("FEXT")
fext_app_module = types.ModuleType("FEXT.app")
fext_constants_module = types.ModuleType("FEXT.app.constants")

for attribute in dir(pharm_constants):
    if attribute.startswith("__"):
        continue
    setattr(fext_constants_module, attribute, getattr(pharm_constants, attribute))

sys.modules.setdefault("FEXT", fext_module)
sys.modules.setdefault("FEXT.app", fext_app_module)
sys.modules.setdefault("FEXT.app.constants", fext_constants_module)

from Pharmagent.app.api.schemas.clinical import PatientDrugs
from Pharmagent.app.utils.services.clinical import DrugToxicityEssay


class FakeResponse:
    def __init__(self, *, json_data=None, text="", status_code=200):
        self._json_data = json_data
        self.text = text
        self.status_code = status_code

    def json(self):
        if self._json_data is None:
            raise ValueError("No JSON payload available")
        return self._json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeAsyncClient:
    def __init__(self, routes):
        self.routes = routes
        self.calls = []

    async def get(self, url, params=None, **_):
        key = make_key(url, **(params or {}))
        self.calls.append(key)
        if key not in self.routes:
            raise AssertionError(f"Unexpected request: {key}")
        result = self.routes[key]
        if isinstance(result, Exception):
            raise result
        return result


def make_key(url: str, **params):
    normalized = tuple(sorted((key, str(value)) for key, value in params.items()))
    return url, normalized


@pytest.fixture
def essay():
    DrugToxicityEssay._match_cache.clear()
    return DrugToxicityEssay(PatientDrugs(entries=[]))


pytestmark = pytest.mark.asyncio


async def test_search_direct_match(essay, caplog):
    caplog.set_level("INFO")
    search_key = make_key(
        essay._search_url,
        db="books",
        term='LiverTox[book] AND "Acetaminophen"[title]',
        retmode="json",
        retmax="1",
    )
    summary_key = make_key(
        essay._summary_url,
        db="books",
        id="NBK1",
        retmode="json",
    )
    client = FakeAsyncClient(
        {
            search_key: FakeResponse(
                json_data={"esearchresult": {"idlist": ["NBK1"]}}
            ),
            summary_key: FakeResponse(
                json_data={
                    "result": {
                        "uids": ["NBK1"],
                        "NBK1": {"uid": "NBK1", "title": "Acetaminophen", "other": ["Tylenol"]},
                    }
                }
            ),
        }
    )

    match = await essay._search_livertox_id(client, "Acetaminophen")
    assert match
    assert match.nbk_id == "NBK1"
    assert match.matched_name == "Acetaminophen"
    assert match.reason == "direct_match"
    assert match.confidence == pytest.approx(1.0)
    assert "direct_match" in caplog.text


async def test_search_case_insensitive(essay):
    search_key = make_key(
        essay._search_url,
        db="books",
        term='LiverTox[book] AND "Acetaminophen"[title]',
        retmode="json",
        retmax="1",
    )
    summary_key = make_key(
        essay._summary_url,
        db="books",
        id="NBK2",
        retmode="json",
    )
    client = FakeAsyncClient(
        {
            search_key: FakeResponse(
                json_data={"esearchresult": {"idlist": ["NBK2"]}}
            ),
            summary_key: FakeResponse(
                json_data={
                    "result": {
                        "uids": ["NBK2"],
                        "NBK2": {"uid": "NBK2", "title": "Acetaminophen"},
                    }
                }
            ),
        }
    )

    match = await essay._search_livertox_id(client, "acetaminophen")
    assert match
    assert match.reason == "direct_match"
    assert match.nbk_id == "NBK2"


async def test_search_brand_name_mapping(essay):
    direct_key = make_key(
        essay._search_url,
        db="books",
        term='LiverTox[book] AND "Tylenol"[title]',
        retmode="json",
        retmax="1",
    )
    search_key = make_key(
        essay._search_url,
        db="books",
        term='LiverTox[book] AND "Acetaminophen"[title]',
        retmode="json",
        retmax="1",
    )
    summary_key = make_key(
        essay._summary_url,
        db="books",
        id="NBK3",
        retmode="json",
    )
    approx_key = make_key(
        essay._rxnorm_approx_url,
        term="Tylenol",
        maxEntries="5",
    )
    tty_key = make_key(
        f"{essay._rxnorm_property_url}/123/property.json",
        propName="TTY",
    )
    preferred_key = make_key(
        f"{essay._rxnorm_property_url}/123/property.json",
        propName="RxNorm Name",
    )
    synonyms_key = make_key(f"{essay._rxnorm_property_url}/123/synonyms.json")
    related_key = make_key(
        f"{essay._rxnorm_property_url}/123/related.json",
        tty="IN+PIN",
    )
    client = FakeAsyncClient(
        {
            direct_key: FakeResponse(
                json_data={"esearchresult": {"idlist": []}}
            ),
            search_key: FakeResponse(
                json_data={"esearchresult": {"idlist": ["NBK3"]}}
            ),
            summary_key: FakeResponse(
                json_data={
                    "result": {
                        "uids": ["NBK3"],
                        "NBK3": {"uid": "NBK3", "title": "Acetaminophen", "other": ["Tylenol"]},
                    }
                }
            ),
            approx_key: FakeResponse(
                json_data={
                    "approximateGroup": {
                        "candidate": [
                            {
                                "rxcui": "123",
                                "rxstring": "TYLENOL",
                            }
                        ]
                    }
                }
            ),
            tty_key: FakeResponse(
                json_data={
                    "propConceptGroup": {
                        "propConcept": [
                            {"propName": "TTY", "propValue": "SBD"}
                        ]
                    }
                }
            ),
            preferred_key: FakeResponse(
                json_data={
                    "propConceptGroup": {
                        "propConcept": [
                            {"propName": "RxNorm Name", "propValue": "TYLENOL"}
                        ]
                    }
                }
            ),
            synonyms_key: FakeResponse(
                json_data={"synonymGroup": {"synonym": ["Tylenol"]}}
            ),
            related_key: FakeResponse(
                json_data={
                    "relatedGroup": {
                        "conceptGroup": [
                            {
                                "conceptProperties": [
                                    {"rxcui": "456", "name": "Acetaminophen"}
                                ]
                            }
                        ]
                    }
                }
            ),
        }
    )

    match = await essay._search_livertox_id(client, "Tylenol")
    assert match
    assert match.reason == "brand_resolved"
    assert match.matched_name == "Acetaminophen"
    assert match.confidence == pytest.approx(0.92)


async def test_search_fuzzy_match(essay, caplog):
    caplog.set_level("INFO")
    direct_key = make_key(
        essay._search_url,
        db="books",
        term='LiverTox[book] AND "Acetaminophenn"[title]',
        retmode="json",
        retmax="1",
    )
    fallback_key = make_key(
        essay._search_url,
        db="books",
        term="LiverTox[book] AND Acetaminophenn",
        retmode="json",
        retmax="10",
    )
    approx_key = make_key(
        essay._rxnorm_approx_url,
        term="Acetaminophenn",
        maxEntries="5",
    )
    summary_key = make_key(
        essay._summary_url,
        db="books",
        id="NBK4",
        retmode="json",
    )
    client = FakeAsyncClient(
        {
            direct_key: FakeResponse(
                json_data={"esearchresult": {"idlist": []}}
            ),
            fallback_key: FakeResponse(
                json_data={"esearchresult": {"idlist": ["NBK4"]}}
            ),
            approx_key: FakeResponse(
                json_data={"approximateGroup": {"candidate": []}}
            ),
            summary_key: FakeResponse(
                json_data={
                    "result": {
                        "uids": ["NBK4"],
                        "NBK4": {
                            "uid": "NBK4",
                            "title": "Acetaminophen",
                            "other": ["Tylenol"],
                        },
                    }
                }
            ),
        }
    )

    match = await essay._search_livertox_id(client, "Acetaminophenn")
    assert match
    assert match.reason == "fuzzy_match"
    assert match.nbk_id == "NBK4"
    assert match.confidence == pytest.approx(0.89)
    assert "fuzzy_match" in caplog.text


async def test_search_first_result_fallback(essay):
    direct_key = make_key(
        essay._search_url,
        db="books",
        term='LiverTox[book] AND "Unmatched"[title]',
        retmode="json",
        retmax="1",
    )
    fallback_key = make_key(
        essay._search_url,
        db="books",
        term="LiverTox[book] AND Unmatched",
        retmode="json",
        retmax="10",
    )
    approx_key = make_key(
        essay._rxnorm_approx_url,
        term="Unmatched",
        maxEntries="5",
    )
    summary_key = make_key(
        essay._summary_url,
        db="books",
        id="NBK5",
        retmode="json",
    )
    client = FakeAsyncClient(
        {
            direct_key: FakeResponse(
                json_data={"esearchresult": {"idlist": []}}
            ),
            fallback_key: FakeResponse(
                json_data={"esearchresult": {"idlist": ["NBK5"]}}
            ),
            approx_key: FakeResponse(
                json_data={"approximateGroup": {"candidate": []}}
            ),
            summary_key: FakeResponse(
                json_data={
                    "result": {
                        "uids": ["NBK5"],
                        "NBK5": {"uid": "NBK5", "title": "Another Entry"},
                    }
                }
            ),
        }
    )

    match = await essay._search_livertox_id(client, "Unmatched")
    assert match
    assert match.reason == "list_first"
    assert match.nbk_id == "NBK5"
    assert match.confidence == pytest.approx(0.40)
