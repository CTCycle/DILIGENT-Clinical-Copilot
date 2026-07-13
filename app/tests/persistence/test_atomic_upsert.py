from __future__ import annotations

from repositories.serialization.application_configuration import (
    ApplicationConfigurationSerializer,
)


def test_application_configuration_upsert_is_atomic(persistence_engine) -> None:  # type: ignore[no-untyped-def]
    serializer = ApplicationConfigurationSerializer(engine=persistence_engine)
    assert serializer.load() is None
    assert serializer.save({"revision": "first"}) == {"revision": "first"}
    assert serializer.save({"revision": "second"}) == {"revision": "second"}
    assert serializer.load() == {"revision": "second"}
