from __future__ import annotations

from pathlib import Path

from common.runtime_layout import RuntimeLayout
from configurations import runtime_bootstrap


def _packaged_layout(runtime_root: Path, data_root: Path) -> RuntimeLayout:
    return RuntimeLayout(
        packaged=True,
        runtime_root=runtime_root,
        application_root=runtime_root / "app",
        settings_template_root=runtime_root / "settings",
        settings_root=data_root / "settings",
        immutable_resources_root=runtime_root / "app" / "resources",
        mutable_resources_root=data_root / "resources",
        client_dist_root=runtime_root / "app" / "client" / "dist" / "browser",
    )


def test_first_launch_seeds_only_settings_and_mutable_directories(
    monkeypatch,
    tmp_path: Path,
) -> None:
    runtime_root = tmp_path / "runtime"
    data_root = tmp_path / "data"
    layout = _packaged_layout(runtime_root, data_root)
    template_root = layout.settings_template_root
    template_root.mkdir(parents=True)
    (template_root / ".env.example").write_text("EXAMPLE=true\n", encoding="utf-8")
    (template_root / "configurations.json").write_text("{}", encoding="utf-8")
    (layout.immutable_resources_root / "database.db").parent.mkdir(parents=True)
    (layout.immutable_resources_root / "database.db").write_text("seed", encoding="utf-8")
    (layout.immutable_resources_root / "access-key-material.json").write_text(
        "secret", encoding="utf-8"
    )
    monkeypatch.setattr(runtime_bootstrap, "resolve_runtime_layout", lambda: layout)

    runtime_bootstrap.ensure_runtime_data_layout()

    assert (data_root / "settings/.env").read_text(encoding="utf-8") == "EXAMPLE=true\n"
    assert (data_root / "settings/configurations.json").read_text(encoding="utf-8") == "{}"
    assert not (data_root / "resources/database.db").exists()
    assert not (data_root / "resources/access-key-material.json").exists()
    for relative_path in (
        "logs",
        "models/embeddings",
        "sources/archives",
        "sources/documents",
        "sources/vectors",
        "exports",
        "state",
    ):
        assert (data_root / "resources" / relative_path).is_dir()


def test_existing_settings_are_preserved(monkeypatch, tmp_path: Path) -> None:
    runtime_root = tmp_path / "runtime"
    data_root = tmp_path / "data"
    layout = _packaged_layout(runtime_root, data_root)
    layout.settings_template_root.mkdir(parents=True)
    (layout.settings_template_root / ".env.example").write_text("new", encoding="utf-8")
    (layout.settings_template_root / "configurations.json").write_text("new", encoding="utf-8")
    layout.settings_root.mkdir(parents=True)
    (layout.settings_root / ".env").write_text("user", encoding="utf-8")
    (layout.settings_root / "configurations.json").write_text("user", encoding="utf-8")
    monkeypatch.setattr(runtime_bootstrap, "resolve_runtime_layout", lambda: layout)

    runtime_bootstrap.ensure_runtime_data_layout()

    assert (layout.settings_root / ".env").read_text(encoding="utf-8") == "user"
    assert (layout.settings_root / "configurations.json").read_text(encoding="utf-8") == "user"


def test_source_mode_is_a_no_op(monkeypatch, tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    layout = RuntimeLayout(
        packaged=False,
        runtime_root=source_root,
        application_root=source_root / "app",
        settings_template_root=source_root / "settings",
        settings_root=source_root / "settings",
        immutable_resources_root=source_root / "app/resources",
        mutable_resources_root=source_root / "app/resources",
        client_dist_root=source_root / "app/client/dist/browser",
    )
    monkeypatch.setattr(runtime_bootstrap, "resolve_runtime_layout", lambda: layout)

    runtime_bootstrap.ensure_runtime_data_layout()

    assert not source_root.exists()
