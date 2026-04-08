from __future__ import annotations

from pathlib import Path

from DILIGENT.server.configurations import env_loader


def test_runtime_env_values_are_loaded_from_dotenv(tmp_path, monkeypatch) -> None:
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text(
        "\n".join(
            [
                "FASTAPI_HOST=0.0.0.0",
                "FASTAPI_PORT=9000",
                "UI_HOST=127.0.0.1",
                "UI_PORT=7999",
                "KERAS_BACKEND=tensorflow",
                "MPLBACKEND=Agg",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(env_loader, "ENV_FILE_PATH", str(Path(dotenv_path)))
    monkeypatch.delenv("FASTAPI_HOST", raising=False)
    monkeypatch.delenv("FASTAPI_PORT", raising=False)
    monkeypatch.delenv("UI_HOST", raising=False)
    monkeypatch.delenv("UI_PORT", raising=False)
    monkeypatch.delenv("KERAS_BACKEND", raising=False)
    monkeypatch.delenv("MPLBACKEND", raising=False)

    settings = env_loader.load_environment()

    assert settings.fastapi_host == "0.0.0.0"
    assert settings.fastapi_port == 9000
    assert settings.ui_host == "127.0.0.1"
    assert settings.ui_port == 7999
    assert settings.keras_backend == "tensorflow"
    assert settings.mpl_backend == "Agg"
