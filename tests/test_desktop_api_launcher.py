import json

import desktop_api_launcher as launcher


def test_launcher_creates_persistent_key_and_defaults(tmp_path, monkeypatch):
    settings_file = tmp_path / "settings.json"
    monkeypatch.setenv("T8STAR_INDEXTTS_SETTINGS", str(settings_file))
    settings = launcher.ensure_api_settings({"modelDir": "X"})
    assert len(settings["apiKey"]) >= 16
    assert settings["apiPort"] == 7861
    assert settings["apiHost"] == "127.0.0.1"
    assert json.loads(settings_file.read_text(encoding="utf-8"))["apiKey"] == settings["apiKey"]


def test_launcher_status_only_accepts_matching_service_identity(tmp_path, monkeypatch):
    monkeypatch.setenv("T8STAR_INDEXTTS_SETTINGS", str(tmp_path / "settings.json"))
    settings = launcher.ensure_api_settings({"apiKey": "0123456789abcdef", "apiPort": 7861})
    fingerprint = __import__("hashlib").sha256(settings["apiKey"].encode()).hexdigest()[:16]
    monkeypatch.setattr(
        launcher,
        "request_json",
        lambda *args, **kwargs: {
            "service": "t8star-indextts-2.5",
            "api_key_fingerprint": fingerprint,
        },
    )
    assert launcher.running_health(settings)
    monkeypatch.setattr(
        launcher,
        "request_json",
        lambda *args, **kwargs: {
            "service": "t8star-indextts-2.5",
            "api_key_fingerprint": "different",
        },
    )
    assert launcher.running_health(settings) is None
