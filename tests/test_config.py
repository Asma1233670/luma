"""Timezone detection: the agent must speak the user's local time, not UTC."""
from datetime import datetime
from zoneinfo import ZoneInfo

from backend.app import config as config_module


def clear_env(monkeypatch):
    monkeypatch.delenv("LUMA_TZ", raising=False)
    monkeypatch.delenv("TZ", raising=False)


def test_explicit_override_wins(monkeypatch):
    monkeypatch.setenv("LUMA_TZ", "Europe/Paris")
    monkeypatch.setenv("TZ", "UTC")
    assert config_module.detect_timezone() == "Europe/Paris"


def test_tz_environment_variable(monkeypatch):
    clear_env(monkeypatch)
    monkeypatch.setenv("TZ", "Africa/Tunis")
    assert config_module.detect_timezone() == "Africa/Tunis"


def test_machine_zone_matches_the_local_clock(monkeypatch):
    clear_env(monkeypatch)
    zone = config_module.detect_timezone()
    now = datetime.now()
    assert now.astimezone(ZoneInfo(zone)).utcoffset() == now.astimezone().utcoffset()


def test_detects_zone_from_localtime_symlink(monkeypatch, tmp_path):
    clear_env(monkeypatch)
    zoneinfo = tmp_path / "zoneinfo" / "Europe"
    zoneinfo.mkdir(parents=True)
    (zoneinfo / "Paris").touch()
    localtime = tmp_path / "localtime"
    localtime.symlink_to(zoneinfo / "Paris")
    monkeypatch.setattr(config_module, "TZ_FILE", tmp_path / "timezone")
    monkeypatch.setattr(config_module, "ZONEINFO_DIR", tmp_path / "zoneinfo")
    monkeypatch.setattr(config_module, "LOCALTIME_FILE", localtime)
    assert config_module.detect_timezone() == "Europe/Paris"


def test_falls_back_to_utc_without_zone_information(monkeypatch, tmp_path):
    clear_env(monkeypatch)
    monkeypatch.setattr(config_module, "TZ_FILE", tmp_path / "timezone")
    monkeypatch.setattr(config_module, "LOCALTIME_FILE", tmp_path / "localtime")
    assert config_module.detect_timezone() == "UTC"
