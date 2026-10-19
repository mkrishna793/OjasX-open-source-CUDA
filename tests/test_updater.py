"""
tests/test_updater.py

Unit test suite for OjasX Self-Updating Engine & Live Release Synchronizer.
"""

import os
from unittest.mock import patch, MagicMock
import pytest

from ojasx2.updater import (
    _parse_version,
    get_current_version,
    get_repo_root,
    check_for_updates,
    format_update_banner,
    perform_update,
)


def test_parse_version():
    assert _parse_version("0.7.1") == (0, 7, 1)
    assert _parse_version("v0.7.1") == (0, 7, 1)
    assert _parse_version("v0.8.1-rc1") == (0, 8, 1)
    assert _parse_version("1.0.0") > _parse_version("0.8.1")
    assert _parse_version("0.8.2") > _parse_version("0.8.1")


def test_get_current_version():
    ver = get_current_version()
    assert isinstance(ver, str)
    assert ver == "0.8.1"


def test_get_repo_root():
    root = get_repo_root()
    assert root is not None
    assert os.path.exists(os.path.join(root, ".git"))


def test_format_update_banner():
    banner = format_update_banner("0.8.1", "0.8.2", is_git=True)
    assert "OjasX Update Available" in banner
    assert "v0.8.1 → v0.8.2" in banner
    assert "ojasx update" in banner


@patch("ojasx2.updater.fetch_latest_release_info")
def test_check_for_updates_available(mock_fetch):
    mock_fetch.return_value = {
        "tag_name": "0.8.2",
        "name": "OjasX v0.8.2",
        "published_at": "2026-10-19T18:00:00Z",
    }
    has_update, latest_ver, meta = check_for_updates(force=True)
    assert has_update is True
    assert latest_ver == "0.8.2"
    assert meta["name"] == "OjasX v0.8.2"


@patch("ojasx2.updater.fetch_latest_release_info")
def test_check_for_updates_already_current(mock_fetch):
    mock_fetch.return_value = {
        "tag_name": "0.8.1",
        "name": "OjasX v0.8.1",
        "published_at": "2026-10-19T06:00:00Z",
    }
    has_update, latest_ver, meta = check_for_updates(force=True)
    assert has_update is False
    assert latest_ver == "0.8.1"
