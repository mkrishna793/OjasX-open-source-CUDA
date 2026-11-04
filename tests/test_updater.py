"""
tests/test_updater.py

Unit test suite for OjasX Self-Updating Engine & Live Release Synchronizer.
"""

import os
import sys
from unittest.mock import patch, MagicMock
import pytest

from ojasx2.updater import (
    _parse_version,
    get_current_version,
    get_repo_root,
    check_for_updates,
    format_update_banner,
    perform_update,
    main as updater_main,
)


def test_parse_version():
    assert _parse_version("0.7.1") == (0, 7, 1)
    assert _parse_version("v0.7.1") == (0, 7, 1)
    assert _parse_version("v0.8.1-rc1") == (0, 8, 1)
    assert _parse_version("1.0.0") > _parse_version("0.9.0")
    assert _parse_version("0.9.1") > _parse_version("0.9.0")


def test_get_current_version():
    ver = get_current_version()
    assert isinstance(ver, str)
    import ojasx2
    assert ver == ojasx2.__version__


def test_get_repo_root():
    root = get_repo_root()
    assert root is not None
    assert os.path.exists(os.path.join(root, ".git"))


def test_format_update_banner():
    banner = format_update_banner("0.11.0", "0.11.1", is_git=True)
    assert "OjasX Update Available" in banner
    assert "v0.11.0 → v0.11.1" in banner
    assert "ojasx update" in banner


@patch("ojasx2.updater.fetch_latest_release_info")
def test_check_for_updates_available(mock_fetch):
    mock_fetch.return_value = {
        "tag_name": "0.17.0",
        "name": "OjasX v0.17.0",
        "published_at": "2026-10-25T18:00:00Z",
    }
    has_update, latest_ver, meta = check_for_updates(force=True)
    assert has_update is True
    assert latest_ver == "0.17.0"
    assert meta["name"] == "OjasX v0.17.0"


@patch("ojasx2.updater.fetch_latest_release_info")
def test_check_for_updates_already_current(mock_fetch):
    mock_fetch.return_value = {
        "tag_name": "0.9.0",
        "name": "OjasX v0.9.0",
        "published_at": "2026-10-19T06:00:00Z",
    }
    has_update, latest_ver, meta = check_for_updates(force=True)
    assert has_update is False
    assert latest_ver == "0.9.0"


@patch("ojasx2.updater.subprocess.run")
@patch("ojasx2.updater.get_repo_root", return_value=None)
def test_pip_update_honors_target_version_in_quiet_mode(_repo, run_mock):
    assert perform_update(target_version="0.12.0", quiet=True) is True
    command = run_mock.call_args_list[0].args[0]
    assert command[-1].endswith(".git@v0.12.0")
    assert run_mock.call_args_list[0].kwargs == {"check": True}


@patch("ojasx2.updater.perform_update", return_value=True)
def test_auto_flag_calls_quiet_update(perform_mock):
    with patch.object(sys, "argv", ["ojasx-update", "--auto", "v0.14.0"]):
        updater_main()
    perform_mock.assert_called_once_with(target_version="v0.14.0", quiet=True)


def test_check_rejects_update_actions():
    with patch.object(sys, "argv", ["ojasx-update", "--check", "--auto"]):
        with pytest.raises(SystemExit) as caught:
            updater_main()
    assert caught.value.code == 2
