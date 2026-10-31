import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "libresidfp_release", ROOT / "scripts" / "libresidfp_release.py"
)
tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tool)

LS_REMOTE = (
    "aaa\trefs/tags/v1.2.1\n"
    "bbb\trefs/tags/v1.10.0\n"
    "ccc\trefs/tags/v1.10.0^{}\n"
    "ddd\trefs/tags/v2.0.0-rc1\n"
)


def test_semver_orders_numerically():
    assert tool.semver("v1.10.0") > tool.semver("v1.9.9")
    assert tool.semver("v2.0.0-rc1") is None
    assert tool.semver("1.2.3") is None


def test_highest_ignores_non_releases():
    assert tool.highest(["v1.2.3", "v1.10.0", "v2.0.0-rc1", "nightly"]) == "v1.10.0"
    assert tool.highest([]) == ""


def test_ls_remote_peels_annotated_tags():
    commits = tool.parse_ls_remote(LS_REMOTE)
    assert commits["v1.10.0"] == "ccc"
    assert commits["v1.2.1"] == "aaa"


def test_bump_patch_raises_only_the_patch_component():
    assert tool.bump_patch('name = "x"\nversion = "0.2.9"\n') == (
        'name = "x"\nversion = "0.2.10"\n'
    )
    assert tool.bump_patch('__version__ = "1.0.0"\n') == '__version__ = "1.0.1"\n'


def test_bump_version_keeps_the_files_in_step(tmp_path, monkeypatch, capsys):
    for name, line in zip(
        tool.VERSION_FILES, ('version = "0.3.4"', '__version__ = "0.3.4"')
    ):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(line + "\n")
    monkeypatch.setattr(tool, "ROOT", tmp_path)
    assert tool.main(["bump-version"]) == 0
    assert "0.3.4 -> 0.3.5" in capsys.readouterr().out
    assert all('"0.3.5"' in (tmp_path / n).read_text() for n in tool.VERSION_FILES)
    (tmp_path / tool.VERSION_FILES[1]).write_text('__version__ = "9.9.9"\n')
    with pytest.raises(SystemExit):
        tool.main(["bump-version"])


@pytest.mark.parametrize(
    "pinned, newer", [("v1.2.2", "true"), ("v1.2.3", "false"), ("v1.2.4", "false")]
)
def test_check_compares_against_the_pin(pinned, newer, capsys):
    assert tool.main(["check", "--latest", "v1.2.3", "--pinned", pinned]) == 0
    assert f"newer={newer}" in capsys.readouterr().out


def test_check_refuses_unparseable_tags():
    with pytest.raises(SystemExit):
        tool.main(["check", "--latest", "nightly", "--pinned", "v1.2.3"])


def test_patches_apply_to_the_pinned_submodule():
    assert tool.main(["patches"]) == 0


def test_a_patch_that_does_not_apply_is_reported(tmp_path, monkeypatch, capsys):
    (tmp_path / "patches").mkdir()
    (tmp_path / "patches" / "0001-bad.patch").write_text(
        "--- a/src/none.h\n+++ b/src/none.h\n@@ -1 +1 @@\n-x\n+y\n"
    )
    monkeypatch.setattr(tool, "ROOT", tmp_path)
    monkeypatch.setattr(tool, "SUBMODULE", str(ROOT / tool.SUBMODULE))
    assert tool.main(["patches"]) == 1
    assert "0001-bad.patch does not apply" in capsys.readouterr().err


def test_library_version_reads_configure_ac():
    text = "".join(
        f"m4_define([lib_{k}], [{v}])\n"
        for k, v in (("major", 1), ("minor", 12), ("level", 3))
    )
    assert tool.library_version(text) == (1, 12, 3)


@pytest.mark.parametrize(
    "version, names",
    [((1, 2, 2), ["a", "old"]), ((1, 2, 3), ["a"]), ((1, 3, 0), ["a"])],
)
def test_patch_files_select_by_version(tmp_path, version, names):
    (tmp_path / "before-1.2.3").mkdir()
    (tmp_path / "a.patch").touch()
    (tmp_path / "before-1.2.3" / "old.patch").touch()
    assert [p.stem for p in tool.patch_files(version, tmp_path)] == names
