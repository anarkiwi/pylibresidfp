"""Track libresidfp release tags for the ``extern/libresidfp`` submodule.

``check`` prints ``latest=``, ``pinned=`` and ``newer=`` lines (appendable to
``$GITHUB_OUTPUT``), ``bump-version`` raises pylibresidfp's patch version in
``pyproject.toml`` and ``src/pylibresidfp/__init__.py``, and ``patches`` applies
``patches/*.patch`` in order to a scratch copy of the submodule's sources.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUBMODULE = "extern/libresidfp"
UPSTREAM = "https://github.com/libsidplayfp/libresidfp"
RELEASE_TAG = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")
VERSION_FILES = ("pyproject.toml", "src/pylibresidfp/__init__.py")
VERSION_LINE = re.compile(r'^(version|__version__) = "(\d+)\.(\d+)\.(\d+)"$', re.M)


def semver(tag: str) -> tuple[int, int, int] | None:
    """``(major, minor, patch)`` of a ``vX.Y.Z`` release tag, else ``None``."""
    match = RELEASE_TAG.match(tag)
    return tuple(map(int, match.groups())) if match else None


def highest(tags) -> str:
    """The highest release tag among ``tags``; non-release tags are ignored."""
    return max((t for t in tags if semver(t)), key=semver, default="")


def parse_ls_remote(text: str) -> dict[str, str]:
    """``tag -> commit`` from ``git ls-remote --tags`` (annotated tags peeled)."""
    commits: dict[str, str] = {}
    for line in text.splitlines():
        sha, _, ref = line.partition("\t")
        name = ref.removeprefix("refs/tags/")
        if name.endswith("^{}") or name not in commits:
            commits[name.removesuffix("^{}")] = sha
    return commits


def run(*cmd: str, **kwargs) -> str:
    return subprocess.run(
        cmd, check=True, capture_output=True, text=True, cwd=ROOT, **kwargs
    ).stdout.strip()


def remote_tags() -> dict[str, str]:
    return parse_ls_remote(run("git", "ls-remote", "--tags", UPSTREAM, "v*"))


def latest_release(tags) -> str:
    """The latest GitHub release tag, else the highest ``v*`` tag."""
    try:
        tag = run(
            "gh",
            "api",
            "repos/libsidplayfp/libresidfp/releases/latest",
            "--jq",
            ".tag_name",
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        tag = ""
    return tag if semver(tag) else highest(tags)


def pinned_tag(tags: dict[str, str]) -> str:
    """The release tag the submodule gitlink points at."""
    sha = run("git", "ls-tree", "HEAD", SUBMODULE).split()[2]
    return highest(t for t, commit in tags.items() if commit == sha)


def bump_patch(text: str) -> str:
    """``text`` with every ``version = "X.Y.Z"`` line's patch component raised."""
    return VERSION_LINE.sub(
        lambda m: f'{m[1]} = "{m[2]}.{m[3]}.{int(m[4]) + 1}"', text, count=1
    )


def check(args) -> int:
    latest = args.latest or latest_release(remote_tags())
    pinned = args.pinned or pinned_tag(remote_tags())
    if not semver(latest) or not semver(pinned):
        sys.exit(f"cannot compare latest={latest!r} with pinned={pinned!r}")
    newer = semver(latest) > semver(pinned)
    print(f"latest={latest}\npinned={pinned}\nnewer={str(newer).lower()}")
    return 0


def bump_version(_args) -> int:
    texts = {name: (ROOT / name).read_text() for name in VERSION_FILES}
    found = {VERSION_LINE.search(text).groups()[1:] for text in texts.values()}
    if len(found) != 1:
        sys.exit(f"version mismatch across {VERSION_FILES}: {sorted(found)}")
    major, minor, patch = found.pop()
    for name, text in texts.items():
        (ROOT / name).write_text(bump_patch(text))
    print(f"{major}.{minor}.{patch} -> {major}.{minor}.{int(patch) + 1}")
    return 0


def patches(_args) -> int:
    with tempfile.TemporaryDirectory() as scratch:
        shutil.copytree(ROOT / SUBMODULE / "src", Path(scratch) / "src")
        for patch in sorted((ROOT / "patches").glob("*.patch")):
            done = subprocess.run(
                ["patch", "-p1", "-i", str(patch)], cwd=scratch, check=False
            )
            if done.returncode:
                print(f"{patch.name} does not apply", file=sys.stderr)
                return 1
            print(f"{patch.name} applies")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    chk = sub.add_parser("check", help="compare the latest release with the pin")
    chk.add_argument("--latest", help="override the latest release tag")
    chk.add_argument("--pinned", help="override the pinned tag")
    chk.set_defaults(func=check)
    sub.add_parser("bump-version").set_defaults(func=bump_version)
    sub.add_parser("patches").set_defaults(func=patches)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
