"""Where a test case's baseline and its latest replay live, and promoting one
to the other (FR-P9).

    <store>/cassettes/<case>.jsonl        the baseline, committed to git
    <store>/cassettes/<case>/<n>-<name>.jsonl  its child runs, in start order
    <store>/replays/<case>.jsonl          the latest replay, for fr accept
"""

import re
import shutil
from contextlib import closing
from pathlib import Path

from flight_recorder.errors import CassetteError
from flight_recorder.model import Run, Step
from flight_recorder.store import index
from flight_recorder.store.cassette import load

Recording = tuple[Run, list[Step]]
_CHILD = re.compile(r"(\d+)-.*\.jsonl")


def baseline_path(root: Path, case: str) -> Path:
    return root / "cassettes" / f"{case}.jsonl"


def replay_path(root: Path, case: str) -> Path:
    return root / "replays" / f"{case}.jsonl"


def children(path: Path) -> list[Path]:
    """A case file's child run files, in start order."""
    folder = path.with_suffix("")
    found = [p for p in folder.glob("*.jsonl") if _CHILD.fullmatch(p.name)]
    return sorted(found, key=lambda p: int(_CHILD.fullmatch(p.name)[1]))  # type: ignore[index]


def load_case(path: Path) -> tuple[Recording, list[Recording]]:
    """The case run at ``path`` and its child runs."""
    return load(path), [load(p) for p in children(path)]


def forget(root: Path, path: Path) -> None:
    """Delete a case file's child runs, and their index rows, before it is
    written again, so a sub-agent run that no longer happens leaves nothing."""
    stale = children(path)
    for child in stale:
        child.unlink()
    if stale:
        with closing(index.connect(root / "index.sqlite")) as db, db:
            for child in stale:
                index.forget(db, child.relative_to(root).as_posix())


def accept(root: Path, case: str) -> Path:
    """Make the case's latest replay its baseline; returns the baseline path."""
    source, target = replay_path(root, case), baseline_path(root, case)
    if not source.is_file():
        raise CassetteError(f"no replay of {case!r} at {source}; run pytest first")
    forget(root, target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    for child in children(source):
        dest = target.with_suffix("") / child.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(child, dest)
    index.rebuild(root)  # the copies carry the replay's run ids
    return target
