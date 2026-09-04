"""
Per-file build progress, written as an append-only JSONL event log.

The build runs across three process pools, so there is no single place in
memory that knows where every PDF currently is. Instead each process appends
one small JSON object per state transition to a shared file, and a reader
folds those events back into the current state of the run. That keeps the
producers trivial (one line, no locks, no shared objects that would have to
survive "spawn") and lets a monitor attach, detach and re-attach at will --
including after the build has finished, since the file is the whole history.

Ordering across processes does not have to be perfect: every event carries the
path it is about, and the reducer applies them in the order they were written.

Two properties this module has to preserve:

* **Nothing here may break a build.** Every entry point swallows its own
  exceptions; a full disk or an unwritable path costs progress reporting, not
  the conversion.
* **It stays import-light.** The metadata pool deliberately imports only pypdf
  (see :func:`mcp_manual_walker.pdf_utils.extract_pdf_fingerprint`), so this
  module uses nothing outside the standard library.

Writes are single ``write()`` calls to a file opened ``O_APPEND``, which the
kernel keeps atomic below PIPE_BUF; the lines are far smaller than that, and
:func:`_sanitize` keeps error text from making one that is not.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

# Passed to worker processes through the environment. The pools use the
# "spawn" start method, so a module-level global set in the parent would not
# survive; os.environ does.
PROGRESS_FILE_ENV = "MANUAL_WALKER_PROGRESS_FILE"
PROGRESS_ROOT_ENV = "MANUAL_WALKER_PROGRESS_ROOT"

# Bytes of the file head the reader compares between polls to notice that a
# new build has truncated the log out from under it. One run_start line.
HEAD_BYTES = 512

# Longest error text kept in an event, so one failure cannot produce a line
# big enough to be split by a concurrent append.
MAX_ERROR_CHARS = 400

# The stages a file moves through, in the order the pipeline visits them.
# "skipped" and "failed" are terminal and can be reached from several points.
STAGE_PENDING = "pending"
STAGE_SCANNING = "scanning"
STAGE_SCANNED = "scanned"
STAGE_QUEUED = "queued"
STAGE_CONVERTING = "converting"
# Converted, but the parent has not picked the result up yet. The parent
# collects one finished document at a time while it embeds and writes the
# previous one, so without this a finished file would look like it was still
# converting -- and look stuck, since its clock keeps running.
STAGE_CONVERTED = "converted"
STAGE_INGESTING = "ingesting"
STAGE_DONE = "done"
STAGE_SKIPPED = "skipped"
STAGE_FAILED = "failed"

STAGE_ORDER = (
    STAGE_PENDING,
    STAGE_SCANNING,
    STAGE_SCANNED,
    STAGE_QUEUED,
    STAGE_CONVERTING,
    STAGE_CONVERTED,
    STAGE_INGESTING,
    STAGE_DONE,
    STAGE_SKIPPED,
    STAGE_FAILED,
)

TERMINAL_STAGES = frozenset({STAGE_DONE, STAGE_SKIPPED, STAGE_FAILED})
ACTIVE_STAGES = frozenset({STAGE_CONVERTING, STAGE_CONVERTED, STAGE_INGESTING})


def configure(progress_file: Path | str | None, root: Path | str | None = None) -> None:
    """
    Points this process (and any it spawns) at ``progress_file``.

    ``root`` is the directory file paths are reported relative to, so that the
    events carry the same identifier the database stores. Passing ``None`` for
    the file turns reporting off again.
    """
    if progress_file is None:
        os.environ.pop(PROGRESS_FILE_ENV, None)
        os.environ.pop(PROGRESS_ROOT_ENV, None)
        return
    os.environ[PROGRESS_FILE_ENV] = str(progress_file)
    if root is not None:
        os.environ[PROGRESS_ROOT_ENV] = str(root)


def is_enabled() -> bool:
    return bool(os.environ.get(PROGRESS_FILE_ENV))


def relative_path(pdf_path: Path | str) -> str:
    """The identifier a file is reported under: its path relative to the root."""
    path = Path(pdf_path)
    root = os.environ.get(PROGRESS_ROOT_ENV)
    if root:
        try:
            return path.relative_to(root).as_posix()
        except ValueError:
            pass
    return path.name


def _sanitize(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str) and len(value) > MAX_ERROR_CHARS:
        return value[:MAX_ERROR_CHARS] + "..."
    return value


def emit(event: str, **fields: Any) -> None:
    """Appends one event. Never raises, and does nothing when unconfigured."""
    target = os.environ.get(PROGRESS_FILE_ENV)
    if not target:
        return
    try:
        payload = {"ts": time.time(), "event": event}
        payload.update({k: _sanitize(v) for k, v in fields.items() if v is not None})
        line = json.dumps(payload, ensure_ascii=False) + "\n"
        with open(target, "a", encoding="utf-8") as handle:
            handle.write(line)
    except Exception:
        # Progress reporting is never worth failing a conversion over.
        pass


def emit_file(pdf_path: Path | str, stage: str, **fields: Any) -> None:
    """Records that ``pdf_path`` has entered ``stage``."""
    emit("stage", path=relative_path(pdf_path), stage=stage, **fields)


@dataclass
class FileState:
    """Where one PDF currently is, folded from every event about it."""

    path: str
    stage: str = STAGE_PENDING
    stage_since: float = 0.0
    size: int | None = None
    pages: int | None = None
    # Pages that have left the last pipeline stage, for the file being
    # converted right now. Reset when it enters "converting" so a re-run does
    # not start from the previous attempt's count.
    pages_done: int = 0
    chunks: int | None = None
    figures: int | None = None
    worker: int | None = None
    error: str | None = None
    # Wall clock spent converting: set when the file leaves "converting".
    convert_started: float | None = None
    finished_at: float | None = None

    @property
    def is_terminal(self) -> bool:
        return self.stage in TERMINAL_STAGES

    @property
    def fraction(self) -> float | None:
        """How far through the document the conversion is, 0.0-1.0.

        None until there is both a page count (from the scan) and a page
        report. Clamped, because Docling counts the pages it actually
        processed and pypdf counts the pages in the file; a malformed page can
        make them disagree, and a bar past 100% reads as a bug in the build.
        """
        if not self.pages or not self.pages_done:
            return None
        return min(1.0, self.pages_done / self.pages)

    @property
    def is_active(self) -> bool:
        return self.stage in ACTIVE_STAGES


@dataclass
class RunState:
    """The whole run, as reconstructed from the event log."""

    started_at: float | None = None
    finished_at: float | None = None
    last_event_at: float | None = None
    pdf_dir: str | None = None
    include: list[str] = field(default_factory=list)
    workers: int | None = None
    reset: bool = False
    min_pages: int | None = None
    max_pages: int | None = None
    total: int = 0
    files: dict[str, FileState] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    @property
    def page_range(self) -> str:
        """The run's --min-pages/--max-pages as a label, empty when unbounded."""
        if self.min_pages is None and self.max_pages is None:
            return ""
        low = f"{self.min_pages:,}" if self.min_pages is not None else ""
        high = f"{self.max_pages:,}" if self.max_pages is not None else ""
        return f"{low}-{high}"

    def ordered(self) -> list[FileState]:
        """Files in discovery order (largest first, as the builder submits)."""
        return [self.files[path] for path in self.order]

    def counts(self) -> dict[str, int]:
        counts = {stage: 0 for stage in STAGE_ORDER}
        for state in self.files.values():
            counts[state.stage] = counts.get(state.stage, 0) + 1
        return counts

    def pages_done(self) -> int:
        """Pages converted so far, counting partial progress on files in flight.

        Without the in-flight part the bar sits still for the length of a
        2900-page manual, which is exactly when someone is watching it.
        """
        total = 0
        for state in self.files.values():
            if state.stage in (STAGE_DONE, STAGE_CONVERTED, STAGE_INGESTING):
                total += state.pages or 0
            elif state.stage == STAGE_CONVERTING:
                total += min(state.pages_done, state.pages or state.pages_done)
        return total

    def pages_total(self) -> int:
        """Pages of everything still expected to convert, plus what is done.

        Skipped files are excluded: they are already in the database and cost
        no conversion time, so counting them would flatten the rate estimate.
        """
        return sum(
            f.pages or 0 for f in self.files.values() if f.stage != STAGE_SKIPPED
        )

    @property
    def is_finished(self) -> bool:
        return self.finished_at is not None


def _state_for(run: RunState, path: str) -> FileState:
    state = run.files.get(path)
    if state is None:
        state = FileState(path=path)
        run.files[path] = state
        run.order.append(path)
    return state


def apply_event(run: RunState, event: dict[str, Any]) -> RunState:
    """Folds a single event into ``run``, in place."""
    kind = event.get("event")
    ts = event.get("ts")
    if isinstance(ts, (int, float)):
        run.last_event_at = ts

    if kind == "run_start":
        run.started_at = ts
        run.finished_at = None
        run.pdf_dir = event.get("pdf_dir")
        run.include = list(event.get("include") or [])
        run.workers = event.get("workers")
        run.reset = bool(event.get("reset"))
        run.total = event.get("total") or 0
        run.min_pages = event.get("min_pages")
        run.max_pages = event.get("max_pages")
        return run

    if kind == "run_end":
        run.finished_at = ts
        run.summary = {
            key: value for key, value in event.items() if key not in ("ts", "event")
        }
        return run

    if kind not in ("discovered", "stage", "page"):
        return run
    path = event.get("path")
    if not isinstance(path, str):
        return run
    state = _state_for(run, path)

    if kind == "discovered":
        state.size = event.get("size", state.size)
        return run

    if kind == "page":
        done = event.get("pages_done")
        if isinstance(done, int):
            # Page events can be reordered against each other by three
            # processes appending at once; only ever move forward.
            state.pages_done = max(state.pages_done, done)
        return run

    stage = event.get("stage")
    if stage:
        if stage == STAGE_CONVERTING:
            state.convert_started = ts
            state.pages_done = 0
        if stage in TERMINAL_STAGES:
            state.finished_at = ts
        state.stage = stage
        if isinstance(ts, (int, float)):
            state.stage_since = ts

    for key in ("pages", "chunks", "figures", "worker", "size"):
        if event.get(key) is not None:
            setattr(state, key, event[key])
    # An error belongs to the transition that reported it; a later successful
    # stage (a retry, a rebuild) clears it.
    if event.get("error") is not None:
        state.error = event["error"]
    elif stage and stage not in TERMINAL_STAGES:
        state.error = None
    return run


def parse_lines(lines: Iterable[str]) -> Iterator[dict[str, Any]]:
    """Yields the well-formed events out of raw log lines, skipping junk."""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict):
            yield event


def reduce_events(events: Iterable[dict[str, Any]]) -> RunState:
    run = RunState()
    for event in events:
        apply_event(run, event)
    return run


class ProgressReader:
    """
    Incremental reader over a progress file.

    Keeps a byte offset so each poll only parses what was appended since the
    last one, and holds back a trailing fragment until its newline shows up --
    otherwise a poll landing mid-append would drop that event.

    A new build truncates the file and writes a fresh log into it. Comparing
    sizes is not enough to notice that -- by the next poll the new run may
    already be longer than the old offset -- so the reader also keeps the first
    bytes of the file and restarts whenever they stop being a prefix of what it
    reads now. The leading ``run_start`` carries a timestamp, so two runs never
    share a head.
    """

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.run = RunState()
        self._offset = 0
        self._partial = ""
        self._head = b""
        self.exists = False

    def _restart(self) -> None:
        self.run = RunState()
        self._offset = 0
        self._partial = ""

    def poll(self) -> RunState:
        try:
            size = self.path.stat().st_size
        except OSError:
            self.exists = False
            return self.run
        self.exists = True

        try:
            with open(self.path, "rb") as handle:
                head = handle.read(HEAD_BYTES)
        except OSError:
            return self.run

        if size < self._offset or not head.startswith(self._head):
            self._restart()
        self._head = head

        if size == self._offset:
            return self.run

        try:
            with open(self.path, "r", encoding="utf-8", errors="replace") as handle:
                handle.seek(self._offset)
                chunk = handle.read()
                self._offset = handle.tell()
        except OSError:
            return self.run

        buffer = self._partial + chunk
        if buffer.endswith("\n"):
            self._partial = ""
        else:
            buffer, _, self._partial = buffer.rpartition("\n")

        for event in parse_lines(buffer.splitlines()):
            apply_event(self.run, event)
        return self.run


def read_progress(path: Path | str) -> RunState:
    """Reads a whole progress file in one go."""
    return ProgressReader(path).poll()
