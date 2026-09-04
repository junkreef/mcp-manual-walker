"""
A terminal monitor for a running build.

Reads the JSONL event log written by :mod:`mcp_manual_walker.progress` and
renders it as a live list: every PDF in the run, the stage it is currently in,
and how long it has been there. It is a reader only -- it never writes to the
log and never touches the build -- so it can be started, stopped and restarted
at any point, including after the build has finished, when the log doubles as
the run's history.
"""

from __future__ import annotations

import queue
import sys
import threading
import time
from pathlib import Path

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from mcp_manual_walker.progress import (
    STAGE_CONVERTED,
    STAGE_CONVERTING,
    STAGE_DONE,
    STAGE_FAILED,
    STAGE_INGESTING,
    STAGE_PENDING,
    STAGE_QUEUED,
    STAGE_SCANNED,
    STAGE_SCANNING,
    STAGE_SKIPPED,
    ProgressReader,
    RunState,
)

REFRESH_SECONDS = 0.5

# A run whose last event is older than this, with no run_end, has almost
# certainly died (a killed process, a broken worker pool) rather than gone
# quiet: a single conversion still emits nothing for minutes, so the threshold
# has to be generous.
STALE_AFTER_SECONDS = 900

# Marker, label and colour per stage. The order here is also the order the
# counters appear in the header.
STAGE_STYLE = {
    STAGE_PENDING: ("·", "pending", "grey42"),
    STAGE_SCANNING: ("◌", "scanning", "cyan"),
    STAGE_SCANNED: ("◍", "scanned", "cyan"),
    STAGE_QUEUED: ("◇", "queued", "yellow"),
    STAGE_CONVERTING: ("▶", "converting", "bright_blue"),
    STAGE_CONVERTED: ("⇢", "converted", "cyan"),
    STAGE_INGESTING: ("◆", "ingesting", "magenta"),
    STAGE_DONE: ("✓", "done", "green"),
    STAGE_SKIPPED: ("=", "skipped", "grey50"),
    STAGE_FAILED: ("✗", "failed", "bright_red"),
}

FILTERS = ("all", "active", "unfinished")
FILTER_HELP = {
    "all": "every file",
    "active": "converting/ingesting only",
    "unfinished": "everything not yet done or skipped",
}


def format_duration(seconds: float | None) -> str:
    """h:mm:ss, or m:ss below an hour. Empty for an unknown duration."""
    if seconds is None or seconds < 0:
        return ""
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def convert_started_at(run: RunState) -> float | None:
    """When the first conversion began, i.e. when the GPU work actually started."""
    starts = [f.convert_started for f in run.files.values() if f.convert_started]
    return min(starts) if starts else None


# Stages where a clock says something. Time spent queued is time since the
# whole batch was queued -- the same number on every row, which reads like a
# per-file measurement and is not one.
TIMED_STAGES = frozenset(
    {
        STAGE_SCANNING,
        STAGE_CONVERTING,
        STAGE_CONVERTED,
        STAGE_INGESTING,
        STAGE_DONE,
        STAGE_FAILED,
    }
)


def file_elapsed(state, now: float) -> float | None:
    """Time the file took, or has been taking, where that is meaningful."""
    if state.stage not in TIMED_STAGES:
        return None
    if state.convert_started and state.stage in (STAGE_DONE, STAGE_FAILED):
        # The whole conversion, not just the last leg of it.
        return (state.finished_at or now) - state.convert_started
    if state.stage_since:
        end = state.finished_at if state.is_terminal else now
        return (end or now) - state.stage_since
    return None


def progress_bar(fraction: float, width: int = 28, style: str = "green") -> Text:
    filled = max(0, min(width, round(fraction * width)))
    bar = Text("█" * filled, style=style)
    bar.append("░" * (width - filled), style="grey35")
    return bar


def file_progress(state) -> Text:
    """The per-document bar: only meaningful while a document is converting."""
    if state.stage == STAGE_CONVERTING:
        fraction = state.fraction
        if fraction is None:
            # Pages are reported in batches, so the first one takes a moment.
            return Text("      ...", style="grey42")
        cell = progress_bar(fraction, width=8, style="bright_blue")
        cell.append(f" {fraction * 100:3.0f}%", style="bright_blue")
        return cell
    if state.stage in (STAGE_CONVERTED, STAGE_INGESTING):
        return Text("        100%", style="grey50")
    return Text("")


class BuildMonitor:
    """Folds the event log into the two panels that make up the display."""

    def __init__(self, run: RunState, path: Path):
        self.run = run
        self.path = path
        self.cursor = 0
        self.follow = True
        self.filter = "all"
        self.rows_visible = 10

    # -- selection ---------------------------------------------------------
    def visible_files(self) -> list:
        files = self.run.ordered()
        if self.filter == "active":
            return [f for f in files if f.is_active]
        if self.filter == "unfinished":
            return [f for f in files if f.stage not in (STAGE_DONE, STAGE_SKIPPED)]
        return files

    def follow_target(self, files: list) -> int:
        """Scroll position that keeps the working set on screen."""
        active = [i for i, f in enumerate(files) if f.is_active]
        if not active:
            # Before anything converts, follow the scan front instead.
            active = [
                i
                for i, f in enumerate(files)
                if f.stage in (STAGE_SCANNING, STAGE_INGESTING)
            ]
        if not active:
            return self.cursor
        return max(0, min(active) - 1)

    def clamp(self, files: list) -> None:
        highest = max(0, len(files) - self.rows_visible)
        self.cursor = max(0, min(self.cursor, highest))

    # -- key handling ------------------------------------------------------
    def handle_key(self, key: str) -> bool:
        """Applies one keypress. Returns False when the user asked to quit."""
        page = max(1, self.rows_visible - 1)
        if key in ("q", "\x03", "\x04"):
            return False
        if key in ("j", "DOWN"):
            self.cursor += 1
            self.follow = False
        elif key in ("k", "UP"):
            self.cursor -= 1
            self.follow = False
        elif key in ("PGDN", " "):
            self.cursor += page
            self.follow = False
        elif key == "PGUP":
            self.cursor -= page
            self.follow = False
        elif key == "g":
            self.cursor = 0
            self.follow = False
        elif key == "G":
            self.cursor = 10**9
            self.follow = False
        elif key == "f":
            self.follow = not self.follow
        elif key == "a":
            self.filter = FILTERS[(FILTERS.index(self.filter) + 1) % len(FILTERS)]
            self.cursor = 0
        return True

    # -- rendering ---------------------------------------------------------
    def header(self, now: float) -> Panel:
        run = self.run
        counts = run.counts()

        title = Text()
        title.append(run.pdf_dir or str(self.path), style="bold")
        if run.include:
            title.append("  include=", style="grey62")
            title.append(" ".join(run.include), style="bold yellow")
        if run.page_range:
            title.append("  pages=", style="grey62")
            title.append(run.page_range, style="bold yellow")
        if run.workers:
            title.append(f"  {run.workers} worker(s)", style="grey62")
        if run.reset:
            title.append("  --reset", style="bright_red")

        line = Text()
        for stage in (
            STAGE_DONE,
            STAGE_CONVERTING,
            STAGE_CONVERTED,
            STAGE_INGESTING,
            STAGE_QUEUED,
            STAGE_SCANNING,
            STAGE_SCANNED,
            STAGE_PENDING,
            STAGE_SKIPPED,
            STAGE_FAILED,
        ):
            count = counts.get(stage, 0)
            # Only the two that matter even at zero: "done" anchors the run,
            # and "failed" must never be something the eye has to hunt for.
            if not count and stage not in (STAGE_DONE, STAGE_FAILED):
                continue
            _, label, colour = STAGE_STYLE[stage]
            if line:
                line.append("   ")
            line.append(f"{label} ", style="grey62")
            line.append(str(count), style=f"bold {colour}")

        pages_done = run.pages_done()
        pages_total = run.pages_total()
        fraction = pages_done / pages_total if pages_total else 0.0

        bar = Text()
        bar.append_text(progress_bar(fraction))
        bar.append(f" {fraction * 100:5.1f}%", style="bold")
        bar.append(f"  {pages_done:,}/{pages_total:,} pages", style="grey62")

        started = convert_started_at(run) or run.started_at
        if run.is_finished and run.last_event_at and started:
            elapsed = run.last_event_at - started
        elif started:
            elapsed = now - started
        else:
            elapsed = None
        bar.append(f"   elapsed {format_duration(elapsed)}", style="grey62")

        if run.is_finished:
            bar.append("   finished", style="bold green")
        elif pages_done and elapsed and elapsed > 0:
            rate = pages_done / elapsed
            remaining = max(0, pages_total - pages_done)
            bar.append(f"   {rate * 60:.0f} pages/min", style="grey62")
            bar.append(f"   ETA {format_duration(remaining / rate)}", style="bold")
        elif run.last_event_at and now - run.last_event_at > STALE_AFTER_SECONDS:
            bar.append("   stalled?", style="bold bright_red")

        return Panel(
            Group(title, line, bar),
            title="mcp-manual-walker build",
            border_style="blue",
            padding=(0, 1),
        )

    def table(self, files: list, now: float) -> Table:
        table = Table(
            expand=True, box=None, pad_edge=False, show_edge=False, padding=(0, 1)
        )
        table.add_column("#", justify="right", style="grey42", width=4)
        # Wide enough for the longest marker + label ("▶ converting").
        table.add_column("stage", width=12, no_wrap=True)
        table.add_column("pages", justify="right", width=6, no_wrap=True)
        table.add_column("progress", width=13, no_wrap=True)
        table.add_column("time", justify="right", width=8, no_wrap=True)
        table.add_column("chunks", justify="right", width=7, no_wrap=True)
        table.add_column("figs", justify="right", width=5, no_wrap=True)
        table.add_column("file", ratio=1, no_wrap=True, overflow="ellipsis")

        window = files[self.cursor : self.cursor + self.rows_visible]
        for offset, state in enumerate(window):
            marker, label, colour = STAGE_STYLE.get(state.stage, ("?", "?", "white"))
            row_style = "bold" if state.is_active else None
            name = Text(state.path, style=row_style or "")
            if state.error:
                name.append(f"  {state.error.splitlines()[0]}", style="bright_red")
            table.add_row(
                str(self.cursor + offset + 1),
                Text(f"{marker} {label}", style=colour),
                f"{state.pages:,}" if state.pages else "",
                file_progress(state),
                format_duration(file_elapsed(state, now)),
                f"{state.chunks:,}" if state.chunks else "",
                str(state.figures) if state.figures else "",
                name,
                style=row_style,
            )
        return table

    def footer(self, files: list) -> Text:
        shown_from = self.cursor + 1 if files else 0
        shown_to = min(len(files), self.cursor + self.rows_visible)
        text = Text()
        text.append(f" {shown_from}-{shown_to} of {len(files)} ", style="grey62")
        text.append(f"[{self.filter}: {FILTER_HELP[self.filter]}] ", style="cyan")
        if self.follow:
            text.append("following ", style="green")
        text.append(
            " j/k scroll · PgUp/PgDn page · g/G ends · f follow · a filter · q quit",
            style="grey42",
        )
        return text

    def render(self, height: int) -> Group:
        now = time.time()
        # Header (5 lines incl. borders), footer, and the table header row.
        self.rows_visible = max(3, height - 8)
        files = self.visible_files()
        if self.follow:
            self.cursor = self.follow_target(files)
        self.clamp(files)
        return Group(self.header(now), self.table(files, now), self.footer(files))


class KeyReader:
    """
    Reads single keypresses from a TTY on a background thread.

    Put in raw mode so keys arrive without a newline; arrow keys and page keys
    come in as escape sequences and are translated to names. On a non-TTY this
    is inert, which is what makes the monitor safe to pipe to a file.
    """

    ESCAPES = {
        "[A": "UP",
        "[B": "DOWN",
        "[C": "RIGHT",
        "[D": "LEFT",
        "[5~": "PGUP",
        "[6~": "PGDN",
        "[H": "g",
        "[F": "G",
    }

    def __init__(self):
        self.queue: queue.Queue[str] = queue.Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._settings = None
        self.enabled = sys.stdin.isatty()

    def __enter__(self) -> KeyReader:
        if not self.enabled:
            return self
        import termios
        import tty

        self._settings = termios.tcgetattr(sys.stdin.fileno())
        tty.setcbreak(sys.stdin.fileno())
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._settings is not None:
            import termios

            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._settings)

    def _loop(self) -> None:
        import select

        while not self._stop.is_set():
            if not select.select([sys.stdin], [], [], 0.2)[0]:
                continue
            char = sys.stdin.read(1)
            if not char:
                break
            if char == "\x1b":
                sequence = ""
                while select.select([sys.stdin], [], [], 0.02)[0]:
                    sequence += sys.stdin.read(1)
                    if sequence in self.ESCAPES:
                        break
                    if len(sequence) > 4:
                        break
                char = self.ESCAPES.get(sequence, "")
            if char:
                self.queue.put(char)

    def drain(self) -> list[str]:
        keys = []
        while True:
            try:
                keys.append(self.queue.get_nowait())
            except queue.Empty:
                return keys


def render_plain(run: RunState, console: Console) -> None:
    """One-shot text dump, for a non-interactive terminal or --once."""
    counts = run.counts()
    console.print(
        f"{run.pdf_dir or ''}  "
        + "  ".join(
            f"{STAGE_STYLE[stage][1]}={counts.get(stage, 0)}"
            for stage in STAGE_STYLE
            if counts.get(stage, 0)
        )
    )
    for state in run.ordered():
        marker, label, _ = STAGE_STYLE.get(state.stage, ("?", "?", ""))
        pages = f"{state.pages:>6,}" if state.pages else " " * 6
        console.print(
            f"  {marker} {label:<11}{pages}  {state.path}"
            + (f"  {state.error.splitlines()[0]}" if state.error else "")
        )
    if run.summary:
        console.print(
            "  ".join(f"{k}={v}" for k, v in run.summary.items()), style="bold"
        )


def watch(
    progress_file: Path,
    once: bool = False,
    refresh: float = REFRESH_SECONDS,
    exit_when_finished: bool = False,
) -> int:
    """
    Renders ``progress_file`` until the user quits.

    Returns a process exit status: non-zero when the run it watched recorded
    failures, so it can be used in a script.
    """
    console = Console()
    reader = ProgressReader(progress_file)
    run = reader.poll()

    if once or not console.is_terminal:
        if not reader.exists:
            console.print(f"No progress file at {progress_file}", style="yellow")
            return 1
        render_plain(run, console)
        return 1 if run.counts().get(STAGE_FAILED) else 0

    monitor = BuildMonitor(run, progress_file)
    with KeyReader() as keys:
        with Live(
            console=console, screen=True, auto_refresh=False, transient=False
        ) as live:
            while True:
                monitor.run = reader.poll()
                if not reader.exists:
                    live.update(
                        Panel(
                            Text(
                                f"Waiting for {progress_file}\n"
                                "(start a build, or pass --progress-file)",
                                justify="center",
                            ),
                            border_style="yellow",
                        ),
                        refresh=True,
                    )
                else:
                    live.update(monitor.render(console.size.height), refresh=True)
                for key in keys.drain():
                    if not monitor.handle_key(key):
                        return 1 if monitor.run.counts().get(STAGE_FAILED) else 0
                if exit_when_finished and monitor.run.is_finished:
                    break
                time.sleep(refresh)

    render_plain(monitor.run, console)
    return 1 if monitor.run.counts().get(STAGE_FAILED) else 0
