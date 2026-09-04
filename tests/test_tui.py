"""The build monitor's pure logic: no terminal, no live rendering."""

import pytest
from rich.console import Console

from mcp_manual_walker.progress import (
    STAGE_CONVERTING,
    STAGE_DONE,
    STAGE_INGESTING,
    STAGE_QUEUED,
    STAGE_SCANNING,
    STAGE_SKIPPED,
    RunState,
    reduce_events,
)
from mcp_manual_walker.tui import (
    FILTERS,
    BuildMonitor,
    convert_started_at,
    file_elapsed,
    format_duration,
    render_plain,
    watch,
)


def stage(path, name, ts=100.0, **fields):
    return {"event": "stage", "ts": ts, "path": path, "stage": name, **fields}


@pytest.fixture
def monitor():
    run = reduce_events(
        [
            {"event": "run_start", "ts": 0.0, "total": 5, "workers": 3},
            stage("a.pdf", STAGE_DONE, pages=10),
            stage("b.pdf", STAGE_CONVERTING, pages=20),
            stage("c.pdf", STAGE_INGESTING, pages=30),
            stage("d.pdf", STAGE_QUEUED, pages=40),
            stage("e.pdf", STAGE_SKIPPED, pages=50),
        ]
    )
    monitor = BuildMonitor(run, "progress.jsonl")
    monitor.rows_visible = 2
    monitor.follow = False
    return monitor


# -- formatting --------------------------------------------------------------


@pytest.mark.parametrize(
    "seconds,expected",
    [
        (0, "0:00"),
        (9, "0:09"),
        (61, "1:01"),
        (3599, "59:59"),
        (3600, "1:00:00"),
        (45296, "12:34:56"),
    ],
)
def test_format_duration(seconds, expected):
    assert format_duration(seconds) == expected


def test_format_duration_of_an_unknown_span_is_blank():
    assert format_duration(None) == ""
    assert format_duration(-5) == ""


def test_a_finished_file_reports_its_conversion_time_not_its_idle_time():
    # After the run moves on, "time" must keep meaning how long the file took,
    # not how long ago it finished.
    run = reduce_events(
        [
            stage("a.pdf", STAGE_CONVERTING, ts=100.0),
            stage("a.pdf", STAGE_DONE, ts=160.0),
        ]
    )
    assert file_elapsed(run.files["a.pdf"], now=9999.0) == 60.0


def test_an_in_flight_file_reports_time_in_its_current_stage():
    run = reduce_events([stage("a.pdf", STAGE_CONVERTING, ts=100.0)])
    assert file_elapsed(run.files["a.pdf"], now=130.0) == 30.0


def test_conversion_start_is_the_earliest_conversion():
    run = reduce_events(
        [
            stage("a.pdf", STAGE_CONVERTING, ts=300.0),
            stage("b.pdf", STAGE_CONVERTING, ts=200.0),
        ]
    )
    assert convert_started_at(run) == 200.0


def test_conversion_start_is_unknown_before_anything_converts():
    assert convert_started_at(reduce_events([stage("a.pdf", STAGE_QUEUED)])) is None


# -- filtering and scrolling -------------------------------------------------


def test_the_default_view_lists_every_file(monitor):
    assert len(monitor.visible_files()) == 5


def test_active_filter_keeps_only_work_in_flight(monitor):
    monitor.filter = "active"
    assert [f.path for f in monitor.visible_files()] == ["b.pdf", "c.pdf"]


def test_unfinished_filter_drops_done_and_skipped(monitor):
    monitor.filter = "unfinished"
    assert [f.path for f in monitor.visible_files()] == ["b.pdf", "c.pdf", "d.pdf"]


def test_a_cycles_through_the_filters_and_returns(monitor):
    for expected in FILTERS[1:] + FILTERS[:1]:
        monitor.handle_key("a")
        assert monitor.filter == expected


def test_scrolling_stops_at_the_top(monitor):
    monitor.handle_key("k")
    monitor.clamp(monitor.visible_files())
    assert monitor.cursor == 0


def test_scrolling_stops_before_running_off_the_bottom(monitor):
    monitor.handle_key("G")
    monitor.clamp(monitor.visible_files())
    # 5 files, 2 rows on screen: the last full screen starts at index 3.
    assert monitor.cursor == 3


def test_paging_moves_by_a_screen_less_one_row(monitor):
    monitor.handle_key("PGDN")
    assert monitor.cursor == 1


def test_arrow_keys_scroll_like_j_and_k(monitor):
    monitor.handle_key("DOWN")
    assert monitor.cursor == 1
    monitor.handle_key("UP")
    assert monitor.cursor == 0


def test_scrolling_turns_following_off(monitor):
    monitor.follow = True
    monitor.handle_key("j")
    assert monitor.follow is False


def test_f_toggles_following(monitor):
    monitor.follow = False
    monitor.handle_key("f")
    assert monitor.follow is True


def test_q_and_ctrl_c_ask_to_quit(monitor):
    assert monitor.handle_key("q") is False
    assert monitor.handle_key("\x03") is False
    assert monitor.handle_key("j") is True


def test_following_scrolls_to_the_work_in_flight(monitor):
    monitor.follow = True
    files = monitor.visible_files()
    # b.pdf is at index 1; the view opens one row above it.
    assert monitor.follow_target(files) == 0


def test_following_tracks_the_scan_front_before_conversion_starts():
    run = reduce_events(
        [stage(f"{i}.pdf", STAGE_QUEUED) for i in range(10)]
        + [stage("8.pdf", STAGE_SCANNING)]
    )
    monitor = BuildMonitor(run, "p.jsonl")
    monitor.rows_visible = 3
    assert monitor.follow_target(monitor.visible_files()) == 7


def test_following_holds_position_when_nothing_is_in_flight():
    run = reduce_events([stage(f"{i}.pdf", STAGE_QUEUED) for i in range(10)])
    monitor = BuildMonitor(run, "p.jsonl")
    monitor.cursor = 4
    assert monitor.follow_target(monitor.visible_files()) == 4


# -- rendering ---------------------------------------------------------------


def test_render_produces_a_screenful_without_touching_a_terminal(monitor):
    console = Console(width=100, height=20, record=True)
    console.print(monitor.render(height=20))
    out = console.export_text()
    assert "converting" in out and "b.pdf" in out
    # Percentage over pages, excluding the skipped file: 10 of 100.
    assert "10.0%" in out


def test_render_marks_a_finished_run(monitor):
    monitor.run.finished_at = 500.0
    console = Console(width=100, height=20, record=True)
    console.print(monitor.render(height=20))
    assert "finished" in console.export_text()


def test_an_error_is_shown_next_to_its_file():
    run = reduce_events(
        [{"event": "stage", "ts": 1.0, "path": "a.pdf", "stage": "failed",
          "error": "RuntimeError: kaboom"}]
    )
    monitor = BuildMonitor(run, "p.jsonl")
    console = Console(width=120, height=20, record=True)
    console.print(monitor.render(height=20))
    assert "kaboom" in console.export_text()


def test_render_survives_a_run_that_has_produced_no_events():
    monitor = BuildMonitor(RunState(), "p.jsonl")
    console = Console(width=100, height=20, record=True)
    console.print(monitor.render(height=20))


def test_render_survives_a_terminal_too_short_for_the_chrome(monitor):
    console = Console(width=100, height=6, record=True)
    console.print(monitor.render(height=6))


def test_plain_render_lists_every_file(monitor):
    console = Console(width=120, record=True)
    render_plain(monitor.run, console)
    out = console.export_text()
    assert all(name in out for name in ("a.pdf", "b.pdf", "e.pdf"))


# -- the watch entry point ---------------------------------------------------


def test_watch_reports_a_missing_progress_file(tmp_path, capsys):
    assert watch(tmp_path / "nope.jsonl", once=True) == 1
    assert "No progress file" in capsys.readouterr().out


def test_watch_once_exits_zero_on_a_clean_run(tmp_path, capsys):
    log = tmp_path / "p.jsonl"
    log.write_text(
        '{"event": "run_start", "ts": 1.0, "total": 1}\n'
        '{"event": "stage", "ts": 2.0, "path": "a.pdf", "stage": "done"}\n'
        '{"event": "run_end", "ts": 3.0, "converted": 1, "failed": 0}\n'
    )
    assert watch(log, once=True) == 0
    assert "a.pdf" in capsys.readouterr().out


def test_watch_once_exits_nonzero_when_a_file_failed(tmp_path, capsys):
    log = tmp_path / "p.jsonl"
    log.write_text(
        '{"event": "stage", "ts": 2.0, "path": "a.pdf", "stage": "failed",'
        ' "error": "boom"}\n'
    )
    assert watch(log, once=True) == 1
