"""The build progress event log and the state it folds into."""

import json
import os
from pathlib import Path

import pytest

from mcp_manual_walker import progress
from mcp_manual_walker.progress import (
    STAGE_CONVERTED,
    STAGE_CONVERTING,
    STAGE_DONE,
    STAGE_FAILED,
    STAGE_INGESTING,
    STAGE_QUEUED,
    STAGE_SCANNED,
    STAGE_SKIPPED,
    ProgressReader,
    read_progress,
    reduce_events,
)


@pytest.fixture
def log(tmp_path, monkeypatch):
    """A configured progress file, with the environment restored afterwards."""
    path = tmp_path / "progress.jsonl"
    monkeypatch.delenv(progress.PROGRESS_FILE_ENV, raising=False)
    monkeypatch.delenv(progress.PROGRESS_ROOT_ENV, raising=False)
    progress.configure(path, tmp_path)
    return path


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


# -- writing -----------------------------------------------------------------


def test_emit_writes_one_json_object_per_line(log):
    progress.emit("run_start", total=2)
    progress.emit("run_end", found=2)
    assert [e["event"] for e in lines(log)] == ["run_start", "run_end"]


def test_emit_stamps_every_event_with_a_time(log):
    progress.emit("run_start")
    assert isinstance(lines(log)[0]["ts"], float)


def test_emit_drops_none_fields_rather_than_writing_nulls(log):
    progress.emit("stage", path="a.pdf", stage=STAGE_QUEUED, pages=None)
    assert "pages" not in lines(log)[0]


def test_emit_file_reports_the_path_relative_to_the_root(log, tmp_path):
    progress.emit_file(tmp_path / "zOS" / "V3R1" / "a.pdf", STAGE_QUEUED)
    assert lines(log)[0]["path"] == "zOS/V3R1/a.pdf"


def test_a_path_outside_the_root_falls_back_to_its_name(log):
    progress.emit_file(Path("/elsewhere/other.pdf"), STAGE_QUEUED)
    assert lines(log)[0]["path"] == "other.pdf"


def test_long_errors_are_truncated_so_a_line_stays_atomically_appendable(log):
    progress.emit_file(Path("a.pdf"), STAGE_FAILED, error="x" * 5000)
    error = lines(log)[0]["error"]
    assert len(error) == progress.MAX_ERROR_CHARS + 3
    assert len(log.read_text()) < 4096


def test_emit_is_a_no_op_when_unconfigured(tmp_path, monkeypatch):
    monkeypatch.delenv(progress.PROGRESS_FILE_ENV, raising=False)
    progress.emit("run_start")
    assert not list(tmp_path.iterdir())
    assert progress.is_enabled() is False


def test_configure_none_turns_reporting_back_off(log):
    progress.configure(None)
    progress.emit("run_start")
    assert not log.exists()


def test_emit_never_raises_when_the_file_cannot_be_written(tmp_path, monkeypatch):
    # A build must not die because progress reporting cannot write.
    monkeypatch.setenv(
        progress.PROGRESS_FILE_ENV, str(tmp_path / "missing-dir" / "p.jsonl")
    )
    progress.emit("run_start")


def test_the_setting_travels_through_the_environment_for_spawned_workers(log):
    # The pools use "spawn", so a module global would not survive; os.environ
    # is what the child actually inherits.
    assert os.environ[progress.PROGRESS_FILE_ENV] == str(log)


# -- reducing ----------------------------------------------------------------


def make(event, **fields):
    fields.setdefault("ts", 100.0)
    return {"event": event, **fields}


def test_run_metadata_comes_from_run_start():
    run = reduce_events(
        [
            make(
                "run_start",
                pdf_dir="/corpus",
                include=["zOS/*"],
                workers=3,
                total=7,
                reset=True,
            )
        ]
    )
    assert (run.pdf_dir, run.include, run.workers, run.total) == (
        "/corpus",
        ["zOS/*"],
        3,
        7,
    )
    assert run.reset is True
    assert run.is_finished is False


def test_files_keep_their_discovery_order():
    run = reduce_events(
        [
            make("discovered", path="big.pdf", size=900),
            make("discovered", path="small.pdf", size=10),
        ]
    )
    assert [f.path for f in run.ordered()] == ["big.pdf", "small.pdf"]


def test_the_last_stage_for_a_file_wins():
    run = reduce_events(
        [
            make("stage", path="a.pdf", stage=STAGE_QUEUED),
            make("stage", path="a.pdf", stage=STAGE_CONVERTING),
            make("stage", path="a.pdf", stage=STAGE_INGESTING),
            make("stage", path="a.pdf", stage=STAGE_DONE, chunks=12),
        ]
    )
    assert run.files["a.pdf"].stage == STAGE_DONE
    assert run.files["a.pdf"].chunks == 12


def test_facts_learned_early_survive_later_stages():
    # pages arrives with the scan, chunks only at the end; neither must erase
    # the other as the file moves on.
    run = reduce_events(
        [
            make("discovered", path="a.pdf", size=500),
            make("stage", path="a.pdf", stage=STAGE_SCANNED, pages=42),
            make("stage", path="a.pdf", stage=STAGE_CONVERTING, worker=77),
            make("stage", path="a.pdf", stage=STAGE_DONE, chunks=3, figures=1),
        ]
    )
    state = run.files["a.pdf"]
    assert (state.size, state.pages, state.worker, state.chunks, state.figures) == (
        500,
        42,
        77,
        3,
        1,
    )


def test_conversion_start_and_finish_times_are_kept():
    run = reduce_events(
        [
            make("stage", path="a.pdf", stage=STAGE_CONVERTING, ts=100.0),
            make("stage", path="a.pdf", stage=STAGE_DONE, ts=160.0),
        ]
    )
    state = run.files["a.pdf"]
    assert state.convert_started == 100.0
    assert state.finished_at == 160.0


def test_an_event_for_an_unannounced_file_still_creates_it():
    # Events can be written by three processes; a stage may land before the
    # parent's "discovered" line does.
    run = reduce_events([make("stage", path="surprise.pdf", stage=STAGE_CONVERTING)])
    assert run.files["surprise.pdf"].stage == STAGE_CONVERTING


def test_counts_and_page_totals():
    run = reduce_events(
        [
            make("stage", path="a.pdf", stage=STAGE_DONE, pages=10),
            make("stage", path="b.pdf", stage=STAGE_CONVERTING, pages=20),
            make("stage", path="c.pdf", stage=STAGE_SKIPPED, pages=1000),
        ]
    )
    counts = run.counts()
    assert (counts[STAGE_DONE], counts[STAGE_CONVERTING], counts[STAGE_SKIPPED]) == (
        1,
        1,
        1,
    )
    assert run.pages_done() == 10
    # Skipped files cost no conversion time, so counting their pages would
    # make the run look permanently stuck at a low percentage.
    assert run.pages_total() == 30


def test_run_end_marks_the_run_finished_and_keeps_the_summary():
    run = reduce_events(
        [make("run_start", total=1), make("run_end", found=1, failed=2, converted=3)]
    )
    assert run.is_finished
    assert run.summary["failed"] == 2
    assert run.summary["converted"] == 3


def test_an_error_is_retained_on_the_failed_stage():
    run = reduce_events(
        [make("stage", path="a.pdf", stage=STAGE_FAILED, error="boom")]
    )
    assert run.files["a.pdf"].error == "boom"


def test_a_rebuild_clears_a_previous_error():
    run = reduce_events(
        [
            make("stage", path="a.pdf", stage=STAGE_FAILED, error="boom"),
            make("stage", path="a.pdf", stage=STAGE_QUEUED),
        ]
    )
    assert run.files["a.pdf"].error is None


def test_unknown_events_are_ignored():
    run = reduce_events([make("something_new", path="a.pdf"), make("run_start")])
    assert run.files == {}


def test_events_without_a_usable_path_are_ignored():
    run = reduce_events([make("stage", stage=STAGE_DONE), make("stage", path=7)])
    assert run.files == {}


# -- reading -----------------------------------------------------------------


def test_read_progress_folds_a_whole_file(log):
    progress.emit("run_start", total=1)
    progress.emit_file(Path("a.pdf"), STAGE_DONE, pages=3)
    run = read_progress(log)
    assert run.total == 1
    assert run.files["a.pdf"].stage == STAGE_DONE


def test_reader_only_parses_what_was_appended_since_the_last_poll(log):
    progress.emit_file(Path("a.pdf"), STAGE_QUEUED)
    reader = ProgressReader(log)
    reader.poll()
    progress.emit_file(Path("b.pdf"), STAGE_QUEUED)
    run = reader.poll()
    assert sorted(run.files) == ["a.pdf", "b.pdf"]


def test_a_partially_written_line_is_held_back_until_its_newline_arrives(log):
    # A poll can land in the middle of another process's append; dropping the
    # fragment would silently lose that event forever.
    log.write_text('{"event": "run_start", "total": 5}\n{"event": "sta')
    reader = ProgressReader(log)
    assert reader.poll().total == 5

    with open(log, "a") as handle:
        handle.write('ge", "path": "a.pdf", "stage": "done"}\n')
    assert reader.poll().files["a.pdf"].stage == STAGE_DONE


def test_a_truncated_file_restarts_the_reconstruction(log):
    # Every build truncates the log; a watcher left running must show the new
    # run rather than a mix of the two.
    progress.emit("run_start", total=9)
    reader = ProgressReader(log)
    assert reader.poll().total == 9

    log.write_text("")
    progress.emit("run_start", total=2)
    run = reader.poll()
    assert run.total == 2
    assert run.files == {}


def test_corrupt_lines_are_skipped_not_fatal(log):
    log.write_text('not json\n{"event": "run_start", "total": 4}\n[]\n')
    assert read_progress(log).total == 4


def test_a_missing_file_reads_as_an_empty_run(tmp_path):
    reader = ProgressReader(tmp_path / "nope.jsonl")
    run = reader.poll()
    assert reader.exists is False
    assert run.files == {}


def test_polling_an_unchanged_file_is_stable(log):
    progress.emit_file(Path("a.pdf"), STAGE_DONE)
    reader = ProgressReader(log)
    first = reader.poll()
    assert reader.poll() is first
    assert len(first.files) == 1


# -- per-page progress and the resume-relevant stages -------------------------


def test_page_events_advance_the_per_document_count():
    run = reduce_events(
        [
            make("stage", path="a.pdf", stage=STAGE_SCANNED, pages=200),
            make("stage", path="a.pdf", stage=STAGE_CONVERTING),
            make("page", path="a.pdf", pages_done=10),
            make("page", path="a.pdf", pages_done=50),
        ]
    )
    assert run.files["a.pdf"].pages_done == 50
    assert run.files["a.pdf"].fraction == 0.25


def test_page_counts_only_move_forward():
    # Three processes append at once, so a late event can carry a stale count.
    run = reduce_events(
        [
            make("stage", path="a.pdf", stage=STAGE_CONVERTING),
            make("page", path="a.pdf", pages_done=90),
            make("page", path="a.pdf", pages_done=30),
        ]
    )
    assert run.files["a.pdf"].pages_done == 90


def test_a_reconversion_restarts_the_page_count():
    # Queuing is what starts a document over, not converting: a long document
    # is converted as several parts, each of which emits its own "converting",
    # and those must not wipe the progress of the parts already running.
    run = reduce_events(
        [
            make("stage", path="a.pdf", stage=STAGE_CONVERTING),
            make("page", path="a.pdf", pages_done=90),
            make("stage", path="a.pdf", stage=STAGE_FAILED, error="died"),
            make("stage", path="a.pdf", stage=STAGE_QUEUED),
            make("stage", path="a.pdf", stage=STAGE_CONVERTING),
        ]
    )
    assert run.files["a.pdf"].pages_done == 0


def test_a_second_part_does_not_reset_the_first_ones_progress():
    run = reduce_events(
        [
            make("stage", path="a.pdf", stage=STAGE_CONVERTING, part=1, ts=10.0),
            make("page", path="a.pdf", part=1, pages_done=90),
            make("stage", path="a.pdf", stage=STAGE_CONVERTING, part=251, ts=20.0),
            make("page", path="a.pdf", part=251, pages_done=40),
        ]
    )
    state = run.files["a.pdf"]
    assert state.pages_done == 130
    # The document's clock belongs to whichever part started first.
    assert state.convert_started == 10.0


def test_parts_are_summed_not_maximised():
    # Taking a maximum would report a 12-part document as barely started for as
    # long as the last part lags, and jump when it catches up.
    run = reduce_events(
        [
            make("stage", path="a.pdf", stage=STAGE_SCANNED, pages=1200),
            make("stage", path="a.pdf", stage=STAGE_CONVERTING, part=1),
            make("page", path="a.pdf", part=1, pages_done=300),
            make("page", path="a.pdf", part=301, pages_done=300),
            make("page", path="a.pdf", part=601, pages_done=200),
        ]
    )
    assert run.files["a.pdf"].pages_done == 800
    assert run.files["a.pdf"].fraction == pytest.approx(800 / 1200)


def test_an_out_of_order_report_within_a_part_is_ignored():
    run = reduce_events(
        [
            make("stage", path="a.pdf", stage=STAGE_CONVERTING, part=1),
            make("page", path="a.pdf", part=1, pages_done=90),
            make("page", path="a.pdf", part=1, pages_done=30),
        ]
    )
    assert run.files["a.pdf"].pages_done == 90


def test_fraction_is_unknown_until_both_numbers_exist():
    run = reduce_events(
        [
            make("stage", path="a.pdf", stage=STAGE_CONVERTING),
            make("page", path="a.pdf", pages_done=5),
            make("stage", path="b.pdf", stage=STAGE_SCANNED, pages=10),
        ]
    )
    assert run.files["a.pdf"].fraction is None  # no page count from the scan
    assert run.files["b.pdf"].fraction is None  # no page reported yet


def test_fraction_is_clamped_when_the_page_counts_disagree():
    # Docling counts pages it processed, pypdf counts pages in the file; a
    # malformed page makes them differ, and a bar past 100% reads as a bug.
    run = reduce_events(
        [
            make("stage", path="a.pdf", stage=STAGE_SCANNED, pages=100),
            make("stage", path="a.pdf", stage=STAGE_CONVERTING),
            make("page", path="a.pdf", pages_done=140),
        ]
    )
    assert run.files["a.pdf"].fraction == 1.0


def test_overall_pages_include_partial_progress_on_files_in_flight():
    # Otherwise the bar does not move for the length of a 2900-page manual.
    run = reduce_events(
        [
            make("stage", path="a.pdf", stage=STAGE_DONE, pages=100),
            make("stage", path="b.pdf", stage=STAGE_SCANNED, pages=1000),
            make("stage", path="b.pdf", stage=STAGE_CONVERTING),
            make("page", path="b.pdf", pages_done=400),
        ]
    )
    assert run.pages_done() == 500


def test_a_converted_file_counts_all_its_pages_before_it_is_ingested():
    run = reduce_events(
        [
            make("stage", path="a.pdf", stage=STAGE_SCANNED, pages=300),
            make("stage", path="a.pdf", stage=STAGE_CONVERTING),
            make("page", path="a.pdf", pages_done=200),
            make("stage", path="a.pdf", stage=STAGE_CONVERTED),
        ]
    )
    assert run.pages_done() == 300


def test_partial_progress_never_exceeds_the_document():
    run = reduce_events(
        [
            make("stage", path="a.pdf", stage=STAGE_SCANNED, pages=100),
            make("stage", path="a.pdf", stage=STAGE_CONVERTING),
            make("page", path="a.pdf", pages_done=150),
        ]
    )
    assert run.pages_done() == 100


def test_converted_counts_as_work_in_flight():
    run = reduce_events([make("stage", path="a.pdf", stage=STAGE_CONVERTED)])
    assert run.files["a.pdf"].is_active


def test_page_range_label():
    assert reduce_events([make("run_start")]).page_range == ""
    assert reduce_events([make("run_start", min_pages=1800)]).page_range == "1,800-"
    assert reduce_events([make("run_start", max_pages=999)]).page_range == "-999"
    assert (
        reduce_events([make("run_start", min_pages=1, max_pages=2)]).page_range == "1-2"
    )


def test_a_failed_document_is_not_revived_by_its_other_parts():
    # The parts of a split document keep running after one of them fails, and
    # they report back. A document that failed must stay failed.
    run = reduce_events(
        [
            make("stage", path="a.pdf", stage=STAGE_CONVERTING, part=1),
            make("stage", path="a.pdf", stage=STAGE_FAILED, error="part exploded"),
            make("stage", path="a.pdf", stage=STAGE_CONVERTED, part=251),
            make("stage", path="a.pdf", stage=STAGE_CONVERTED, part=501),
        ]
    )
    assert run.files["a.pdf"].stage == STAGE_FAILED
    assert run.files["a.pdf"].error == "part exploded"


def test_requeuing_clears_a_failure():
    run = reduce_events(
        [
            make("stage", path="a.pdf", stage=STAGE_FAILED, error="boom"),
            make("stage", path="a.pdf", stage=STAGE_QUEUED),
            make("stage", path="a.pdf", stage=STAGE_CONVERTING),
            make("stage", path="a.pdf", stage=STAGE_DONE, chunks=5),
        ]
    )
    assert run.files["a.pdf"].stage == STAGE_DONE
    assert run.files["a.pdf"].error is None


# -- a document is converted only when all of its parts are -------------------


def parted(n_parts, converted_parts, pages=1200):
    events = [
        make("stage", path="a.pdf", stage=STAGE_SCANNED, pages=pages),
        make("stage", path="a.pdf", stage=STAGE_QUEUED, parts=n_parts),
    ]
    starts = [1 + i * (pages // n_parts) for i in range(n_parts)]
    for start in starts:
        events.append(make("stage", path="a.pdf", stage=STAGE_CONVERTING, part=start))
    for start in starts[:converted_parts]:
        events.append(make("stage", path="a.pdf", stage=STAGE_CONVERTED, part=start))
    return reduce_events(events).files["a.pdf"]


def test_one_finished_part_does_not_convert_the_document():
    # The regression: pages_done() credits a converted document with all of its
    # pages, so a premature flag reported a 2900-page manual as finished while
    # eleven of its twelve parts were still running -- and made the throughput
    # figure derived from it about twelve times too high.
    state = parted(12, 1)
    assert state.stage == STAGE_CONVERTING


def test_the_document_converts_when_the_last_part_lands():
    assert parted(12, 11).stage == STAGE_CONVERTING
    assert parted(12, 12).stage == STAGE_CONVERTED


def test_a_single_part_document_converts_on_its_only_part():
    assert parted(1, 1).stage == STAGE_CONVERTED


def test_a_repeated_part_report_does_not_count_twice():
    run = reduce_events(
        [
            make("stage", path="a.pdf", stage=STAGE_QUEUED, parts=3),
            make("stage", path="a.pdf", stage=STAGE_CONVERTED, part=1),
            make("stage", path="a.pdf", stage=STAGE_CONVERTED, part=1),
            make("stage", path="a.pdf", stage=STAGE_CONVERTED, part=1),
        ]
    )
    assert run.files["a.pdf"].stage != STAGE_CONVERTED


def test_pages_are_not_credited_in_full_until_every_part_is_in():
    # What the broken stage actually cost: the page total, and every rate
    # computed from it.
    assert parted(12, 1, pages=2900).pages_done == 0
    assert parted(12, 12, pages=2900).stage == STAGE_CONVERTED


def test_an_unknown_part_count_does_not_block_the_document():
    # A log that starts mid-run has no "queued" event, so parts is None. Better
    # to flag the document early than to leave it converting forever.
    run = reduce_events([make("stage", path="a.pdf", stage=STAGE_CONVERTED, part=1)])
    assert run.files["a.pdf"].stage == STAGE_CONVERTED


def test_requeuing_forgets_which_parts_had_converted():
    # Without clearing the set, the retry would inherit the previous attempt's
    # converted parts and flag the document on its first part.
    attempt = [
        make("stage", path="a.pdf", stage=STAGE_QUEUED, parts=2),
        make("stage", path="a.pdf", stage=STAGE_CONVERTING, part=1),
        make("stage", path="a.pdf", stage=STAGE_CONVERTED, part=1),
        make("stage", path="a.pdf", stage=STAGE_FAILED, error="other part died"),
    ]
    retry = [
        make("stage", path="a.pdf", stage=STAGE_QUEUED, parts=2),
        make("stage", path="a.pdf", stage=STAGE_CONVERTING, part=1),
        make("stage", path="a.pdf", stage=STAGE_CONVERTED, part=1),
    ]
    state = reduce_events(attempt + retry).files["a.pdf"]
    assert state.stage == STAGE_CONVERTING
    assert state.parts_converted == {1}

    # Only the second part of the retry finishes it.
    state = reduce_events(
        attempt
        + retry
        + [
            make("stage", path="a.pdf", stage=STAGE_CONVERTING, part=601),
            make("stage", path="a.pdf", stage=STAGE_CONVERTED, part=601),
        ]
    ).files["a.pdf"]
    assert state.stage == STAGE_CONVERTED


def test_a_failed_document_still_ignores_its_remaining_parts():
    run = reduce_events(
        [
            make("stage", path="a.pdf", stage=STAGE_QUEUED, parts=3),
            make("stage", path="a.pdf", stage=STAGE_FAILED, error="boom"),
            make("stage", path="a.pdf", stage=STAGE_CONVERTED, part=1),
            make("stage", path="a.pdf", stage=STAGE_CONVERTED, part=251),
            make("stage", path="a.pdf", stage=STAGE_CONVERTED, part=501),
        ]
    )
    assert run.files["a.pdf"].stage == STAGE_FAILED
