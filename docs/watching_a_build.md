# Watching a build

A corpus build runs for hours across three process pools and a GPU that four
processes share. This is what to run, what to look at, and — mostly — which
numbers turned out to mean something and which ones lied.

Written after building 369 manuals / 172,021 pages on an L4 host
(8 vCPU, 31 GB RAM, 23 GB VRAM, no swap).

## Running it

Keep the build and the monitor in separate tmux sessions so neither dies with
a disconnected shell:

```sh
tmux new-session -d -s build 'uv run db_manager build --pdf_dir /mnt/manual-walker-input --include "zOS/V3R1/*"'
tmux new-session -d -s watch 'uv run db_manager watch'
tmux set-option -t build remain-on-exit on   # keep the pane after it ends
```

`watch` follows a build that restarts and rewrites the log, so it does not
need restarting when the build does — only when its own code changes. It reads
the file and never writes, so killing it cannot hurt a build.

Do **not** stop a build with `pkill -f "db_manager build"`: the pattern matches
the shell running the `pkill`, which then kills itself and reports failure
while the workers survive. Match on something absent from your own command
line, or kill the pids.

## Reading the panel

```
╭─ GPU slots ───────────────────────────────────────────────────────────╮
│ slot  doing        what               time  document                  │
│ 1     ▶ convert    part 2/12  242p    4:31  zOS/V3R1/asf1a411.pdf     │
│ 2     ▶ convert    part 3/12  242p    3:58  zOS/V3R1/asf1a411.pdf     │
│ 3     ◆ embedding  10,994 chunks      1:12  zOS/V3R1/bpxbd00_v3r1.pdf │
╰───────────────────────────────────────────────────────────────────────╯
```

The slot rows come from the semaphore itself, so they are the truth about who
is on the GPU. The file list below them is per *document*, which is a
different thing: a document split into twelve parts shows one `converting` row
whether one worker or three are inside it. Three busy workers looked like two
until the slot panel existed.

The two clocks say different things on purpose. A file's `time` is the whole
document from its first part; a slot's `time` is that part.

## What to watch, and what it costs to be wrong

| symptom | what it actually means |
| --- | --- |
| `Failed to ingest ... CUDA out of memory` | a document lost. Recoverable: no `converted_at`, so the next run converts it again. |
| `Stage ocr failed` | **not** recoverable. The page's OCR is dropped, the document *completes*, gets stamped, and is skipped from then on. Nothing afterwards can tell you which pages are missing. |
| host `available` under 3 GB | with no swap, the OOM killer is next, and it may pick a worker — which breaks the pool and ends the build. |
| parent RSS 6-7 GB | normal with three documents in flight. Not a leak. |

The second row is the one to stop a build for. The others can be left alone.

## Useful one-liners

Progress and the slots, without attaching:

```sh
python - <<'PY'
import time
from mcp_manual_walker.progress import read_progress
r = read_progress("data/build_progress.jsonl"); now = time.time()
print({k: v for k, v in r.counts().items() if v})
for s in r.active_slots():
    what = f"part {s.part_index}/{s.part_count}" if s.part_index else f"{s.chunks:,} chunks"
    print(f"  {s.role:8} {what:16} {now - s.since:5.0f}s  {s.path}")
PY
```

What is committed, which is the only thing a restart preserves:

```sh
sqlite3 data/mcp_manual_walker.db \
  "select count(*), sum(page_count) from manuals where converted_at is not null"
```

Per-process VRAM, which is where the ceiling is actually reached:

```sh
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader
```

What the parent is doing when it goes quiet for minutes:

```sh
sudo env "PATH=$PATH" uvx py-spy dump --pid $(pgrep -f "bin/db_manager build" | head -1)
```

That last one found the quadratic table export that a whole afternoon of
guesswork had attributed to the conversion stage.

## Numbers that lied

Three measurements were confidently wrong before being caught. Each cost hours.

**Throughput from the progress log.** `pages_done()` credited a document with
all of its pages the moment *one* part reported converted, so the same
document measured 661 pages/min at one sampling instant and 106 at another,
against a real 181 from the build log's own timestamps. Two opposite
conclusions about a batch-size change were drawn from those numbers. If a rate
matters, take it from `Converting ...` and `Generated ... chunks` timestamps in
the log, not from the progress file.

**"The GPU is exhausted, so it must be spilling to host memory."** It does not:
on Linux a `cudaMalloc` that does not fit fails, and the hard OOM errors are
themselves the proof that nothing was spilled. PCIe averaged 1 GB/s of a
15.75 GB/s Gen3 x16 link while SM sat at 79-100%; a spill looks like the
opposite.

**A benchmark on a synthetic document.** The per-table serializer measured only
1.8-3.1x slower than a shared one until the synthetic document was given
provenance and 100 pages, at which point the same comparison showed 51.9x. The
shape of the input was the whole effect.

The pattern in all three: a number that agreed with the current theory was not
checked against a second, independent measurement. Measuring the same thing
twice, by different means, is what broke each of them.
