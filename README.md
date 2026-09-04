# MCP Manual Walker 🚶‍♂️

An MCP server to bridge the information gap between AI agents and PDF manuals.

This server scans a directory for PDF files, extracts their bookmarks, and provides tools for an AI agent to access the content of the manuals in a structured, token-efficient way.

## ✨ About The Project

This project was created to solve a simple problem: AI Agents are great, but they struggle with large, unstructured data like PDF manuals. This server acts as a "librarian" for the AI, organizing the information in the manuals and making it easy for the AI to find what it needs.

The server will:

*   📚 Scan a directory for PDF files.
*   🔖 Extract bookmarks from the PDFs.
*   📝 Convert the bookmarked sections into Markdown.
*   🧠 Provide a set of tools for the AI to use.
*   ⚡ Cache the results for speedy access.

### Built With

*   [Python 3.11+](https://www.python.org/)
*   [FastMCP](https://github.com/jlowin/fastmcp)
*   [pypdf](https://pypi.org/project/pypdf/)
*   [Docling](https://github.com/DS4SD/docling)
*   [ChromaDB](https://www.trychroma.com/)
*   [Sentence Transformers](https://www.sbert.net/) (Qwen3-Embedding)
*   [SQLAlchemy](https://www.sqlalchemy.org/)
*   [Pydantic](https://pydantic-docs.helpmanual.io/)
*   [Pydantic-Settings](https://docs.pydantic.dev/latest/concepts/settings/)

## 🚀 Getting Started

To get a local copy up and running follow these simple steps.

### Prerequisites

You need to have Python 3.11+ and `uv` installed.

*   [Python 3.11+](https://www.python.org/downloads/)
*   `uv`
    ```sh
    pip install uv
    ```

### Installation

1.  Clone the repo
    ```sh
    git clone https://github.com/junkreef/mcp-manual-walker.git
    ```
2.  Install Python packages using `uv`. Choose the extra that matches this machine's role — plain `uv sync` with no extra leaves PyTorch unresolved, so always pass one:
    *   Running the search server (CPU-only):
        ```sh
        uv sync --extra cpu
        ```
    *   Building the database (GPU machine, see "Building the Database" below):
        ```sh
        uv sync --extra builder
        ```
    *   Add `--extra dev` to either for development and tests, e.g. `uv sync --extra cpu --extra dev`.
3.  Place your PDF manuals in the `data/pdfs` directory.
4.  Run the server
    ```sh
    uv run python -m mcp_manual_walker.main
    ```

## 🛠️ Usage

The server provides a set of tools for AI agents. The typical workflow is as follows:

1.  Call `list_manuals()` to get a list of all available manuals and their unique IDs.
2.  Use `get_manual_metadata(manual_id)` to retrieve the hierarchical table of contents (bookmarks) for a specific manual.
3.  Finally, call `get_markdown_content(bookmark_id)` to get the token-efficient Markdown content for the desired section.
4.  When a section or a search hit points at a figure, call `get_figure(figure_id)` to look at the picture itself.

This process allows an agent to intelligently navigate large documents and only pull the necessary information into its context.

### Tools

| Tool | Input | Output |
| --- | --- | --- |
| `list_manuals` | — | Every manual with its `id`, `file_name` and `document_title`. |
| `get_manual_metadata` | `manual_id` | The manual's metadata and its hierarchical `table_of_contents`; each bookmark carries the `id` the content tools need. |
| `get_markdown_content` | `bookmark_id` | The Markdown of that section and its subsections, plus the `figures` it contains. |
| `search_manual` | `manual_id`, `query`, optional `bookmark_id` | The top matching chunks with their bookmark path, `chunk_type` and, for figures, a `figure` reference. |
| `get_figure` | `figure_id` | The figure's PNG image plus a JSON block of its metadata. |

### Figures in the tool responses

Figures are first-class results, so an agent can find a diagram by what it shows and then actually look at it:

*   `search_manual` reports a `chunk_type` for every hit — `"text"`, `"table"` or `"figure"`. A figure hit matched the figure's caption, its on-page labels and its description (see "Figure descriptions" below), which are also returned as the hit's `context`, and it carries a `figure` object with the figure's `id`, `page`, `caption`, `description` and `bookmark_id`. Text and table hits have `figure: null`.
*   `get_markdown_content` inserts each figure into the Markdown as a marker followed by the same text, so the picture keeps its place in the section:

    ```
    [Figure: 3f6c1c1e-... (page 12)]

    Figure 4: Wiring of the control unit

    Labels: Power unit, Controller, Sensor

    A diagram showing the controller wired to the sensor.
    ```

    The same figures are listed in document order in the response's `figures` field, so the ids can be used without parsing the Markdown. Sections without figures return an empty list.
*   `get_figure(figure_id)` returns two content blocks: the PNG image itself (an image content block, `image/png`) and a JSON text block with the figure's `id`, `manual_id`, `bookmark_id`, `page`, `caption`, `labels`, `description`, `width`, `height` and `mime_type`. An unknown id is an error.

The typical figure workflow is therefore: `search_manual(...)` → take `figure.id` from a hit whose `chunk_type` is `"figure"` (or an id from the `figures` list of `get_markdown_content`) → `get_figure(figure_id=...)`.

## 🏗️ Building the Database

Building the searchable database (Docling PDF conversion, chunking, embeddings, and ChromaDB indexing) is a separate step from running the server, and it has its own dependencies. Install them first:

```sh
uv sync --extra builder
```

The `builder` extra pulls in Docling, `sentence-transformers`, and PyTorch's CUDA 13.0 build (via the nested `cu130` extra, resolved from the `pytorch` index configured in `pyproject.toml`), so embeddings and Docling's models can run on the GPU. `builder` and the server's `cpu` extra are mutually exclusive — install whichever matches the machine (see Installation above), and add `--extra dev` on top for development and tests.

Then run the builder:

```sh
uv run db_manager build --pdf_dir ./data/pdfs \
    [--reset] [--save-markdown] [--include GLOB] [--min-pages N] [--max-pages N]
```

`--include` converts only the PDFs whose path *relative to `--pdf_dir`* matches
the glob, which is how a large corpus gets built a product or a release at a
time:

```sh
uv run db_manager build --pdf_dir ./data/pdfs --include 'zOS/V3R1/*'
```

The flag is repeatable and a file matching any pattern is kept. These are
`fnmatch` patterns, so `*` also matches `/` — `zOS/V3R1/*` takes that directory
and everything nested below it.

Leave `--pdf_dir` pointing at the corpus root and narrow with `--include`,
rather than pointing `--pdf_dir` at the subdirectory. `--pdf_dir` is also the
anchor every stored `relative_path` is computed from, so moving it renames the
manuals: the same file becomes `bpxbd00_v3r1.pdf` instead of
`zOS/V3R1/bpxbd00_v3r1.pdf`, which collides with the identically named file in
another release and no longer resolves against `PDF_ROOT_DIR` when the server
opens the PDF. Narrowing with `--include` keeps the anchor put, so subsets
built one at a time add up to exactly the database a single whole-corpus build
would have produced.

The build pipeline is designed to keep the GPU busy instead of processing one PDF at a time. The main process scans the PDF directory (largest files first), runs a fast metadata pass (hashing + bookmark extraction), and syncs the results to SQLite. It then submits each new or changed PDF to a pool of Docling worker processes for conversion; as each conversion finishes, the main process chunks the text, computes embeddings on the GPU, and writes the result into ChromaDB, so embedding of one file overlaps with Docling converting the next ones. Files whose SHA256 hash hasn't changed are skipped entirely, and rebuilding a manual first removes its old chunks from ChromaDB before adding the new ones.

### 🎛️ Rationing the GPU

The Docling workers and the builder's own embedding model share one device,
and neither yields to the other. On a 23 GB L4 with three workers that is
about 17 GB of converting plus 5 GB of embedding, and the collision showed up
three separate ways in one afternoon:

- `torch.OutOfMemoryError` in the parent, unable to allocate 192 MB to embed a
  finished document — the document was lost and re-converted from scratch;
- RapidOCR's ONNX arena unable to allocate 142 MB, which does **not** fail the
  document: the page's OCR is simply dropped, the conversion is recorded as a
  success, and nothing afterwards can tell you which pages are missing;
- earlier, before `EMBEDDING_TOKEN_BUDGET`, the embedder alone asking for
  16 GB.

`DOCLING_GPU_SLOTS` puts a semaphore in front of the device, shared by the
worker processes and the parent. A slot is held for one conversion or one
embedding call, so a finished document's embedding simply costs a converting
worker until it is done, and the peak becomes a number you set rather than
whatever happens to overlap.

The slot is deliberately coarse — a whole conversion, not an allocation — and
the parent can therefore wait minutes for one. That is the trade: the
alternative is a lock around every allocation inside Docling and torch, which
neither library offers.

Sizing it against `DOCLING_WORKERS`: leaving them equal means the embedder
displaces a worker whenever it runs, which is the intended behaviour. Setting
slots *below* the worker count reserves headroom permanently; setting it
above lets everything race again.

### 👀 Watching a build

A build spreads its work over three process pools, so nothing on the console
tells you where any individual PDF is. Every process therefore appends its
state changes to a JSONL log (`BUILD_PROGRESS_FILE`, truncated at the start of
each build), and a second terminal can render it:

```sh
uv run db_manager watch
```

A panel above the list shows the GPU slots — one row each, whether it is a
worker converting (which document, which part of how many) or the parent
embedding (which document, how many chunks), and how long it has held the
slot. Empty slots are drawn too, so the shape of `DOCLING_GPU_SLOTS` is
visible rather than implied. The rows come from the slot acquisition itself,
so the panel is the semaphore's state rather than an inference from which
documents happen to be moving.

The display lists every file in the run with the stage it is in — `scanning`,
`queued`, `converting`, `converted`, `ingesting`, `done`, `skipped` or
`failed` — the page count, how long it has been there, and the chunks and
figures it produced. A document being converted carries its own progress bar,
fed by pages leaving the last pipeline stage, so a 2900-page manual visibly
moves rather than sitting on `converting` for forty minutes. (`converted`
means the conversion finished and the parent has not collected the result yet;
it collects one at a time while embedding the previous one.)
Above the list are the per-stage totals, progress measured in pages rather
than files (files differ by two orders of magnitude in length), the observed
pages/min and an ETA. A failed file carries its error on its own row.

| Key | |
| --- | --- |
| `j` / `k`, `↑` / `↓` | scroll one row |
| `PgUp` / `PgDn` | scroll one screen |
| `g` / `G` | jump to the top or the bottom |
| `f` | follow the files currently being converted (on by default) |
| `a` | cycle the view: all files → in flight only → everything unfinished |
| `q` | quit (the build keeps running) |

The monitor only ever reads the log, so it can be started, stopped and
restarted while the build runs, and reading it after the build has finished
replays the whole run. `--once` prints a single snapshot instead of taking
over the terminal — useful from a script, since it exits non-zero if any file
failed — and `--no-progress` on `build` turns the log off entirely.

Running the build under `tmux` keeps the two apart:

```sh
tmux new-session -d -s build 'uv run db_manager build --pdf_dir ./data/pdfs --include "zOS/V3R1/*"'
uv run db_manager watch
```

The pipeline's concurrency and device placement are tuned through environment variables (see `.env.example`):

| Variable | Default | What it does |
| --- | --- | --- |
| `METADATA_WORKERS` | `max(1, cpu_count // 2)` | Processes used for the fast hashing/bookmark pass. 1 or less runs it inline. |
| `DOCLING_WORKERS` | `1` | Number of Docling converter processes, each with its own copy of the models in VRAM. The main knob for GPU utilization, and the one that will run you out of memory — see [Sizing the workers](#sizing-the-workers) below. |
| `DOCLING_NUM_THREADS` | CPU count | Total CPU-thread budget for Docling, split evenly across `DOCLING_WORKERS`. |
| `DOCLING_GPU_SLOTS` | `3` | How many things may use the GPU at once. Docling workers and the builder's own embedding model share one device and neither yields, so this rations it: embedding a finished document costs one converting worker until it is done. `0` removes the limit. |
| `DOCLING_SPLIT_PAGES` | `250` | Pages per conversion unit. A longer document is converted as several page ranges in parallel and merged, which makes a worker's peak memory a function of this number instead of the longest document in the corpus. `0` converts every document whole. |
| `DOCLING_QUEUE_MAX_SIZE` | `16` | Pages allowed to queue in front of each pipeline stage. A queued page holds its rendered image, so this — not the page count — sets peak memory per worker (600-page manual: `100` → 8.2 GB, `16` → 5.0 GB, same output, same wall time). |
| `DOCLING_DEVICE` | `auto` | Accelerator for Docling's layout/table/OCR models (`auto`, `cpu`, `cuda`, `cuda:N`, `mps`). |
| `DOCLING_OCR_BACKEND` | `onnxruntime` | RapidOCR inference backend: `onnxruntime` (default; models for `japan`/`chinese`/`en` ship with the rapidocr wheel, works offline, GPU via `onnxruntime-gpu`) or `torch` (downloads checkpoints from modelscope.cn on first use). |
| `DOCLING_OCR_LANG` | `japan` | RapidOCR language token for the recognition model. `japan`/`chinese`/`en` need no download; other languages (e.g. `korean`) are fetched on first use. |
| `DOCLING_IMAGES_SCALE` | `2.0` | Render scale for the figure crops stored in SQLite (`1.0` = 72 dpi, `2.0` = 144 dpi). Higher means sharper PNGs and a bigger database file. |
| `EMBEDDING_DEVICE` | `auto` | Device for the SentenceTransformers embedding model (`auto`, `cpu`, `cuda`, `cuda:N`). |
| `EMBEDDING_DTYPE` | `auto` | Dtype the weights load under. `auto` reads it from the checkpoint (bfloat16); set `float32` on a CPU-only server (see below). |

OCR only runs on layout regions without a PDF text layer (scanned pages, text
inside images) — text-based PDFs are read from their text layer, so the OCR
backend/language choice doesn't affect them.

### Sizing the workers

Both host RAM and VRAM bind, and **what sets the peak is the length and the
table density of the single document a worker happens to be holding**, not the
size of the corpus. A worker accumulates per-page structures for the whole
document it is converting and only releases them when it finishes, so a long
manual is a long, rising ramp.

Measured on an L4 host (31 GB RAM, 23 GB VRAM) at `DOCLING_QUEUE_MAX_SIZE=16`:

| Document | Peak RSS | Per page |
| --- | --- | --- |
| 2252 pages, moderate tables | 8.3 GB | 1.9 MB |
| 2900 pages, table-dense | 16.7 GB *(still climbing)* | ≥ 4.9 MB |

So per-page retention varies by **more than 2.5x with content**, and a formula
fitted to one document will not hold for another. Budget with the pessimistic
figure:

    per worker ≈ 4 GB  (models, CUDA context, allocator floor, pages in flight)
               + 4.5 MB x pages of the longest document that worker may hold

VRAM is the second ceiling and it is easy to miss, because the *builder's own
embedding model shares the GPU with the workers*. Three workers were measured
holding 7.2 + 6.9 + 5.3 = 19.4 GB of 22 GB, at which point the parent process
could not allocate 542 MB to embed a finished document and the ingest failed
with `torch.OutOfMemoryError` — the conversion had gone fine.

`DOCLING_SPLIT_PAGES` is what stops that variability from deciding the worker
count. A document longer than it is converted as several page ranges, spread
across the pool and merged in the parent, so a worker's peak becomes

    per worker ≈ 4 GB + 4.5 MB x DOCLING_SPLIT_PAGES

— a number you choose, the same for every document in the corpus. It also
stops one 2900-page manual from occupying a single worker for forty minutes
while the rest of the GPU idles. Measured on a 1246-page manual, 250-page
parts across 3 workers: 403 s and 5.93 GB whole against 286 s and a 3.94 GB
worst worker, with byte-identical output.

Set it to 0 to convert every document whole.

`--min-pages` / `--max-pages` still select part of a corpus by document length,
which is useful for retrying or for building the long documents separately, but
they are no longer how you keep a build inside its memory budget.

### Resuming an interrupted build

A manual's row in SQLite is written by the fast metadata pass, before anything
is converted. A build therefore cannot decide what to skip from the file hash
alone: the hash of a document an interrupted build never reached matches
perfectly. Each manual instead carries a `converted_at` stamp, set only once
its chunks are in ChromaDB, and a re-run skips a file only when the hash
matches **and** that stamp is present.

So a build that dies at file 200 of 369 — out of memory, a killed worker, a
disconnected session — is resumed by re-running the same command without
`--reset`. The 199 finished manuals are skipped and the rest are converted.
Re-running is also how you retry the failures from a previous pass.

### 🖼️ Figures

Every picture Docling detects is rendered (at `DOCLING_IMAGES_SCALE`) and stored
as a PNG blob in the SQLite `figures` table, together with its page, bounding
box, caption, the labels drawn inside it and the manual and bookmark it belongs
to. The chunk that describes the figure carries only the `figure_id` in its
ChromaDB metadata, so the SQLite file and the Chroma directory are the only two
artifacts that have to travel between machines — the image bytes are never
duplicated into the vector store.

Consequences worth knowing:

*   `db_manager export` writes each figure as a separate `figures/<id>.png`
    member of the archive (`format_version: 2` in `manifest.json`) and
    `db_manager import` restores the rows with their bytes; older archives
    without figures still import unchanged. Deleting a manual deletes its
    figures with it.
*   `--save-markdown` additionally writes the PNGs next to the markdown dump,
    in a `<name>_artifacts/` directory referenced by the markdown image links.
    That copy is a viewing convenience; the database stays the source of truth.
*   The `figures` table is created by `init_db()`, which only ever adds missing
    tables. **A database built before figures existed has no images in it and
    must be rebuilt:**

    ```sh
    uv run db_manager build --pdf_dir ./data/pdfs --reset
    ```

### 🗣️ Figure descriptions (optional)

Each figure crop can additionally be sent to a local vision model over an
OpenAI-compatible `chat/completions` API (Docling's built-in
`PictureDescriptionApiOptions`). The returned text is stored in
`figures.description` and folded into the figure chunk's text, so figures
become searchable by what they actually show instead of just their caption
and on-page labels.

The feature is off by default (`PICTURE_DESCRIPTION_URL` empty). To enable
it, run any vision-capable model (e.g. Gemma 3 4B) behind an
OpenAI-compatible server — [Ollama](https://ollama.com) or
[llama.cpp's server](https://github.com/ggml-org/llama.cpp) both work — and
set `PICTURE_DESCRIPTION_URL` (and `PICTURE_DESCRIPTION_MODEL` for Ollama,
which requires the model name in the request body; llama.cpp ignores it
since the model is fixed at server startup):

```sh
PICTURE_DESCRIPTION_URL=http://localhost:11434/v1/chat/completions
PICTURE_DESCRIPTION_MODEL=gemma3:4b
```

| Variable | Default | What it does |
| --- | --- | --- |
| `PICTURE_DESCRIPTION_URL` | `""` | OpenAI-compatible `chat/completions` endpoint. Empty disables the feature. |
| `PICTURE_DESCRIPTION_MODEL` | `""` | Sent as `"model"` in the request payload. Required by Ollama, ignored by llama.cpp's server. |
| `PICTURE_DESCRIPTION_API_KEY` | `""` | Sent as `Authorization: Bearer ...` when set. |
| `PICTURE_DESCRIPTION_PROMPT` | (Japanese prompt, see `config.py`) | Prompt sent alongside each figure crop. |
| `PICTURE_DESCRIPTION_MAX_TOKENS` | `300` | Upper bound on the generated description length. |
| `PICTURE_DESCRIPTION_TIMEOUT` | `120.0` | HTTP timeout per request to the vision API, in seconds. |
| `PICTURE_DESCRIPTION_CONCURRENCY` | `1` | Parallel requests in flight per Docling worker. With `DOCLING_WORKERS > 1`, every worker sends its own requests, so the effective load on the vision server is `PICTURE_DESCRIPTION_CONCURRENCY * DOCLING_WORKERS`. |
| `PICTURE_DESCRIPTION_AREA_THRESHOLD` | `0.02` | Pictures smaller than this fraction of the page area are skipped (not sent to the vision API). |

A failing or unreachable endpoint does not fail the build: Docling logs an
error per request and leaves that figure's description empty, and the
builder additionally logs a warning naming the manual and how many figures
got no description, as a hint to check whether the vision server is up.

### 🧠 Batching the embeddings

`SentenceTransformer.encode` pads a batch to its longest member, so a batch of
32 costs `32 × longest`, not the sum of its lengths. On this corpus that is not
a rounding error: of 562 chunks from one manual, 561 averaged 298 tokens and
**one** — the OCR'd labels of a dense form, 6943 characters at 1.8 characters
per token — came to 3836. Every batch that chunk landed in was padded out to
it, and the builder died with `torch.OutOfMemoryError` while a single Docling
worker held 5 GB.

| batching | peak VRAM | wall clock |
| --- | --- | --- |
| rows, 32 per batch | 18.55 GB | 27.1 s |
| token budget, 24576 | **4.39 GB** | **11.9 s** |

Four times less memory *and* twice as fast, because the padding was pure
waste. `EMBEDDING_TOKEN_BUDGET` caps `len(batch) × longest`, so a batch of long
texts is small and a batch of short ones is large. A single text over the
budget still gets its own batch; truncation at `EMBEDDING_MAX_SEQ_LENGTH`
remains the model's business.

Vectors are not bit-identical between batchings, but neither are two runs of
the same batching: the largest cosine difference measured across those 562
chunks was 7.4e-3 between two identical row-batched runs and 7.0e-3 between row
and budget batching — the model's own non-determinism on the GPU, not the
batching.

Figure chunks longer than `CHUNK_SIZE` are split, for the same reason and for
one more: at 3836 tokens against an `EMBEDDING_MAX_SEQ_LENGTH` of 4096, a
slightly larger figure would have been truncated with no warning at all. Every
piece keeps the same `picture_index`, so they all resolve to a single figure
row.

### 🧠 Embedding Model

Both the builder and the server embed text with [Sentence Transformers](https://www.sbert.net/) using [`Qwen/Qwen3-Embedding-0.6B`](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) (Apache-2.0). It was chosen for:

*   **Long context (up to 32k tokens)** — comfortably covers the 2000-character chunks this project produces, so nothing gets silently truncated.
*   **Strong multilingual quality**, including Japanese, which matters for non-English manuals.
*   **A permissive, self-hostable license** (Apache-2.0).
*   **A size (0.6B params, 1024-dim output) small enough to run on CPU** for query embedding at search time — the server only embeds one short query per call, so it never needs a GPU, even though the builder does for embedding whole manuals at scale.

    On a CPU-only server, set `EMBEDDING_DTYPE=float32`. The checkpoint is bfloat16, and `auto` honours that, but bfloat16 has no fast CPU kernels: one query measured **0.86 s** under bfloat16 against **0.43 s** under float32 on an L4 host's CPU. float32 doubles the resident weights (1.11 GiB → 2.22 GiB) and halves the latency. Leave it on `auto` for the GPU builder, where bfloat16 is both smaller and fast.

The chosen model is recorded in the ChromaDB collection's metadata when the database is built. The server checks this against its own configured `EMBEDDING_MODEL` at startup and refuses to serve searches if they don't match, with an error telling you to rebuild. **Any existing database (e.g. one built with the earlier `intfloat/multilingual-e5-small` model) must be rebuilt after an embedding model change:**

```sh
uv run db_manager build --pdf_dir ./data/pdfs --reset
```

| Variable | Default | What it does |
| --- | --- | --- |
| `EMBEDDING_MODEL` | `Qwen/Qwen3-Embedding-0.6B` | Sentence Transformers model id. Must be identical for the builder and the server. |
| `EMBEDDING_DEVICE` | `auto` | Device for the embedding model (`auto`, `cpu`, `cuda`, `cuda:N`). |
| `EMBEDDING_DTYPE` | `auto` | Dtype the weights load under (`auto`, `float32`, `bfloat16`, `float16`). |
| `EMBEDDING_QUERY_PREFIX` | *(model default)* | Text prepended to search queries before embedding. Unset (`None`) uses the prompt the model ships under the name `query` (for Qwen3-Embedding: the "Instruct: ... \nQuery:" instruction). Set it explicitly only when switching to a model without stored prompts (e5-style models use `query: ` / `passage: `). |
| `EMBEDDING_DOCUMENT_PREFIX` | *(model default)* | Text prepended to every chunk at build time. Unset (`None`) uses the prompt the model ships under the name `document` (for Qwen3-Embedding: nothing). |
| `EMBEDDING_MAX_SEQ_LENGTH` | `4096` | Token cap per text passed to the model. Raise for very long chunks, at the cost of VRAM/RAM. |
| `EMBEDDING_BATCH_SIZE` | `32` | Upper bound on the rows in one encode batch. |
| `EMBEDDING_TOKEN_BUDGET` | `24576` | Upper bound on the padded tokens in one encode batch (`len(batch) × the longest text in it`). A batch is padded to its longest member, so budgeting rows alone prices every batch at its worst one. `0` falls back to plain row batching — see [Batching the embeddings](#-batching-the-embeddings). |

## 🗺️ Roadmap

*   [ ] Implement a more sophisticated search functionality.
*   [ ] Add a web interface for managing manuals.

See the [open issues](https://github.com/junkreef/mcp-manual-walker/issues) for a full list of proposed features (and known issues).

## 🤝 Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes. This project follows the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) specification.
    ```sh
    git commit -m "feat: Add some AmazingFeature"
    ```
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

## 📧 Contact

Junpei Kishi - junkreef@longarch.net

Project Link: [https://github.com/junkreef/mcp-manual-walker](https://github.com/junkreef/mcp-manual-walker)
