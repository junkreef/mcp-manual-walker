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

This process allows an agent to intelligently navigate large documents and only pull the necessary information into its context.

## 🏗️ Building the Database

Building the searchable database (Docling PDF conversion, chunking, embeddings, and ChromaDB indexing) is a separate step from running the server, and it has its own dependencies. Install them first:

```sh
uv sync --extra builder
```

The `builder` extra pulls in Docling, `sentence-transformers`, and PyTorch's CUDA 13.0 build (via the nested `cu130` extra, resolved from the `pytorch` index configured in `pyproject.toml`), so embeddings and Docling's models can run on the GPU. `builder` and the server's `cpu` extra are mutually exclusive — install whichever matches the machine (see Installation above), and add `--extra dev` on top for development and tests.

Then run the builder:

```sh
uv run db_manager build --pdf_dir ./data/pdfs [--reset] [--save-markdown]
```

The build pipeline is designed to keep the GPU busy instead of processing one PDF at a time. The main process scans the PDF directory (largest files first), runs a fast metadata pass (hashing + bookmark extraction), and syncs the results to SQLite. It then submits each new or changed PDF to a pool of Docling worker processes for conversion; as each conversion finishes, the main process chunks the text, computes embeddings on the GPU, and writes the result into ChromaDB, so embedding of one file overlaps with Docling converting the next ones. Files whose SHA256 hash hasn't changed are skipped entirely, and rebuilding a manual first removes its old chunks from ChromaDB before adding the new ones.

The pipeline's concurrency and device placement are tuned through environment variables (see `.env.example`):

| Variable | Default | What it does |
| --- | --- | --- |
| `METADATA_WORKERS` | `max(1, cpu_count // 2)` | Processes used for the fast hashing/bookmark pass. 1 or less runs it inline. |
| `DOCLING_WORKERS` | `1` | Number of Docling converter processes, each with its own copy of the models in VRAM. The main knob for GPU utilization. |
| `DOCLING_NUM_THREADS` | CPU count | Total CPU-thread budget for Docling, split evenly across `DOCLING_WORKERS`. |
| `DOCLING_DEVICE` | `auto` | Accelerator for Docling's layout/table/OCR models (`auto`, `cpu`, `cuda`, `cuda:N`, `mps`). |
| `DOCLING_OCR_BACKEND` | `onnxruntime` | RapidOCR inference backend: `onnxruntime` (default; models for `japan`/`chinese`/`en` ship with the rapidocr wheel, works offline, GPU via `onnxruntime-gpu`) or `torch` (downloads checkpoints from modelscope.cn on first use). |
| `DOCLING_OCR_LANG` | `japan` | RapidOCR language token for the recognition model. `japan`/`chinese`/`en` need no download; other languages (e.g. `korean`) are fetched on first use. |
| `DOCLING_IMAGES_SCALE` | `2.0` | Render scale for the figure crops stored in SQLite (`1.0` = 72 dpi, `2.0` = 144 dpi). Higher means sharper PNGs and a bigger database file. |
| `EMBEDDING_DEVICE` | `auto` | Device for the SentenceTransformers embedding model (`auto`, `cpu`, `cuda`, `cuda:N`). |

OCR only runs on layout regions without a PDF text layer (scanned pages, text
inside images) — text-based PDFs are read from their text layer, so the OCR
backend/language choice doesn't affect them.

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

### 🧠 Embedding Model

Both the builder and the server embed text with [Sentence Transformers](https://www.sbert.net/) using [`Qwen/Qwen3-Embedding-0.6B`](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) (Apache-2.0). It was chosen for:

*   **Long context (up to 32k tokens)** — comfortably covers the 2000-character chunks this project produces, so nothing gets silently truncated.
*   **Strong multilingual quality**, including Japanese, which matters for non-English manuals.
*   **A permissive, self-hostable license** (Apache-2.0).
*   **A size (0.6B params, 1024-dim output) small enough to run on CPU** for query embedding at search time — the server only embeds one short query per call, so it never needs a GPU, even though the builder does for embedding whole manuals at scale.

The chosen model is recorded in the ChromaDB collection's metadata when the database is built. The server checks this against its own configured `EMBEDDING_MODEL` at startup and refuses to serve searches if they don't match, with an error telling you to rebuild. **Any existing database (e.g. one built with the earlier `intfloat/multilingual-e5-small` model) must be rebuilt after an embedding model change:**

```sh
uv run db_manager build --pdf_dir ./data/pdfs --reset
```

| Variable | Default | What it does |
| --- | --- | --- |
| `EMBEDDING_MODEL` | `Qwen/Qwen3-Embedding-0.6B` | Sentence Transformers model id. Must be identical for the builder and the server. |
| `EMBEDDING_DEVICE` | `auto` | Device for the embedding model (`auto`, `cpu`, `cuda`, `cuda:N`). |
| `EMBEDDING_QUERY_PREFIX` | *(model default)* | Text prepended to search queries before embedding. Unset (`None`) uses the prompt the model ships under the name `query` (for Qwen3-Embedding: the "Instruct: ... \nQuery:" instruction). Set it explicitly only when switching to a model without stored prompts (e5-style models use `query: ` / `passage: `). |
| `EMBEDDING_DOCUMENT_PREFIX` | *(model default)* | Text prepended to every chunk at build time. Unset (`None`) uses the prompt the model ships under the name `document` (for Qwen3-Embedding: nothing). |
| `EMBEDDING_MAX_SEQ_LENGTH` | `4096` | Token cap per text passed to the model. Raise for very long chunks, at the cost of VRAM/RAM. |
| `EMBEDDING_BATCH_SIZE` | `32` | Encode batch size used by the builder. |

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
