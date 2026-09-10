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

    Or run it in a container — see [Running in containers](#-running-in-containers).

## 🐳 Running in containers

`compose.yaml` brings up the search server and the Qdrant it talks to. It is
not in the repository: it is your copy of `compose.example.yaml`, because
ports, bind mounts and the user the containers run as are properties of one
host. Copy it, edit it, and diff it against the original when you want to see
what you changed. The builder is deliberately not in it: it wants a GPU,
Docling's model downloads and several gigabytes of VRAM, and it runs once per
corpus rather than continuously, so it belongs on the machine with the card.
Build there and move the result with `db_manager export` / `import`, which is
backend-neutral.

```sh
mkdir -p data/qdrant                                  # before the first `up`
cp compose.example.yaml compose.yaml                  # yours to edit
cp .env.example .env                                  # set EMBEDDING_API_BASE
printf 'MMW_UID=%s\nMMW_GID=%s\n' "$(id -u)" "$(id -g)" >> .env
docker compose up -d --build
docker compose --profile tools run --rm db_manager import --input /app/data/corpus.zip
docker compose --profile gui up -d --build            # optional: the console
```

The server answers on two ports: MCP on `http://127.0.0.1:8000/mcp`, and the
[REST search API](#-the-rest-api) on `http://127.0.0.1:8001`, which is what a
caller that does not speak MCP uses. The optional
[browser console](#-the-browser-console) is on `http://127.0.0.1:8080`.

The first line matters: Docker creates a missing bind-mount source itself, and
it creates it owned by root. The fourth is only needed if your account is not
`1000:1000`, which is the default — see [Where the data lives](#where-the-data-lives).

**The server does not need to be restarted after the corpus arrives.** Starting
it next to an empty Qdrant is the normal ordering, so it re-attaches on the
first request once the collection exists.

### Two images, because the server has two shapes

| Target | Size | Needs |
| --- | --- | --- |
| `server-remote` (default) | **905 MB** | an OpenAI-compatible endpoint |
| `server` (`--profile local`) | 2.43 GB | nothing; downloads 1.2 GiB of weights on first use |

The size difference is the whole of it: `sentence-transformers` and torch are
not in the remote image at all. They moved to the `local-embeddings` extra,
which `cpu` and `cu130` pull in, so nothing changes for a `uv sync --extra cpu`
install.

**Which is faster is a question about your machines, not about these images.**
The two targets do not do the work differently; they do it in different places.
`server-remote` sends the query somewhere else, so its latency is that host
plus the network. `server` does it here, so its latency is this host's CPU. On
one setup — a Lemonade Server on the LAN with a GPU against CPU float32 in a
container on this host — `search_manual` measured a p50 of 131 ms remote
against 295 ms local over 15 searches. Move either machine and the comparison
moves with it. Do not read that as a property of the images.

What does not depend on the environment is the shape of the decision:

*   Take `server-remote` if you already run an inference endpoint, or if the
    server host is small — it is a third of the image and holds no model.
*   Take `server` if you would rather not depend on another service being up,
    or if there is no endpoint to point at.

The one comparison that does generalise is between a *query* and a *chunk*,
because it follows the token count rather than the hardware: a search query is
a few dozen tokens and a chunk is a few hundred, so embedding a whole corpus
through an endpoint costs roughly an order of magnitude more per item than
embedding a query does. That is why the builder is not offered either of these
images and keeps its GPU.

```sh
docker compose up -d --build                  # server-remote
docker compose --profile local up -d --build  # server
```

Run one or the other, never both: they publish the same port.

### Where the data lives

**`./data`, the same place a non-container install puts it.** The whole
directory is bind-mounted at `/app/data`, so a container database and a local
one are the same database, and a backup is a copy of a directory. Set
`MMW_DATA_DIR` to an absolute path when the checkout is not the right home for
it — a systemd install keeps its data in `/var/lib`. Inside the container the
path is `/app/data` either way, so nothing else in the configuration changes.

| Path | What |
| --- | --- |
| `data/mcp_manual_walker.db` | Manuals, bookmarks, figures, the BM25 index, the recorded embedding model. Half the corpus: a chunk id in Qdrant means nothing without it. |
| `data/qdrant/` | The vectors. |
| `data/model-cache/` | The embedding model, `--profile local` only. Re-downloadable, so leave it out of backups. |
| `data/pdfs/`, `data/*.zip` | Sources: what the builder reads and what `import` reads. |

Because those files are yours rather than the container's, **the containers run
as you**, not as root or as the image's own user. The default is `1000:1000`,
which is the first login account on most Linux systems. If that is not you:

```sh
printf 'MMW_UID=%s\nMMW_GID=%s\n' "$(id -u)" "$(id -g)" >> .env
```

`UID` cannot be used directly, incidentally: bash marks it readonly and never
exports it, so compose would not see it.

Qdrant's own image expects to be root — it writes an init marker and a
snapshots directory next to its binary rather than into its storage — so the
compose file redirects both into the volume with `QDRANT_INIT_FILE_PATH` and
`QDRANT__STORAGE__SNAPSHOTS_PATH`. Without that it panics on startup as any
other user.

Qdrant's ports are published on loopback only, because it has no
authentication unless `QDRANT__SERVICE__API_KEY` is set. Publish it wider only
once you have set one.

### Running it as a systemd service

`docker compose up -d` already survives a reboot on its own — `restart:
unless-stopped` sees to that — so a unit file buys something narrower:
`systemctl stop` as the one way to take the stack down, ordering against other
units, and the stack in `systemctl status` next to everything else on the
machine. `deploy/mcp-manual-walker.service.example` is that unit.

An installed copy differs from a checkout in two ways, and both are about who
owns what. The code lives under `/opt`, where root owns it and nothing writes
to it. The data does not: it belongs to a service account rather than to a
login, so it goes to `/var/lib` and `MMW_DATA_DIR` points at it.

```sh
# A user with no login and no home, whose only job is to own the data.
sudo useradd --system --no-create-home --shell /usr/sbin/nologin mmw

# The code. Everything the image needs is tracked, so a clone is enough --
# and it leaves `git pull` as the way to update.
sudo git clone https://github.com/junkreef/mcp-manual-walker.git \
    /opt/mcp-manual-walker
cd /opt/mcp-manual-walker

# The data, created before the first `up`: Docker makes a missing bind-mount
# source itself, and it makes it owned by root. Name both directories --
# `install -d` only sets ownership on what it creates, so a parent that
# already exists keeps whichever owner it was made with, and a root-owned
# parent stops the server creating data/pdfs however right the child looks.
sudo install -d -o mmw -g mmw /var/lib/mcp-manual-walker
sudo install -d -o mmw -g mmw /var/lib/mcp-manual-walker/qdrant

# The two files that describe this host rather than the project.
sudo cp compose.example.yaml compose.yaml
sudo cp .env.example .env
printf 'MMW_DATA_DIR=/var/lib/mcp-manual-walker\nMMW_UID=%s\nMMW_GID=%s\n' \
    "$(id -u mmw)" "$(id -g mmw)" | sudo tee -a .env
sudoedit .env    # set EMBEDDING_API_BASE

# Build now rather than during the first boot: the unit has no timeout, but a
# machine that takes ten minutes to come up is its own problem.
sudo docker compose build

sudo install -m 644 deploy/mcp-manual-walker.service.example \
    /etc/systemd/system/mcp-manual-walker.service
sudo systemctl daemon-reload
sudo systemctl enable --now mcp-manual-walker
```

`--profile local` is a line in the unit rather than a flag you remember:
change `ExecStart` to `docker compose --profile local up -d --wait` and give
`ExecStop` and `ExecReload` the same profile, or `down` will leave the
container it does not know about running.

The unit is `Type=oneshot` with `RemainAfterExit=yes`, because `up -d` exits
immediately and leaves systemd no process to follow; without it the unit would
go inactive the moment it succeeded, and `systemctl stop` would never run
`docker compose down`. `--wait` makes a successful start mean the stack
answers rather than that it was launched.

Day to day:

```sh
sudo systemctl status mcp-manual-walker
sudo systemctl reload mcp-manual-walker   # after editing compose.yaml or .env
sudo journalctl -u mcp-manual-walker      # the unit's own output; container
sudo docker compose -f /opt/mcp-manual-walker/compose.yaml logs -f mcp
```

`db_manager` is unchanged, but it has to run from the install directory and as
root, because that is where `compose.yaml` and `.env` are:

```sh
cd /opt/mcp-manual-walker
sudo docker compose --profile tools run --rm db_manager \
    import --input /app/data/corpus.zip
```

Note the argument: the path is the container's, so a file you want imported
goes into `/var/lib/mcp-manual-walker` on the host and is named `/app/data/...`
on the command line.

#### When the unit fails to start

`--wait` reports `container mcp-manual-walker-mcp-1 is unhealthy` for anything
that stops the server answering, which includes a container that exited a
second after it started. The unit's own journal will not say why — that is in
the container's log, and `restart: unless-stopped` may already have replaced
the container that holds it:

```sh
sudo docker compose -f /opt/mcp-manual-walker/compose.yaml logs --tail 40 mcp
```

Two failures account for most first starts, and neither looks like what it is:

**`PermissionError: [Errno 13] Permission denied: '/app/data/pdfs'`.** The data
directory is not owned by `MMW_UID:MMW_GID`. Check the parent, not just what is
under it — a root-owned parent with correctly-owned children is exactly what
you get from creating the directory one way and its contents another:

```sh
ls -ldn /var/lib/mcp-manual-walker
sudo chown -R mmw:mmw /var/lib/mcp-manual-walker
```

**Nothing answers at the embedding endpoint.** `localhost` in `.env` means the
container itself, where nothing is listening — the value has to be an address
the container can reach, so a service on the host or the LAN needs its real IP.
Ask the container rather than the host, because they do not share a network:

```sh
sudo docker compose -f /opt/mcp-manual-walker/compose.yaml exec mcp \
    python -c "import urllib.request as u; print(u.urlopen('http://192.168.0.1:11434/v1/models', timeout=8).status)"
```

Take the containers down before restarting the unit if one is already looping:
`up` treats a restarting container as present and will not replace it.

```sh
cd /opt/mcp-manual-walker && sudo docker compose down
sudo systemctl restart mcp-manual-walker
```

### Backing it up

Stop the writers first. Qdrant keeps segment files memory-mapped and SQLite
keeps a write-ahead log, so a copy taken while either is running can be a
copy of a half-finished write.

```sh
docker compose stop
tar -czf mmw-backup-$(date +%F).tar.gz \
    --exclude=data/model-cache --exclude='data/*.zip' data/
docker compose start
```

To restore, put the directory back and start:

```sh
docker compose down
rm -rf data/ && tar -xzf mmw-backup-2026-09-09.tar.gz
docker compose up -d
```

An export archive is the other way to move a corpus, and the better one
between *different* machines or backends, because it carries no engine's
on-disk format:

```sh
docker compose --profile tools run --rm db_manager export \
    --target . --output /app/data/corpus.zip
```

It is smaller than the raw directories (the vectors compress) and it can be
imported into either backend — but it takes minutes rather than seconds, so
for a same-machine snapshot the tarball is the right tool.

Under systemd, `systemctl stop mcp-manual-walker` and `start` replace the
`docker compose stop` / `start` above, and the directory to archive is
`MMW_DATA_DIR` rather than `data/`. Stopping the containers behind the unit's
back leaves it believing the stack is up.

| Variable | Default | What it does |
| --- | --- | --- |
| `EMBEDDING_API_BASE` | *(required)* | The endpoint the default image embeds through. Not needed with `--profile local`. |
| `MMW_DATA_DIR` | `./data` | The host directory bind-mounted at `/app/data`. An absolute path when the data does not belong beside the checkout. |
| `MCP_PORT` | `8000` | Host port for the server. |
| `MCP_BIND` | `127.0.0.1` | Host address to publish it on. |
| `QDRANT_VERSION` | `v1.19.1` | Qdrant image tag. |
| `QDRANT_HTTP_PORT` / `QDRANT_GRPC_PORT` | `6333` / `6334` | Host ports for Qdrant, on loopback. |
| `MMW_UID` / `MMW_GID` | `1000` / `1000` | Who the containers run as, so that what they write into the data directory belongs to you. |

Only the variables the compose file names reach the containers — `.env` is read
for interpolation, not injected wholesale, which is why `HOST` and `PORT` can
sit in the same file and mean something else. `EMBEDDING_QUERY_PREFIX` and
`EMBEDDING_DOCUMENT_PREFIX` are passed as bare keys rather than with a `:-`
default, so leaving them out of `.env` leaves them unset in the container and
the model's own stored prompt applies. Anything else you want the container to
see has to be added to `compose.yaml` the same way.

## 🛠️ Usage

The server provides a set of tools for AI agents. The typical workflow is as follows:

1.  Call `list_manuals()` to browse the library. It answers with the entries of one folder at a time, so descend with `list_manuals(folder)` until the manuals themselves appear, each with its unique ID.
2.  Use `get_manual_metadata(manual_id)` to retrieve the hierarchical table of contents (bookmarks) for a specific manual.
3.  Finally, call `get_markdown_content(bookmark_id)` to get the token-efficient Markdown content for the desired section.
4.  When a section or a search hit points at a figure, call `get_figure(figure_id)` to look at the picture itself.

This process allows an agent to intelligently navigate large documents and only pull the necessary information into its context.

### Tools

| Tool | Input | Output |
| --- | --- | --- |
| `list_manuals` | optional `folder` | The entries directly inside that folder (the library root when omitted). Each is either a `"directory"`, with the `path` to pass back and the `manual_count` it holds at any depth, or a `"manual"`, with the `id` the other tools need. |
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

## 🔌 The REST API

MCP is how an agent uses this corpus. It is not the only kind of caller: a RAG
pipeline retrieving passages for a prompt, a script checking what a manual says
about an error code, and the browser console below all want to ask the same
question over plain HTTP. The server answers them on a second port, in the same
process, out of the same index — the vector store is already open and the
embedding model is already resident, and a separate service would need its own
copy of both.

It is on by default, on `127.0.0.1:8001`; `REST_ENABLED=false` turns it off and
`REST_HOST`/`REST_PORT` move it. Interactive documentation, generated from the
same pydantic models the MCP tools return, is at `/docs`.

```bash
curl -s localhost:8001/health

curl -s localhost:8001/search \
  -H 'Content-Type: application/json' \
  -d '{"query": "how do I mount a zFS file system", "limit": 5}'

# The same search, as a GET, for when a browser address bar is what you have.
curl -s 'localhost:8001/search?q=IEF450I&limit=3'
```

| Endpoint | Answers with |
| --- | --- |
| `GET /health` | Whether a search can be answered, the backend and model in use, and how many manuals and chunks are held. Never requires the API key. |
| `POST /search` | The best matching chunks for `query`, with `manual_id`, `bookmark_id` and `limit` all optional. |
| `GET /search?q=…` | The same search from query-string parameters. |
| `GET /manuals?folder=…` | One folder of the library, one level deep, exactly as `list_manuals` returns it. |
| `GET /manuals/{manual_id}` | The manual's metadata and hierarchical table of contents. |
| `GET /bookmarks/{bookmark_id}/markdown` | The Markdown of that section and its subsections, plus the figures in it. |
| `GET /figures/{figure_id}` | A figure's caption, labels, description and size. |
| `GET /figures/{figure_id}/image` | The PNG itself, so an `<img src>` can point straight at it. |

**A search here does not need a manual.** The `search_manual` tool requires one
because an agent has already browsed its way to it; a RAG client has a question
and nothing else, and making it choose a manual first would be asking it to
solve retrieval before it may use retrieval. So `manual_id` narrows a search
rather than starting one, `bookmark_id` narrows it to a section — and since a
bookmark names exactly one manual, `bookmark_id` alone is enough.

Each hit carries more than the tool returns, because a client showing a result
list has to say where it came from:

```json
{
  "query": "how do I mount a zFS file system",
  "manual_id": null,
  "bookmark_id": null,
  "limit": 5,
  "count": 5,
  "results": [
    {
      "chunk_id": "8f0c…", "rank": 1, "score": 0.71, "retrieval": "both",
      "chunk_type": "text", "page": 214,
      "manual_id": "d10e…",
      "manual": {
        "id": "d10e…", "file_name": "zos-unix-sysadmin.pdf",
        "document_title": "z/OS UNIX System Services Planning",
        "relative_path": "z/OS/v2r5/zos-unix-sysadmin.pdf"
      },
      "bookmark_id": "7e14…",
      "bookmarks": [{"id": "ed98…", "title": "Chapter 8", "page": 201, "children": []}],
      "context": "…the text of the chunk…",
      "figure": null
    }
  ]
}
```

`score` is the cosine similarity to the query, and it is `null` for a hit that
only BM25 returned: such a chunk was never scored against the query vector, and
inventing a number for it would be worse than saying nothing. `retrieval` says
which half found it — `"dense"`, `"lexical"` or `"both"`. The ranking itself is
the same rank fusion the MCP tool uses, described under "Lexical retrieval
alongside the vectors".

### Closing it

`REST_API_KEY` is empty by default, which leaves the API open — the right
answer only while it is bound to loopback or to a private network. Setting it
requires every request except `/health` to present the key:

```bash
curl -s localhost:8001/search -H 'X-API-Key: …' -d '{"query": "…"}'
curl -s localhost:8001/search -H 'Authorization: Bearer …' -d '{"query": "…"}'
```

`/health` stays open on purpose: a container health check and a load balancer
call it, and neither should have to hold a credential to ask whether the
process is alive. It reveals sizes and the model name, never content.

### Two doors

Everything that is not a browser talks to `127.0.0.1:8001` directly; that is
what compose publishes and what the examples above use. A page in a browser has
a second option: the console's nginx forwards `/api` to the same server, so
`http://127.0.0.1:8080/api/search` reaches the same endpoint from the page's own
origin. Nothing says the second door is only for the page — publish `8080` and
leave `8001` unpublished and nginx becomes the single front door for
everything.

Which door a browser uses decides whether CORS is involved. Through `/api` it is
not: same origin, no preflight, nothing to configure. Straight to `:8001` it is,
and `REST_CORS_ORIGINS` has to name the page's origin. It defaults to
`http://localhost:8080` and `http://127.0.0.1:8080` — the console's own address
— and empty sends no CORS headers at all.

It does not default to `*`, and the reason is worth stating: binding to loopback
keeps external HTTP clients out, but it does not keep browsers out. A page on
any site the person running this server happens to visit can issue requests to
`127.0.0.1:8001`; without `Access-Control-Allow-Origin` the browser blocks it
from *reading* the answer, and `*` is precisely what would let it read — and
exfiltrate — the corpus. Name the origins you mean, or set an API key, or both.

## 🖥️ The browser console

A page for trying a query by hand: type a question, see what comes back, read
the section a hit belongs to, look at a figure. It is nginx serving three
static files in a container of its own, behind a profile, because it is a way
to look at the corpus rather than part of serving it.

```bash
docker compose --profile gui up -d --build
# http://127.0.0.1:8080
```

<!-- A screenshot belongs here once the console has been run against a real corpus. -->

The console proxies `/api` to the server over the compose network, so the page
and the API share an origin, no CORS is involved and the REST port does not
have to be published to the host at all for it to work. To point it at a
different deployment, open **Settings** and put an absolute URL in **API base
URL** — `http://localhost:8001` to go straight to the API rather than through
the proxy, which is the case `REST_CORS_ORIGINS` exists for. The API key field
beside it fills in `X-API-Key` for a server that wants one. Both are kept in
the browser, not on the server.

`WEBGUI_PORT` and `WEBGUI_BIND` move where it is published.

Moving the API itself is `REST_PORT`, and it is one setting rather than three:
it is the port the server listens on, the port compose publishes, and the port
the console proxies to, all at once. `WEBGUI_API_UPSTREAM` only exists to
change the *host* — `http://mcp-local:8001` when the `local` profile is the one
running, or a server outside this compose file altogether. It is an address on
the compose network, so it takes the port the container listens on, never the
published one; pointing it at the published port is how you get a `502` from
nginx with a perfectly healthy server behind it.

There is no build step and no bundler. `webgui/static/` is HTML, CSS and one
JavaScript file that a browser runs as they are, so editing the console needs
nothing installed and a rebuild of that image is a file copy.

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
worker processes and the parent. The embedder also hands the device back when
it is done: torch's caching allocator keeps the activation peak in its own
pool and the weights stay resident, so an idle embedder was still holding
5.1 GB against the 1.3 GB it starts with. A slot returned while that memory is
held frees the right to run but not the room to run in, so the weights are
moved off and the pool emptied *before* the slot is released — after would
race the worker that took it.

A slot is held for one conversion or one
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
| `VECTOR_BACKEND` | `chroma` | Which vector database holds the chunks: `chroma` (embedded) or `qdrant` (a server). See [Choosing a backend](#choosing-a-backend). |

Chunking builds one markdown serializer per document rather than per table.
`TableItem.export_to_markdown(doc)` constructs a `MarkdownDocSerializer` on
every call, and constructing one revalidates the whole document — pydantic
runs `validate_document`, clamping every table cell's bounding box on every
page — so the cost is *tables × cells in the document*. On a table-dense
manual that dominates the build: a 490-page font reference spent over eleven
minutes in chunking after a 46-minute conversion, with the parent's RSS
climbing from 6 GB to 14.6 GB on a 31 GB host. Measured on a synthetic
100-page document whose cells carry provenance, with identical output:

| tables | cells | per table | one shared |
| --- | --- | --- | --- |
| 25 | 4,000 | 0.58 s | 0.07 s |
| 100 | 16,000 | 8.15 s | 0.31 s |
| 200 | 32,000 | 31.82 s | 0.61 s |

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

### 📦 Moving a built database

`db_manager export --target <prefix> --output <file>.zip` packs everything a
built corpus consists of into one archive, and `db_manager import --input
<file>.zip` merges it into another machine's database. A manual already
present by id is skipped, so importing twice is harmless and two archives can
be merged into one database.

The archive is a zip holding four things:

| Member | What it is | How it is stored |
| --- | --- | --- |
| `manifest.json` | Counts, target, and the embedding model | deflate |
| `sqlite.json` | Manual, bookmark and figure rows | deflate |
| `chunks.jsonl.zst` | One JSON chunk per line — id, vector, metadata, text | zstd, stored |
| `figures/<id>.png` | The figure image bytes | stored |

The chunk stream is almost the whole archive — 10.1 GB of the 12.3 GB in a
zOS/V3R1 export, nearly all of it vector text — so how it is compressed is the
only decision that matters here. It is a zstd frame written into a *stored*
zip member rather than a deflated one, for two reasons:

*   **zstd is parallel; deflate is not.** Python's `zipfile` compresses on one
    core, and that is where the previous export spent roughly 25 minutes. Over
    the whole corpus, deflate produced 1,942 MB and zstd level 6 produced
    1,517 MB — faster *and* smaller. Level 9 would save a further 60 MB for
    about 37 s more; level 12 is worse than 9 on both axes, and long-distance
    matching gains nothing because embedding text has no long-range repeats.
*   **Neither side has to stage it.** Chroma is read, serialized, compressed
    and written into the archive in one pass, and an import decompresses
    straight out of the open zip. Nothing lands on disk uncompressed, so
    exporting no longer needs 10 GB of scratch space beside the archive and
    importing no longer needs room for the archive *and* its contents.

The figure PNGs are stored rather than deflated because they are already
compressed: deflating 2,140 MB of them saved 140 MB and cost roughly a third
of the export's runtime. Giving that back is why the whole archive only fell
from 3.96 GB to 3.68 GB while the chunk stream lost 425 MB — but the export
went from around 30 minutes to 8 and a half, and stopped needing scratch space
at all.

One warning if you ever re-tune the level: this corpus is not uniform. Its
first 20,000 chunks compress at 8.83x under zstd against 6.37x in the middle,
so a sample taken from the head understates the finished archive by a third.
Deflate barely varies over the same slices (5.33x / 5.10x / 5.13x), so
sanity-checking such a sample against deflate does not catch it.

`manifest.json` records the `EMBEDDING_MODEL` that produced the vectors, and
an import into a database configured for a different model is refused —
vectors are only meaningful in the space they were built in. `format_version`
tracks the layout; archives written by earlier versions (`chroma.json`, or a
plain `chunks.jsonl`) still import.

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
    member of the archive and `db_manager import` restores the rows with their
    bytes; older archives without figures still import unchanged. Deleting a
    manual deletes its figures with it.
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

### 🔍 The vector index

Chunks reach the vector database through one interface,
`vector_store.VectorStore` — eight operations and three filter shapes, which is
everything the builder, the search server and `db_manager` actually ask of it.
`VECTOR_BACKEND` chooses the implementation: `chroma` (embedded, the default)
or `qdrant` (a server).

The interface exists because Chroma's HNSW index is held in an anonymous
hnswlib arena that has to be resident, and past roughly a million 1024-dim
chunks that arena stops fitting in RAM. Nothing in the application depended on
Chroma in particular — it depended on eight calls — so the calls were named and
the engine put behind them. Two things every backend has to normalise:

*   **Score direction.** Chroma returns a cosine *distance* (smaller is
    better), most other engines a similarity. `Hit.score` is always a
    similarity.
*   **Chunk identity.** The application's ids are strings
    (`"<manual_id>_<n>"`). Chroma stores them as they are; an engine that
    accepts only integers or UUIDs has to map them itself and keep the original
    in its payload, so that the BM25 index in SQLite and the export archive can
    go on naming chunks the way they already do.

Because the export archive holds nothing engine-specific — one JSON object per
chunk, with `id`, `embedding`, `metadata` and `document` — it is also the
migration path between backends: export from one, import into another.

#### Choosing a backend

|  | Chroma | Qdrant |
| --- | --- | --- |
| Deployment | embedded, no server | a server process (Docker) |
| Resident memory, 684,398 × 1024-dim | **~2.9 GB**, all of it heap | **~350 MB** heap, plus ~2.7 GB the kernel may reclaim |
| Behaviour when RAM runs short | fails | gets slower |
| Right for | a corpus that fits | a corpus that does not |

The memory line is the whole reason the second backend exists, and it is a
difference in kind rather than in degree. Chroma's vectors and graph live in an
hnswlib arena the kernel cannot take back. Qdrant memory-maps both — its
segments report `storage_type: InRamMmap` even when nothing has been asked to
go "on disk" — so the same data is page cache.

The consequence is visible under a hard cgroup limit. The whole
684,398-chunk corpus, in a container capped with `--memory` and swap disabled:

| Container cap | 8 GB | 4 GB | 2 GB | 1 GB | **512 MB** |
| --- | --- | --- | --- | --- | --- |
| One manual, p50 | 2.83 ms | 4.37 ms | 3.92 ms | 2.92 ms | **2.58 ms** |
| Whole corpus, p50 | 5.20 ms | 6.71 ms | 5.84 ms | 5.52 ms | **5.02 ms** |

**Read that as "the process does not need the memory", not as "the machine
does not".** The host had about 10 GB free throughout, so the mmapped pages
stayed in the global page cache even when the container's own cgroup could not
be charged for them. On a host that is genuinely short of memory those pages
get evicted and re-read, and the times rise. The claim the table supports is
the narrow one: nothing was killed and nothing degraded at any cap, because
none of that data has to be anonymous memory. Chroma at the same size cannot
be given a 512 MB cap at all.

The HNSW graph itself is 43 MB on disk for those 684,398 vectors, against
2,677 MB of vector storage — Qdrant compresses the links, so the graph was
never the thing that did not fit.

**Quantization and on-disk vectors are deliberately off.** They were measured
on the same corpus and made matters worse: `int8` with `always_ram` *added*
about 700 MB of unreclaimable memory the mmapped originals do not need, and
took a filtered search from 2.6 ms to 20.4 ms. Binary quantization cost recall
(93.2% against 96.2%) for no memory saving. The settings exist for a corpus
several times this one.

**Per-tenant graphs are off too, and should stay off if you ever want to search
across manuals.** Qdrant can skip the global graph (`m: 0`) and build one per
value of a payload field, which is tempting here because every `search_manual`
call carries a `manual_id`. It indexes in 30 s instead of 405 s and answers a
filtered search perfectly — and answers a corpus-wide one at **0.2% recall@5**,
because there is no global graph left to walk.

One Qdrant-specific behaviour worth knowing: it **L2-normalizes the vectors of
a cosine collection when it stores them**. An export taken from Qdrant
therefore returns unit vectors rather than the exact numbers that went in — on
3,273 real chunks whose stored norms ranged 0.998047–1.003795, the largest
per-dimension difference was 5.99e-04 as stored and 2.42e-08 once the source
was normalized too. Cosine ranking is scale-invariant, so retrieval is
unaffected; only a byte-for-byte archive comparison will notice.

```sh
uv sync --extra cpu --extra qdrant
docker run -d -p 6333:6333 -p 6334:6334 \
    -v "$(pwd)/qdrant_storage:/qdrant/storage" qdrant/qdrant
```

```.env
VECTOR_BACKEND=qdrant
QDRANT_URL=http://localhost:6333
```

An existing Chroma database moves across without a rebuild, because the archive
is backend-neutral:

```sh
VECTOR_BACKEND=chroma uv run db_manager export --target . --output corpus.zip
VECTOR_BACKEND=qdrant uv run db_manager import --input corpus.zip
```

| Variable | Default | What it does |
| --- | --- | --- |
| `QDRANT_URL` | `http://localhost:6333` | Where the server is. |
| `QDRANT_API_KEY` | *(empty)* | Sent when set. |
| `QDRANT_GRPC_PORT` | `6334` | gRPC port. |
| `QDRANT_PREFER_GRPC` | `true` | A build uploads hundreds of thousands of 4 kB vectors; JSON-encoding them is the expensive part. |
| `QDRANT_COLLECTION` | `manual_chunks` | Collection name. |
| `QDRANT_TIMEOUT` | `60.0` | Request timeout, in seconds. |
| `QDRANT_HNSW_EF` | `128` | Candidates held during a graph traversal. Measured 96.2% recall@5 corpus-wide against an exact scan of the same collection. |
| `QDRANT_ON_DISK_VECTORS` | `false` | Qdrant already mmaps vectors it calls "in RAM"; see above. |
| `QDRANT_ON_DISK_PAYLOAD` | `false` | Keeps the chunk text out of memory. Measured no benefit here. |
| `QDRANT_QUANTIZATION` | `none` | `none`, `int8` or `binary`. See above before turning this on. |
| `QDRANT_OVERSAMPLING` | `2.0` | Shortlist multiplier when rescoring a quantized search. Ignored when quantization is `none`. |

The graph parameters are pinned in `collection_metadata()` rather than left to
Chroma's defaults. The defaults (`max_neighbors` 16, `ef_construction` 100) are
sized for smaller collections, and at half a million vectors they leave the
graph fragile enough that the **order the vectors arrive in** decides how good
it is.

Measured on this corpus — 504,346 chunks, 50 real questions in Japanese and
English, recall@5 against an exact full scan:

| how the database was made | recall@5 | ja | en |
| --- | --- | --- | --- |
| built incrementally, Chroma defaults | 89.2% | 85.0% | 92.0% |
| imported from an archive, Chroma defaults | 60.4% / 67.6% | 56.0% / 64.0% | 63.3% / 70.0% |
| imported from an archive, `M=32`, `ef_construction=200` | **96.8%** | 97.0% | 96.7% |

The two import rows are the same code on the same data: HNSW draws node levels
at random, so two builds differ by several points on their own. What they share
is arriving **grouped by manual**, which is how an archive hands chunks over,
and that costs roughly 25 points at the default settings. A build that
interleaves documents — which is what the builder does, three workers finishing
different manuals at different times — happens to produce a better graph.

Raising the two parameters removes the dependence on order rather than papering
over it. The 96.8% run was inserted in that same worst-case order and still beat
the incremental build. Shuffling on import would only help within one archive;
a database fed several archives gets each one as a clustered block anyway.

Japanese queries gain the most (56.0% → 97.0%). A sparse graph strands them in a
local minimum more often — the greedy descent stops as soon as nothing in its
candidate list beats the current worst, and a query further from the document
cluster it wants has more chances to settle early.

The cost is small: an import of this corpus goes from 17:59 to 20:47, the index
grows about 1%, and a query goes from 2.2 ms to 2.7 ms. `ef_search` was measured
too and is not worth raising — it moved recall by 0–4 points while adding 75% to
query latency.

**These parameters are only read when a collection is created.** An existing
database keeps whatever it was built with; check with:

```sh
uv run python -c "import chromadb; \
  print(chromadb.PersistentClient(path='./data/db/chroma_db') \
        .get_collection('manual_chunks').configuration_json['hnsw'])"
```

To adopt them without re-converting anything, export the corpus and import it
into a fresh database — that is 21 minutes, against hours for a rebuild.

> **Recall@5 here means agreement with an exact vector scan, not answer
> quality.** Search is dense-only: there is no BM25 or lexical matching, so an
> exact scan itself returns nothing containing `IEF450I` for the query "what
> does message IEF450I mean", even though 13 chunks contain that string. Index
> tuning closes the gap to exact search; it cannot close that one.

### 🔤 Lexical retrieval alongside the vectors

Dense retrieval cannot find an identifier. Measured against an **exact** scan of
all 504,346 vectors — no approximation involved — the top 5 for "what does
message IEF450I mean" contained no chunk with that string in it, though 13
chunks do. Same for `IEC141I` (14 chunks), `S0C4` (20) and `S806` (3).
Embeddings place `IEF450I` and `IEF451I` almost on top of each other and give a
rare token no extra weight. BM25 does the opposite.

So `search_manual` asks both and fuses by rank:

```
dense top 20  +  BM25 top 20  ->  Reciprocal Rank Fusion (k=60)  ->  top 5
```

The lexical index is an FTS5 table in the manuals database. It is **derived
data and never travels in an export archive** — rebuilding it takes 90 seconds
against the 21 minutes an import costs, so an archive stays free of a search
implementation detail. `db_manager reindex-lexical` builds it for a database
that predates the feature.

`unicode61` tokenizes it, which does not segment Japanese and does not need to:
the corpus is **0.00% Japanese** (67 CJK characters in 3.16 million sampled).
A Japanese analyzer would have nothing to match against. What Japanese queries
carry is identifiers — 12 of the 20 in the evaluation set, including every one
dense retrieval failed — because "メッセージ IEF450I の意味" still spells
IEF450I. Morphological analysis would be the right call for a corpus with
Japanese body text; this one has none.

#### Only rare terms are asked of BM25

This is the part that took measuring. Fusing every query with BM25 improved
identifier lookups and **wrecked everything else**: agreement with an exact
vector scan fell from 95.6% to 59.2%.

The cause is not BM25, which ranks correctly — asked for "what does message
IEF450I mean" it puts chunks containing IEF450I at ranks 2 and 3, inverse
document frequency working as designed. The cause is the other kind of query.
"how do I mount a zFS file system" has no rare term at all (its rarest, `zFS`,
is in 1,972 chunks) and its terms together match 150,279 chunks, 30% of the
corpus. BM25 returns a ranked 20 of them chosen by how densely they repeat
ordinary words, and RRF cannot tell that list from a confident one: it sees
ranks, never scores. Those 20 displace dense hits that were right.

So a term is only sent to BM25 if it appears in at most
`MAX_TERM_DOCUMENT_FREQUENCY_RATIO` of the chunks (0.0005, i.e. 252 here), and a
query left with no such term never reaches the lexical index at all:

| | dense | hybrid |
| --- | --- | --- |
| identifier present in top 5 (10 queries naming one) | 16/50 | **43/50** |
| agreement with exact scan, 41 queries BM25 never saw | 200/205 | **200/205** |
| agreement with exact scan, 9 queries it did | 39/45 | 4/45 |
| added latency | — | 6.7 ms/query |

The middle row is the one that matters: **zero collateral damage.** The bottom
row is not damage but the trade itself — on a query the gate has judged
lexical, a lexical hit deliberately replaces a semantic one, and dense is
measurably useless there.

#### Why the lexical list is weighted and steep

The dense list is fused at the usual flat `k=60`; the lexical one at `k=5` and
weight 0.3. Something has to give here and it is arithmetic, not taste: getting
BM25's 4th hit into an overall top 5 leaves room for at most one dense hit
above it, so BM25's first four come too. **No setting surfaces a message
definition and also keeps the dense ranking.**

What the shape buys is a bounded tail. At `k=5`, weight 0.3, the 20th lexical
hit scores 0.012 against the best dense hit's 0.016, so a long lexical list
cannot swamp the dense one. A flat curve at higher weight (`k=60`, weight 1.5)
scores identically on every measure here while putting the 20th lexical hit
*above* the 1st dense one — at which point it is not fusion, it is a switch.

> The corpus size is recorded in the index rather than counted per query.
> `SELECT count(*)` on an FTS5 table is a full scan: 340 ms here, against 6.5 ms
> for the whole lexical path once it is stored.

#### What this still does not solve

Searched across the whole corpus, "what does message IEF450I mean" still does
not put the *authoritative* chunk — the one reading `IEF450I Explanation A job
step abnormally ended` — in the top 5. It is at rank 13, below console-log
fragments from other manuals that genuinely do contain `IEF450I`.

**Narrowing the search to the manual that documents the message fixes it.**
Scoped to *MVS System Messages, Vol 8 (IEF-IGD)*, the same query puts the
definition at rank 4:

| | BM25 rank | fused rank | in top 5 |
| --- | --- | --- | --- |
| whole corpus | 13 | 13 | no |
| scoped to Vol 8 | 4 | **4** | **yes** |

which is what `search_manual`'s `manual_id` is for. Dense retrieval does not
find that chunk either way — not even within the 1,437 chunks of the volume
holding it.

Bookmarks are no help as a third signal: `BPX1MNT` has two bookmarks naming it,
but `IEF450I`, `IEC141I` and `S0C4` have none, because the System Messages
volumes carry no per-message outline entries.

So the remaining gap is unscoped search over the whole library. Closing that
needs a reranker over the fused candidates, not another retriever.

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

#### 🌐 Embedding through a remote endpoint

The search server does not have to load the model at all. Set
`EMBEDDING_BACKEND=openai` and it will embed each query through an
OpenAI-compatible `/v1/embeddings` endpoint — [Lemonade
Server](https://lemonade-server.ai/), Ollama, vLLM, llama.cpp's server, or
anything else that speaks the same shape — instead of holding 1.11–2.22 GiB of
weights and a torch install in its own process.

This is for the **server**, not the builder. A query is one short text; a build
is hundreds of thousands of 1.2 kB chunks, and an HTTP round trip per batch is
not the way to embed those. Leave the builder on `EMBEDDING_BACKEND=local` with
its GPU.

One data point, on one setup — a Lemonade Server on the LAN serving
`Qwen3-Embedding-0.6B-GGUF` (Q8_0, llama.cpp/Vulkan):

| | |
| --- | --- |
| One query, warm | 24 ms (p50) |
| A whole chunk (~300 tokens) | ~320 ms |

The absolute figures belong to that endpoint and that network; yours will
differ, and whether they beat embedding in-process depends on which machine is
better. **The ratio is the part that carries over**, because it follows the
token count rather than the hardware: a chunk costs roughly an order of
magnitude more than a query. That is what makes an endpoint the right place for
one query per request and the wrong place for a whole corpus.

**A quantized GGUF of the same model is vector-compatible with a database built
from the full-precision checkpoint.** Measured on 64 chunks drawn from a real
export archive, re-embedded through the endpoint and compared with the vectors
the builder had stored for the same text:

| | |
| --- | --- |
| Cosine similarity vs. the stored vector | **0.9994** mean, 0.9991 worst |
| Top-5 neighbour agreement | **98.4 %** |

So switching an existing server to the remote backend needs no rebuild. That is
also why `EMBEDDING_MODEL` — not `EMBEDDING_API_MODEL` — remains the name
recorded on the collection: the collection identifies a *vector space*, and
which process produced today's vectors is a deployment detail. The endpoint's
own id for the model is usually different anyway (`Qwen3-Embedding-0.6B-GGUF`
against `Qwen/Qwen3-Embedding-0.6B`), and folding it into the recorded identity
would make the startup check reject a database it can read perfectly well.

One thing the remote backend has to carry itself: **the instruction prefix**.
With Sentence Transformers the prefixes come from the model's own
`config_sentence_transformers.json`; an OpenAI-compatible endpoint exposes no
such thing and embeds exactly the string it is sent. The known prompts for the
Qwen3-Embedding family are therefore built into `embeddings.py`. For any other
model, set `EMBEDDING_QUERY_PREFIX` and `EMBEDDING_DOCUMENT_PREFIX` explicitly
— getting this wrong is silent, since the vectors are still 1024 well-formed
numbers that simply answer a slightly different question than the stored ones.

```.env
EMBEDDING_BACKEND=openai
EMBEDDING_API_BASE=http://192.168.250.1:13305/v1
EMBEDDING_API_MODEL=Qwen3-Embedding-0.6B-GGUF
# EMBEDDING_MODEL stays as it is: it names the vector space, not the endpoint.
EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B
```

| Variable | Default | What it does |
| --- | --- | --- |
| `EMBEDDING_BACKEND` | `local` | `local` loads the model with Sentence Transformers in-process; `openai` calls a remote OpenAI-compatible endpoint, and `remote` is accepted as a second spelling of it. |
| `EMBEDDING_API_BASE` | *(empty)* | Base URL of the endpoint, e.g. `http://localhost:11434/v1`. Required when the backend is `openai`. |
| `EMBEDDING_API_MODEL` | *(uses `EMBEDDING_MODEL`)* | The id the **endpoint** knows the model by. Not recorded on the collection. |
| `EMBEDDING_API_KEY` | *(empty)* | Sent as `Authorization: Bearer …` when set. |
| `EMBEDDING_API_BATCH_SIZE` | `16` | Texts per request. |
| `EMBEDDING_API_TIMEOUT` | `120.0` | HTTP timeout per request, in seconds. |
| `EMBEDDING_API_MAX_RETRIES` | `3` | Attempts per request, with exponential backoff. |

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
