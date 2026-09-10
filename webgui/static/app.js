/*
 * The search console.
 *
 * Deliberately plain: no framework, no build step, no dependency to install or
 * to keep current. The whole point of this page is to try a query against the
 * REST API and see what comes back, and every line of it is a call to one of
 * the six endpoints the API serves.
 */

const DEFAULTS = {
  // Same origin, proxied to the server by nginx. Overridable because the
  // console is also useful pointed at a deployment somewhere else.
  base: "/api",
  key: "",
};

const settings = {
  base: localStorage.getItem("mmw.base") ?? DEFAULTS.base,
  key: localStorage.getItem("mmw.key") ?? DEFAULTS.key,
};

// The manual or section a search is restricted to. Null means the corpus.
let scope = null;

const $ = (id) => document.getElementById(id);

function headers() {
  const h = { "Content-Type": "application/json" };
  if (settings.key) h["X-API-Key"] = settings.key;
  return h;
}

function url(path) {
  return `${settings.base.replace(/\/$/, "")}${path}`;
}

async function call(path, options = {}) {
  const response = await fetch(url(path), { headers: headers(), ...options });
  if (!response.ok) {
    // The API says what went wrong in `detail`; showing anything else would be
    // hiding the one useful thing in the response.
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      if (body.detail) detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* a non-JSON error body is rare and the status line will do */
    }
    throw new Error(detail);
  }
  return response.json();
}

/* ------------------------------------------------------------------ health */

async function refreshHealth() {
  const el = $("health");
  try {
    const body = await call("/health");
    const ok = body.status === "ok";
    el.className = `health ${ok ? "ok" : "degraded"}`;
    el.textContent = ok
      ? `${body.manuals ?? "?"} manuals · ${(body.chunks ?? 0).toLocaleString()} chunks`
      : "degraded";
    el.title = body.detail ?? `${body.vector_backend} · ${body.embedding_model}`;
  } catch (error) {
    el.className = "health down";
    el.textContent = "unreachable";
    el.title = error.message;
  }
}

/* ------------------------------------------------------------------ search */

async function runSearch(event) {
  event?.preventDefault();
  const query = $("query").value.trim();
  if (!query) return;

  setStatus("Searching…", "working");
  $("submit").disabled = true;
  $("results").replaceChildren();

  const payload = { query, limit: Number($("limit").value) };
  if (scope?.type === "manual") payload.manual_id = scope.id;
  if (scope?.type === "bookmark") payload.bookmark_id = scope.id;

  try {
    const started = performance.now();
    const body = await call("/search", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    const elapsed = Math.round(performance.now() - started);

    if (body.count === 0) {
      setStatus("Nothing matched.", "empty");
    } else {
      setStatus(`${body.count} results in ${elapsed} ms`, "");
      body.results.forEach((hit) => $("results").append(renderHit(hit)));
    }
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    $("submit").disabled = false;
  }
}

function renderHit(hit) {
  const li = document.createElement("li");
  li.className = "hit";

  const head = document.createElement("div");
  head.className = "hit-head";

  const badge = document.createElement("span");
  badge.className = `badge ${hit.chunk_type}`;
  badge.textContent = hit.chunk_type;
  head.append(badge);

  const source = document.createElement("span");
  source.className = "source";
  source.textContent =
    hit.manual?.document_title || hit.manual?.file_name || hit.manual_id;
  source.title = hit.manual?.relative_path ?? "";
  head.append(source);

  if (hit.page != null) {
    const page = document.createElement("span");
    page.className = "page";
    page.textContent = `p.${hit.page}`;
    head.append(page);
  }

  const score = document.createElement("span");
  score.className = "score";
  // A lexical-only hit was never scored against the query vector, and the API
  // says so with a null rather than inventing a number.
  score.textContent =
    hit.score == null ? `#${hit.rank} · ${hit.retrieval}` : `#${hit.rank} · ${hit.score.toFixed(3)} · ${hit.retrieval}`;
  head.append(score);

  li.append(head);

  if (hit.bookmarks?.length) {
    const path = document.createElement("div");
    path.className = "path";
    path.textContent = hit.bookmarks.map((b) => b.title).join(" › ");
    li.append(path);
  }

  if (hit.figure) {
    const image = document.createElement("img");
    image.className = "figure";
    image.loading = "lazy";
    image.alt = hit.figure.caption ?? "figure";
    if (settings.key) {
      // An API key cannot ride along on an <img src>, so a guarded server's
      // figures are fetched like everything else and handed over as a blob.
      fetch(url(`/figures/${hit.figure.id}/image`), { headers: headers() })
        .then((r) => (r.ok ? r.blob() : Promise.reject(new Error(r.statusText))))
        .then((blob) => {
          image.src = URL.createObjectURL(blob);
        })
        .catch(() => image.remove());
    } else {
      image.src = url(`/figures/${hit.figure.id}/image`);
    }
    li.append(image);
  }

  const context = document.createElement("pre");
  context.className = "context";
  context.textContent = hit.context;
  li.append(context);

  const actions = document.createElement("div");
  actions.className = "actions";

  if (hit.bookmark_id) {
    actions.append(
      button("Read the section", () => showSection(hit.bookmark_id, hit.bookmarks))
    );
    actions.append(
      button("Search in this section", () => {
        setScope({
          type: "bookmark",
          id: hit.bookmark_id,
          label: hit.bookmarks?.at(-1)?.title ?? "this section",
        });
        runSearch();
      })
    );
  }
  actions.append(
    button("Search in this manual", () => {
      setScope({
        type: "manual",
        id: hit.manual_id,
        label: hit.manual?.file_name ?? hit.manual_id,
      });
      runSearch();
    })
  );

  li.append(actions);
  return li;
}

/* ------------------------------------------------------------------- scope */

function setScope(next) {
  scope = next;
  const el = $("scope");
  el.replaceChildren();

  if (!next) {
    const all = document.createElement("span");
    all.className = "scope-all";
    all.textContent = "the whole corpus";
    el.append(all);
    return;
  }

  const chip = document.createElement("span");
  chip.className = "chip";
  chip.textContent = next.label;
  chip.append(
    button("×", () => {
      setScope(null);
    }, "chip-clear")
  );
  el.append(chip);
}

/* ----------------------------------------------------------------- library */

async function showFolder(folder) {
  const list = $("entries");
  const crumbs = $("breadcrumb");
  list.replaceChildren();
  crumbs.replaceChildren();

  const parts = folder ? folder.split("/") : [];
  crumbs.append(button("library", () => showFolder(""), "crumb"));
  parts.forEach((part, index) => {
    const path = parts.slice(0, index + 1).join("/");
    crumbs.append(document.createTextNode(" / "));
    crumbs.append(button(part, () => showFolder(path), "crumb"));
  });

  try {
    const entries = await call(
      `/manuals${folder ? `?folder=${encodeURIComponent(folder)}` : ""}`
    );
    entries.forEach((entry) => {
      const li = document.createElement("li");
      if (entry.type === "directory") {
        li.append(
          button(`📁 ${entry.name}`, () => showFolder(entry.path), "entry")
        );
        const count = document.createElement("span");
        count.className = "count";
        count.textContent = `${entry.manual_count} manuals`;
        li.append(count);
      } else {
        li.append(
          button(
            `📄 ${entry.document_title || entry.name}`,
            () => {
              setScope({ type: "manual", id: entry.id, label: entry.name });
              $("browser").hidden = true;
            },
            "entry"
          )
        );
        li.append(button("Contents", () => showToc(entry.id), "ghost small"));
      }
      list.append(li);
    });
  } catch (error) {
    setStatus(error.message, "error");
  }
}

async function showToc(manualId) {
  try {
    const manual = await call(`/manuals/${manualId}`);
    openDetail(manual.document_title || manual.file_name, (body) => {
      const tree = document.createElement("ul");
      tree.className = "toc";
      const walk = (nodes, into) => {
        nodes.forEach((node) => {
          const li = document.createElement("li");
          li.append(
            button(`${node.title} (p.${node.page})`, () => showSection(node.id, [node]), "link")
          );
          if (node.children?.length) {
            const sub = document.createElement("ul");
            walk(node.children, sub);
            li.append(sub);
          }
          into.append(li);
        });
      };
      walk(manual.table_of_contents, tree);
      body.append(tree);
    });
  } catch (error) {
    setStatus(error.message, "error");
  }
}

async function showSection(bookmarkId, path) {
  const title = path?.map((b) => b.title).join(" › ") ?? "Section";
  openDetail(title, (body) => {
    body.textContent = "Loading…";
  });

  try {
    const content = await call(`/bookmarks/${bookmarkId}/markdown`);
    openDetail(title, (body) => {
      const pre = document.createElement("pre");
      pre.className = "markdown";
      pre.textContent = content.markdown_content || "(this section has no text)";
      body.append(pre);
    });
  } catch (error) {
    openDetail(title, (body) => {
      body.className = "error";
      body.textContent = error.message;
    });
  }
}

/* --------------------------------------------------------------- utilities */

function button(label, onClick, className = "ghost") {
  const el = document.createElement("button");
  el.type = "button";
  el.className = className;
  el.textContent = label;
  el.addEventListener("click", onClick);
  return el;
}

function setStatus(text, kind) {
  const el = $("status");
  el.textContent = text;
  el.className = `status ${kind}`;
}

function openDetail(title, fill) {
  $("detail-title").textContent = title;
  const body = $("detail-body");
  body.className = "";
  body.replaceChildren();
  fill(body);
  if (!$("detail").open) $("detail").showModal();
}

/* ------------------------------------------------------------------- setup */

$("search-form").addEventListener("submit", runSearch);
$("detail-close").addEventListener("click", () => $("detail").close());
$("browse-toggle").addEventListener("click", () => {
  const browser = $("browser");
  browser.hidden = !browser.hidden;
  if (!browser.hidden) showFolder("");
});
$("settings-toggle").addEventListener("click", () => {
  $("settings").hidden = !$("settings").hidden;
});

$("api-base").value = settings.base;
$("api-key").value = settings.key;
$("api-base").addEventListener("change", (e) => {
  settings.base = e.target.value.trim() || DEFAULTS.base;
  localStorage.setItem("mmw.base", settings.base);
  refreshHealth();
});
$("api-key").addEventListener("change", (e) => {
  settings.key = e.target.value.trim();
  localStorage.setItem("mmw.key", settings.key);
  refreshHealth();
});

setScope(null);
refreshHealth();
