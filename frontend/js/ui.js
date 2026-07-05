/**
 * DOM rendering helpers. Pure presentation — takes data, updates the DOM.
 * No fetch calls live here (see api.js); no orchestration (see app.js).
 */
const Ui = (() => {
  function el(id) {
    return document.getElementById(id);
  }

  function setStatus(dotId, textId, state, text) {
    const dot = el(dotId);
    dot.classList.remove("status-dot--ok", "status-dot--error", "status-dot--pending");
    dot.classList.add(`status-dot--${state}`);
    el(textId).textContent = text;
  }

  function renderProviders(health) {
    el("provider-generation").textContent = health.generation_provider;
    el("provider-embedding").textContent = health.embedding_provider;
    el("data-dir").textContent = health.data_dir;
  }

  function renderVersion(version) {
    el("api-version").textContent = `v${version.version}`;
  }

  function addDocToList(filename, status) {
    const list = el("doc-list");
    const empty = list.querySelector(".doc-list-empty");
    if (empty) empty.remove();

    const item = document.createElement("li");
    item.className = "doc-item";

    const name = document.createElement("span");
    name.className = "doc-item-name";
    name.textContent = filename;
    name.title = filename;

    const dot = document.createElement("span");
    dot.className = `doc-item-status status-dot--${status === "ready" ? "ok" : "error"}`;

    item.appendChild(name);
    item.appendChild(dot);
    list.prepend(item);
  }

  function toast(message, kind = "success") {
    const region = el("toast-region");
    const node = document.createElement("div");
    node.className = `toast toast--${kind}`;
    node.textContent = message;
    node.setAttribute("role", "status");
    region.appendChild(node);
    setTimeout(() => node.remove(), 4000);
  }

  function setAskLoading(isLoading) {
    const button = el("ask-button");
    button.disabled = isLoading;
    button.innerHTML = isLoading
      ? '<span class="spinner" aria-hidden="true"></span> Asking…'
      : "Ask";
  }

  function setIndexLoading(isLoading) {
    const button = el("index-button");
    button.disabled = isLoading;
    button.textContent = isLoading ? "Indexing…" : "Index document";
  }

  function renderAnswer(result) {
    const card = el("answer-card");
    card.hidden = false;

    const textEl = el("answer-text");
    const errorEl = el("answer-error");
    const citationsEl = el("citation-list");

    if (result.error) {
      textEl.hidden = true;
      citationsEl.hidden = true;
      errorEl.hidden = false;
      errorEl.textContent = `Generation error: ${result.error}`;
    } else {
      errorEl.hidden = true;
      textEl.hidden = false;
      textEl.textContent = result.answer;

      citationsEl.innerHTML = "";
      if (result.citations && result.citations.length > 0) {
        citationsEl.hidden = false;
        for (const citation of result.citations) {
          const chip = document.createElement("span");
          chip.className = "citation-chip";
          chip.textContent = citation;
          citationsEl.appendChild(chip);
        }
      } else {
        citationsEl.hidden = true;
      }
    }

    const c = result.confidence;
    el("metric-overall").textContent = c.overall.toFixed(2);
    el("metric-retrieval").textContent = c.retrieval.toFixed(2);
    el("metric-citations").textContent = c.citations.toFixed(2);
    el("metric-coverage").textContent = c.coverage.toFixed(2);

    el("dev-panel-json").textContent = JSON.stringify(result, null, 2);
  }

  function hideAnswer() {
    el("answer-card").hidden = true;
  }

  return {
    setStatus,
    renderProviders,
    renderVersion,
    addDocToList,
    toast,
    setAskLoading,
    setIndexLoading,
    renderAnswer,
    hideAnswer,
  };
})();
