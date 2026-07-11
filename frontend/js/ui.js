/**
 * DOM rendering helpers. Pure presentation — takes data, updates the DOM.
 * No fetch calls live here (see api.js); no orchestration (see app.js).
 */
const Ui = (() => {
  function el(id) {
    return document.getElementById(id);
  }

  function plainConfidenceMessage(confidence) {
    const overall = confidence.overall;
    if (overall >= 0.75) {
      return { text: "High confidence — this answer is well-supported by the source document.", kind: "high" };
    }
    if (overall >= 0.4) {
      return { text: "Partially verified — parts of this answer could not be fully confirmed.", kind: "medium" };
    }
    return { text: "Low confidence — please review this answer against the source document.", kind: "low" };
  }

  function setStatus(dotId, textId, state, text) {
    const dot = el(dotId);
    dot.classList.remove("status-dot--ok", "status-dot--error", "status-dot--pending");
    dot.classList.add(`status-dot--${state}`);
    el(textId).textContent = text;
  }

  function renderDocuments(documentsResponse) {
    const list = el("doc-list");
    list.innerHTML = "";
    const documents = documentsResponse.documents || [];

    if (documents.length === 0) {
      const empty = document.createElement("li");
      empty.className = "doc-list-empty";
      empty.textContent = "No documents indexed yet.";
      list.appendChild(empty);
      return;
    }

    for (const doc of documents) {
      const item = document.createElement("li");
      item.className = "doc-item";

      const name = document.createElement("span");
      name.className = "doc-item-name";
      name.textContent = `${doc.filename} (${doc.chunk_count})`;
      name.title = `${doc.filename} — ${doc.chunk_count} chunks — ${doc.document_id}`;

      const removeBtn = document.createElement("button");
      removeBtn.type = "button";
      removeBtn.className = "doc-item-delete";
      removeBtn.setAttribute("aria-label", `Delete ${doc.filename}`);
      removeBtn.textContent = "✕";
      removeBtn.dataset.documentId = doc.document_id;

      item.appendChild(name);
      item.appendChild(removeBtn);
      list.appendChild(item);
    }
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
      const structured = result.structured_citations || [];
      if (structured.length > 0) {
        citationsEl.hidden = false;
        const seenCount = new Map();
        for (const citation of structured) {
          const chip = document.createElement("span");
          chip.className = "citation-chip";
          const baseLabel = citation.page
            ? `${citation.document_title} · p.${citation.page}`
            : citation.document_title;
          const count = (seenCount.get(baseLabel) || 0) + 1;
          seenCount.set(baseLabel, count);
          chip.textContent = count === 1 ? baseLabel : `${baseLabel} · excerpt ${count}`;
          chip.title = citation.display;
          citationsEl.appendChild(chip);
        }
      } else if (result.citations && result.citations.length > 0) {
        // Fallback for older responses without structured_citations.
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

    const label = el("confidence-label");
    if (result.error) {
      label.hidden = true;
    } else {
      const { text, kind } = plainConfidenceMessage(result.confidence);
      label.hidden = false;
      label.textContent = text;
      label.className = `confidence-label confidence-label--${kind}`;
    }

    el("dev-panel-json").textContent = JSON.stringify(result, null, 2);
  }

  function hideAnswer() {
    el("answer-card").hidden = true;
  }

  function beginStreamingAnswer() {
    const card = el("answer-card");
    card.hidden = false;
    el("answer-error").hidden = true;
    el("citation-list").hidden = true;
    el("citation-list").innerHTML = "";
    const textEl = el("answer-text");
    textEl.hidden = false;
    textEl.textContent = "";
    el("confidence-label").hidden = true;
    el("dev-panel-json").textContent = "";
  }

  function appendStreamingDelta(text) {
    el("answer-text").textContent += text;
  }

  function setJobStatus(text, kind) {
    const box = el("job-status");
    box.hidden = !text;
    box.textContent = text || "";
    box.className = `job-status${kind ? ` job-status--${kind}` : ""}`;
  }

  return {
    setStatus,
    renderDocuments,
    addDocToList,
    toast,
    setAskLoading,
    setIndexLoading,
    renderAnswer,
    hideAnswer,
    beginStreamingAnswer,
    appendStreamingDelta,
    setJobStatus,
  };
})();
