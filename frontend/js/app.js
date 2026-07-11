/**
 * Orchestration: wires DOM events to Api calls and Ui rendering.
 * No fetch logic (api.js) or DOM rendering details (ui.js) live here.
 */
(function () {
  async function loadHealth() {
    try {
      await Api.getHealth();
      Ui.setStatus("health-dot", "health-text", "ok", "API reachable");
    } catch (err) {
      Ui.setStatus(
        "health-dot",
        "health-text",
        "error",
        "API unreachable (free-tier instances can take ~50s to wake up)"
      );
    }
  }

  async function refreshDocuments() {
    try {
      const documents = await Api.listDocuments();
      Ui.renderDocuments(documents);
    } catch (err) {
      // Sidebar list just stays as-is; health status already reports outages.
    }
  }

  function reportIndexResults(results) {
    for (const result of results) {
      if (result.status === "ready") {
        Ui.toast(`Indexed "${result.filename}"`, "success");
      } else {
        Ui.toast(`Failed to index "${result.filename}": ${result.error}`, "error");
      }
    }
    refreshDocuments();
  }

  async function handleIndexSubmit(event) {
    event.preventDefault();
    const filenameInput = document.getElementById("index-filename");
    const contentInput = document.getElementById("index-content");
    const fileInput = document.getElementById("index-file");

    const hasFiles = fileInput.files && fileInput.files.length > 0;
    const content = contentInput.value.trim();
    if (!hasFiles && !content) return;

    Ui.setIndexLoading(true);
    try {
      if (hasFiles) {
        const response = await Api.uploadFiles(fileInput.files, "general");
        reportIndexResults(response.results);
        fileInput.value = "";
      } else {
        const filename = filenameInput.value.trim();
        const response = await Api.indexDocuments([{ filename, content, document_type: "general" }]);
        reportIndexResults(response.results);
        contentInput.value = "";
      }
    } catch (err) {
      Ui.toast(`Request failed: ${err.message}`, "error");
    } finally {
      Ui.setIndexLoading(false);
    }
  }

  async function handleDocListClick(event) {
    const button = event.target.closest(".doc-item-delete");
    if (!button) return;
    const documentId = button.dataset.documentId;
    try {
      await Api.deleteDocument(documentId);
      Ui.toast("Document deleted", "success");
      refreshDocuments();
    } catch (err) {
      Ui.toast(`Delete failed: ${err.message}`, "error");
    }
  }

  async function handleAskSubmit(event) {
    event.preventDefault();
    const questionInput = document.getElementById("question-input");

    const question = questionInput.value.trim();
    if (!question) return;

    Ui.setAskLoading(true);
    Ui.hideAnswer();
    try {
      Ui.beginStreamingAnswer();
      const result = await Api.streamAnswer(question, true, Ui.appendStreamingDelta);
      Ui.renderAnswer(result);
    } catch (err) {
      Ui.toast(`Request failed: ${err.message}`, "error");
    } finally {
      Ui.setAskLoading(false);
    }
  }

  document.addEventListener("DOMContentLoaded", () => {
    document.getElementById("index-form").addEventListener("submit", handleIndexSubmit);
    document.getElementById("ask-form").addEventListener("submit", handleAskSubmit);
    document.getElementById("doc-list").addEventListener("click", handleDocListClick);
    loadHealth();
    refreshDocuments();
  });
})();
