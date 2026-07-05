/**
 * Orchestration: wires DOM events to Api calls and Ui rendering.
 * No fetch logic (api.js) or DOM rendering details (ui.js) live here.
 */
(function () {
  async function loadHealthAndVersion() {
    try {
      const health = await Api.getHealth();
      Ui.setStatus("health-dot", "health-text", "ok", "API reachable");
      Ui.renderProviders(health);
    } catch (err) {
      Ui.setStatus(
        "health-dot",
        "health-text",
        "error",
        "API unreachable (free-tier instances can take ~50s to wake up)"
      );
    }

    try {
      const version = await Api.getVersion();
      Ui.renderVersion(version);
    } catch (err) {
      // Version badge just stays blank; health status already reports the outage.
    }
  }

  async function handleIndexSubmit(event) {
    event.preventDefault();
    const filenameInput = document.getElementById("index-filename");
    const contentInput = document.getElementById("index-content");

    const filename = filenameInput.value.trim();
    const content = contentInput.value.trim();
    if (!content) return;

    Ui.setIndexLoading(true);
    try {
      const response = await Api.indexDocuments([{ filename, content }]);
      for (const result of response.results) {
        Ui.addDocToList(result.filename, result.status);
        if (result.status === "ready") {
          Ui.toast(`Indexed "${result.filename}"`, "success");
        } else {
          Ui.toast(`Failed to index "${result.filename}": ${result.error}`, "error");
        }
      }
      contentInput.value = "";
    } catch (err) {
      Ui.toast(`Request failed: ${err.message}`, "error");
    } finally {
      Ui.setIndexLoading(false);
    }
  }

  async function handleAskSubmit(event) {
    event.preventDefault();
    const questionInput = document.getElementById("question-input");
    const maxChunksInput = document.getElementById("max-chunks-input");
    const verifyInput = document.getElementById("verify-input");

    const question = questionInput.value.trim();
    if (!question) return;

    Ui.setAskLoading(true);
    Ui.hideAnswer();
    try {
      const result = await Api.answer(
        question,
        Number(maxChunksInput.value),
        verifyInput.checked
      );
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
    loadHealthAndVersion();
  });
})();
