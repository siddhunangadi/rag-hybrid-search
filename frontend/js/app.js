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

  function reportIndexResults(results) {
    for (const result of results) {
      Ui.addDocToList(result.filename, result.status);
      if (result.status === "ready") {
        Ui.toast(`Indexed "${result.filename}"`, "success");
      } else {
        Ui.toast(`Failed to index "${result.filename}": ${result.error}`, "error");
      }
    }
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
        const response = await Api.uploadFiles(fileInput.files);
        reportIndexResults(response.results);
        fileInput.value = "";
      } else {
        const filename = filenameInput.value.trim();
        const response = await Api.indexDocuments([{ filename, content }]);
        reportIndexResults(response.results);
        contentInput.value = "";
      }
    } catch (err) {
      Ui.toast(`Request failed: ${err.message}`, "error");
    } finally {
      Ui.setIndexLoading(false);
    }
  }

  async function handleAskSubmit(event) {
    event.preventDefault();
    const questionInput = document.getElementById("question-input");
    const verifyInput = document.getElementById("verify-input");

    const question = questionInput.value.trim();
    if (!question) return;

    Ui.setAskLoading(true);
    Ui.hideAnswer();
    try {
      const result = await Api.answer(question, verifyInput.checked);
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
