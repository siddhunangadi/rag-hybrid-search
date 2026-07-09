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

  async function pollJob(jobId) {
    Ui.setJobStatus(`Processing… (job ${jobId.slice(0, 8)})`, "processing");
    for (;;) {
      await new Promise((resolve) => setTimeout(resolve, 1000));
      const job = await Api.getJob(jobId);
      if (job.status === "processing") continue;
      if (job.status === "ready") {
        Ui.setJobStatus("Background ingestion complete.", "ready");
        reportIndexResults(job.result.results);
      } else {
        Ui.setJobStatus(`Background ingestion failed: ${job.error}`, "failed");
      }
      setTimeout(() => Ui.setJobStatus(""), 5000);
      return;
    }
  }

  async function handleIndexSubmit(event) {
    event.preventDefault();
    const filenameInput = document.getElementById("index-filename");
    const contentInput = document.getElementById("index-content");
    const fileInput = document.getElementById("index-file");
    const documentTypeInput = document.getElementById("index-document-type");
    const asyncInput = document.getElementById("index-async-input");

    const hasFiles = fileInput.files && fileInput.files.length > 0;
    const content = contentInput.value.trim();
    if (!hasFiles && !content) return;

    Ui.setIndexLoading(true);
    try {
      if (hasFiles) {
        if (asyncInput.checked) {
          const accepted = await Api.uploadFilesAsync(fileInput.files, documentTypeInput.value);
          pollJob(accepted.job_id);
        } else {
          const response = await Api.uploadFiles(fileInput.files, documentTypeInput.value);
          reportIndexResults(response.results);
        }
        fileInput.value = "";
      } else {
        const filename = filenameInput.value.trim();
        const response = await Api.indexDocuments([
          { filename, content, document_type: documentTypeInput.value },
        ]);
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
    const verifyInput = document.getElementById("verify-input");
    const streamInput = document.getElementById("stream-input");

    const question = questionInput.value.trim();
    if (!question) return;

    Ui.setAskLoading(true);
    Ui.hideAnswer();
    try {
      if (streamInput.checked) {
        Ui.beginStreamingAnswer();
        const result = await Api.streamAnswer(question, verifyInput.checked, Ui.appendStreamingDelta);
        Ui.renderAnswer(result);
      } else {
        const result = await Api.answer(question, verifyInput.checked);
        Ui.renderAnswer(result);
      }
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
    loadHealthAndVersion();
    refreshDocuments();
  });
})();
