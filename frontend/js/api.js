/**
 * Thin HTTP client for the rag-hybrid-search FastAPI backend.
 *
 * Every function here maps 1:1 to one API endpoint. No retry/caching/business
 * logic lives here — just fetch + JSON parsing + error normalization.
 */
const Api = (() => {
  async function request(path, options = {}) {
    const response = await fetch(`${CONFIG.API_BASE_URL}${path}`, {
      headers: { "Content-Type": "application/json" },
      ...options,
    });

    let body = null;
    try {
      body = await response.json();
    } catch (_err) {
      // Non-JSON response (e.g. a 502 from a cold-starting free-tier host).
    }

    if (!response.ok) {
      const detail = body && body.detail ? body.detail : response.statusText;
      throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
    }

    return body;
  }

  async function _uploadForm(path, fileList, documentType) {
    const formData = new FormData();
    for (const file of fileList) {
      formData.append("files", file);
    }
    formData.append("document_type", documentType || "general");
    // No Content-Type header here: the browser sets the multipart boundary itself.
    const response = await fetch(`${CONFIG.API_BASE_URL}${path}`, {
      method: "POST",
      body: formData,
    });

    let body = null;
    try {
      body = await response.json();
    } catch (_err) {
      // Non-JSON response (e.g. a 502 from a cold-starting free-tier host).
    }

    if (!response.ok) {
      const detail = body && body.detail ? body.detail : response.statusText;
      throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
    }

    return body;
  }

  const uploadFiles = (fileList, documentType) => _uploadForm("/upload", fileList, documentType);
  const uploadFilesAsync = (fileList, documentType) =>
    _uploadForm("/upload/async", fileList, documentType);

  /**
   * Consume the /answer/stream SSE response, calling onDelta(text) for each
   * "delta" frame and returning the parsed RagAnswer from the "final" frame.
   */
  async function streamAnswer(question, verify, onDelta) {
    const response = await fetch(`${CONFIG.API_BASE_URL}/answer/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, verify }),
    });
    if (!response.ok || !response.body) {
      let detail = response.statusText;
      try {
        detail = (await response.json()).detail || detail;
      } catch (_err) {
        // keep statusText
      }
      throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let finalAnswer = null;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let boundary;
      while ((boundary = buffer.indexOf("\n\n")) !== -1) {
        const frame = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);

        const eventLine = frame.split("\n").find((l) => l.startsWith("event: "));
        const dataLine = frame.split("\n").find((l) => l.startsWith("data: "));
        if (!eventLine || !dataLine) continue;

        const eventType = eventLine.slice("event: ".length);
        const data = JSON.parse(dataLine.slice("data: ".length));

        if (eventType === "delta") {
          onDelta(data.text);
        } else if (eventType === "final") {
          finalAnswer = data;
        }
      }
    }

    return finalAnswer;
  }

  return {
    getHealth: () => request("/health"),
    getVersion: () => request("/version"),
    indexDocuments: (documents) =>
      request("/index", {
        method: "POST",
        body: JSON.stringify({ documents }),
      }),
    uploadFiles,
    uploadFilesAsync,
    getJob: (jobId) => request(`/jobs/${jobId}`),
    listDocuments: () => request("/documents"),
    deleteDocument: (documentId) =>
      request(`/documents/${documentId}`, { method: "DELETE" }),
    answer: (question, verify) =>
      request("/answer", {
        method: "POST",
        body: JSON.stringify({ question, verify }),
      }),
    streamAnswer,
  };
})();
