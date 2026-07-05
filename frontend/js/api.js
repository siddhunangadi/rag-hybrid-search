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

  return {
    getHealth: () => request("/health"),
    getVersion: () => request("/version"),
    indexDocuments: (documents) =>
      request("/index", {
        method: "POST",
        body: JSON.stringify({ documents }),
      }),
    answer: (question, maxChunks, verify) =>
      request("/answer", {
        method: "POST",
        body: JSON.stringify({ question, max_chunks: maxChunks, verify }),
      }),
  };
})();
