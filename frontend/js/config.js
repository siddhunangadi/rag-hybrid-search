/**
 * Runtime configuration for the frontend.
 *
 * When the frontend is served BY the FastAPI app itself (see api/main.py
 * static mount), same-origin relative paths work and API_BASE_URL is "".
 * When opened as a standalone static file (e.g. a separate static host),
 * fall back to the public API URL.
 */
const CONFIG = {
  API_BASE_URL: window.location.protocol === "file:"
    ? "https://rag-hybrid-search.onrender.com"
    : "",
};
