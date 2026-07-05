"""Streamlit frontend for the rag-hybrid-search API.

Thin client only — all retrieval/generation/verification logic lives in the
FastAPI service (see api/). This file just calls that HTTP API and renders
the response. Run with:

    uv sync --extra ui
    uv run streamlit run ui/app.py
"""

import requests
import streamlit as st

_DEFAULT_API_URL = "https://rag-hybrid-search.onrender.com"
_REQUEST_TIMEOUT_SECONDS = 60

st.set_page_config(page_title="rag-hybrid-search", page_icon="🔎", layout="wide")


def _api_url() -> str:
    """Return the configured API base URL, stripped of a trailing slash."""
    return st.session_state.get("api_url", _DEFAULT_API_URL).rstrip("/")


def _get_health(api_url: str) -> dict | None:
    try:
        response = requests.get(f"{api_url}/health", timeout=_REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        st.session_state["health_error"] = str(exc)
        return None


with st.sidebar:
    st.header("Connection")
    st.text_input("API base URL", value=_DEFAULT_API_URL, key="api_url")
    api_url = _api_url()

    health = _get_health(api_url)
    if health:
        st.success("API reachable")
        st.caption(f"Generation: `{health['generation_provider']}`")
        st.caption(f"Embeddings: `{health['embedding_provider']}`")
    else:
        st.error(f"API unreachable: {st.session_state.get('health_error', 'unknown error')}")
        st.caption(
            "Free-tier Render instances spin down when idle — the first "
            "request after inactivity can take ~50s to wake it up."
        )

st.title("🔎 rag-hybrid-search")
st.caption("Hybrid retrieval + citation-verified, confidence-scored generation.")

tab_index, tab_answer = st.tabs(["📄 Index documents", "💬 Ask a question"])

with tab_index:
    st.subheader("Add documents to the index")
    filename = st.text_input("Filename", value="notes.md", key="index_filename")
    content = st.text_area("Content", height=200, key="index_content")

    if st.button("Index document", type="primary", disabled=not content.strip()):
        try:
            response = requests.post(
                f"{api_url}/index",
                json={"documents": [{"filename": filename, "content": content}]},
                timeout=_REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            for result in response.json()["results"]:
                if result["status"] == "ready":
                    st.success(f"Indexed `{result['filename']}`")
                else:
                    st.error(f"Failed to index `{result['filename']}`: {result['error']}")
        except requests.RequestException as exc:
            st.error(f"Request failed: {exc}")

with tab_answer:
    st.subheader("Ask a grounded question")
    question = st.text_input("Question", key="question")
    col_max_chunks, col_verify = st.columns([1, 1])
    max_chunks = col_max_chunks.slider("Max chunks", min_value=1, max_value=10, value=5)
    verify = col_verify.checkbox("Verify citations", value=True)

    if st.button("Ask", type="primary", disabled=not question.strip()):
        try:
            response = requests.post(
                f"{api_url}/answer",
                json={"question": question, "max_chunks": max_chunks, "verify": verify},
                timeout=_REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            result = response.json()

            if result["error"]:
                st.error(f"Generation error: {result['error']}")
            else:
                st.markdown(f"### {result['answer']}")
                if result["citations"]:
                    st.caption("Citations: " + ", ".join(f"`{c}`" for c in result["citations"]))

            confidence = result["confidence"]
            metric_cols = st.columns(4)
            metric_cols[0].metric("Overall", f"{confidence['overall']:.2f}")
            metric_cols[1].metric("Retrieval", f"{confidence['retrieval']:.2f}")
            metric_cols[2].metric("Citations", f"{confidence['citations']:.2f}")
            metric_cols[3].metric("Coverage", f"{confidence['coverage']:.2f}")

            verification = result["verification"]
            if verification["claim_results"]:
                st.subheader("Claim verification")
                st.dataframe(
                    [
                        {
                            "claim": c["claim"]["text"],
                            "citation_ids": ", ".join(c["claim"]["citation_ids"]),
                            "quote_match_score": round(c["quote_match_score"], 2),
                            "passed": c["passed"],
                        }
                        for c in verification["claim_results"]
                    ],
                    width="stretch",
                )
        except requests.RequestException as exc:
            st.error(f"Request failed: {exc}")
