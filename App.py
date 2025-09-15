import io
import numpy as np
import pandas as pd
import streamlit as st

# ========== Optional: OpenAI (only if you toggle AI mode) ==========
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# ---------------- Page setup ----------------
st.set_page_config(page_title="Chat over your Dataset", layout="wide")
st.title("💬 Chat over your CSV (RAG)")

# ---------------- Sidebar: data & settings ----------------
with st.sidebar:
    st.header("📁 Data & Settings")

    uploaded = st.file_uploader("Upload a CSV", type=["csv"])
    use_ai = st.checkbox("Use AI to write the answer (OpenAI)", value=False)
    top_k = st.slider("Top matches (k)", min_value=1, max_value=10, value=5, help="How many rows to retrieve as context")
    chunk_columns = st.text_input(
        "Columns to include (comma-separated, blank = all)",
        value="",
        help="Example: name, description, price"
    )

    st.markdown("---")

    # Quick key input for testing (safer: st.secrets in production)
    api_key = st.text_input("OpenAI API Key", type="password") if use_ai else None
    model_name = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"], index=0) if use_ai else None
    embed_model = "text-embedding-3-small"  # good, fast, cheap

# ---------------- State ----------------
if "df" not in st.session_state:
    st.session_state.df = None
if "index" not in st.session_state:
    # list of dicts: {"text": str, "row_idx": int, "vector": np.array(shape=(1536,))}
    st.session_state.index = []
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------- Helpers ----------------
def cosine_sim(a, b):
    # a: (d,), b: (n,d)
    a_norm = a / (np.linalg.norm(a) + 1e-10)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return np.dot(b_norm, a_norm)

def concat_row_text(row: pd.Series):
    """Turn a row into a friendly snippet like 'col1: val1 | col2: val2 | ...'."""
    parts = []
    for col, val in row.items():
        parts.append(f"{col}: {val}")
    return " | ".join(parts)

def build_snippets(df: pd.DataFrame, include_cols: list[str] | None):
    if include_cols:
        cols = [c.strip() for c in include_cols if c.strip() in df.columns]
        if not cols:
            cols = list(df.columns)
    else:
        cols = list(df.columns)

    texts = []
    for i, row in df[cols].iterrows():
        texts.append(concat_row_text(row))
    return texts

def ensure_openai_client():
    if not use_ai:
        return None
    if OpenAI is None:
        st.error("OpenAI package not installed. Run: `pip install openai`")
        return None
    if not api_key:
        st.error("Please provide your OpenAI API key in the sidebar.")
        return None
    return OpenAI(api_key=api_key)

def embed_texts(client, texts: list[str]) -> np.ndarray:
    # returns (n, d) array
    resp = client.embeddings.create(model=embed_model, input=texts)
    vecs = [np.array(item.embedding, dtype=np.float32) for item in resp.data]
    return np.vstack(vecs)

def embed_one(client, text: str) -> np.ndarray:
    resp = client.embeddings.create(model=embed_model, input=[text])
    return np.array(resp.data[0].embedding, dtype=np.float32)

def make_grounded_prompt(question: str, snippets: list[str]) -> str:
    joined = "\n\n".join(f"- {s}" for s in snippets)
    system_rules = (
        "You are a helpful assistant that answers ONLY using the provided dataset rows.\n"
        "If the answer is not present in the rows, reply: 'I don’t know from the dataset.'\n"
        "Be concise and include specifics. If useful, reference row numbers."
    )
    user_block = (
        f"Question:\n{question}\n\n"
        f"Relevant dataset rows (unordered):\n{joined}\n\n"
        "Answer using only the rows above."
    )
    # We'll send as system + user messages in the chat call
    return system_rules, user_block

# ---------------- Load / index data ----------------
if uploaded is not None:
    try:
        raw = uploaded.read()
        st.session_state.df = pd.read_csv(io.BytesIO(raw))
        st.success(f"Loaded CSV with shape {st.session_state.df.shape}")
        st.dataframe(st.session_state.df.head(20), use_container_width=True)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")

    # Build snippets
    cols = [c.strip() for c in chunk_columns.split(",")] if chunk_columns else None
    snippets = build_snippets(st.session_state.df, cols)

    # Build embeddings index (requires OpenAI even if you won’t generate answers)
    if use_ai:
        client = ensure_openai_client()
        if client:
            with st.spinner("Embedding your dataset (first time only)..."):
                vecs = embed_texts(client, snippets)  # (n,d)
                st.session_state.index = [
                    {"text": snippets[i], "row_idx": int(i), "vector": vecs[i]}
                    for i in range(len(snippets))
                ]
            st.success(f"Indexed {len(st.session_state.index)} rows.")
    else:
        # No AI: still create a “text-only index” for simple keyword matching fallback
        st.session_state.index = [{"text": t, "row_idx": i, "vector": None} for i, t in enumerate(snippets)]

else:
    st.info("Upload a CSV in the sidebar to begin.")

# ---------------- Chat area ----------------
if st.session_state.df is not None and st.session_state.index:
    st.markdown("### Ask a question about your data")
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    question = st.chat_input("Ask something that your CSV can answer…")
    if question:
        # 1) Show / store user message
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # 2) Retrieve
        with st.spinner("Searching your dataset…"):
            if use_ai:
                client = ensure_openai_client()
                if client is None:
                    st.stop()

                # Embed question
                q_vec = embed_one(client, question)

                # Collect matrix of row vectors
                mat = np.vstack([item["vector"] for item in st.session_state.index])  # (n,d)
                sims = cosine_sim(q_vec, mat)  # (n,)
                top_idx = np.argsort(-sims)[:top_k]
            else:
                # Simple keyword fallback: score by number of shared keywords
                q_tokens = set(question.lower().split())
                scores = []
                for i, item in enumerate(st.session_state.index):
                    t_tokens = set(str(item["text"]).lower().split())
                    scores.append((i, len(q_tokens & t_tokens)))
                top_idx = [i for i, _ in sorted(scores, key=lambda x: -x[1])[:top_k]]

            top_items = [st.session_state.index[i] for i in top_idx]
            top_snippets = [it["text"] for it in top_items]

        # 3) Generate (AI) or show (non-AI)
        if use_ai:
            client = ensure_openai_client()
            sys, usr = make_grounded_prompt(question, top_snippets)
            with st.chat_message("assistant"):
                with st.spinner("Writing answer from your data…"):
                    resp = client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": sys},
                            {"role": "user", "content": usr},
                        ],
                        temperature=0.2,
                    )
                    answer = resp.choices[0].message.content
                    st.markdown(answer)

            # Save assistant message
            st.session_state.messages.append({"role": "assistant", "content": answer})
        else:
            # No AI: just show the top matching rows/snippets
            with st.chat_message("assistant"):
                st.markdown("**Top matching rows from your CSV:**")
                # Show a compact table of matched rows
                match_df = st.session_state.df.iloc[[it["row_idx"] for it in top_items]]
                st.dataframe(match_df, use_container_width=True)

            st.session_state.messages.append({"role": "assistant", "content": "Displayed top matching rows above."})

else:
    st.caption("Once a CSV is uploaded and indexed, you can start chatting here.")
