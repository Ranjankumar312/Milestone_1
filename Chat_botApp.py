import streamlit as st
import pandas as pd
from openai import OpenAI

# ------------------- CONFIG -------------------
st.set_page_config(page_title="Chatbot with Dataset", layout="wide")
client = OpenAI(api_key="")  # 👈 put your API key here
DATA_PATH = "bengaluru_house_prices.csv"  # 👈 dataset file path

# ------------------- SESSION STATE -------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "df" not in st.session_state:
    try:
        st.session_state.df = pd.read_csv(DATA_PATH)
        st.success(f"✅ Loaded dataset `{DATA_PATH}` with shape {st.session_state.df.shape}")
    except Exception as e:
        st.error(f"❌ Failed to load dataset: {e}")
        st.stop()

# ------------------- SIDEBAR -------------------
with st.sidebar:
    st.markdown("### ⚡ Chat_bot")

    if st.button("📝 New chat"):
        if st.session_state.messages:
            st.session_state.chat_history.append(st.session_state.messages)
        st.session_state.messages = []
        st.experimental_rerun()

    st.button("🔍 Search chats")
    st.button("⚙️ Settings ")

    st.markdown("---")
    st.subheader("Chats")
    if st.session_state.chat_history:
        for i, chat in enumerate(st.session_state.chat_history):
            if st.button(f"💬 Chat {i+1}", key=f"history_{i}"):
                st.session_state.messages = chat
                st.experimental_rerun()
    else:
        st.caption("No chats yet...")

# ------------------- MAIN CHAT WINDOW -------------------
st.title("💬 Chatbot (with Dataset)")

# Display previous messages (user → right, assistant → left)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ------------------- CHAT INPUT -------------------
if prompt := st.chat_input("Type your question..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # ------------------- BOT RESPONSE -------------------
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Take a small dataset context
            df_sample = st.session_state.df.head(5).to_string()

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": f"You are a helpful assistant. Here is some dataset context:\n{df_sample}"},
                    *st.session_state.messages,
                ]
            )

            reply = response.choices[0].message.content
            st.markdown(reply)

    # Save bot response
    st.session_state.messages.append({"role": "assistant", "content": reply})
