import streamlit as st
import requests

st.set_page_config(page_title="Medical Appointment Assistant", page_icon="💙", layout="wide")

# --- Custom CSS ---
st.markdown("""
    <style>
    .user-bubble {
        background-color: #0078D4;
        color: white;
        padding: 12px;
        border-radius: 15px;
        margin: 8px;
        text-align: right;
        font-family: Arial, sans-serif;
    }
    .bot-bubble {
        background-color: #f1f1f1;
        color: #333;
        padding: 12px;
        border-radius: 15px;
        margin: 8px;
        text-align: left;
        font-family: Arial, sans-serif;
        white-space: pre-wrap;
    }
    </style>
""", unsafe_allow_html=True)

# --- Chat State ---
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "bot", "content": "👋 Hello Heena! I’m here to help you with appointments and care. How are you feeling today?"}
    ]
if "context" not in st.session_state:
    st.session_state["context"] = {}


def render_message(msg):
    if msg["role"] == "user":
        st.markdown(f"<div class='user-bubble'>{msg['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bot-bubble'>{msg['content']}</div>", unsafe_allow_html=True)

# --- Display Chat ---
st.title("💙 Medical Appointment Assistant")
chat_container = st.container()

# --- Input Box ---
query = st.chat_input("Type your concern or request...")

if query:
    st.session_state["messages"].append({"role": "user", "content": query})

with chat_container:
    for msg in st.session_state["messages"]:
        render_message(msg)

if query:
    with chat_container:
        try:
            params = {"query": query}
            patient_info = st.session_state["context"].get("patient_info") or {}
            if patient_info.get("patient_id"):
                params["patient_id"] = patient_info["patient_id"]

            with st.spinner("🤔 Assistant is thinking..."):
                response = requests.post(
                    "http://localhost:8000/rag",
                    params=params,
                    timeout=120
                )
                response.raise_for_status()

            data = response.json()
            reply = data.get("reply", "⚠️ No reply")
            st.session_state["context"] = data.get("context", {})
        except Exception as e:
            reply = f"⚠️ Backend error: {e}"

        bot_message = {"role": "bot", "content": reply}
        st.session_state["messages"].append(bot_message)
        render_message(bot_message)

# --- Sidebar Context ---
st.sidebar.title("📋 Context")
ctx = st.session_state["context"]
if ctx.get("patient_info"):
    st.sidebar.subheader("Patient Info")
    st.sidebar.json(ctx["patient_info"])
if ctx.get("doctor_info"):
    st.sidebar.subheader("Doctor Info")
    st.sidebar.json(ctx["doctor_info"])
if ctx.get("appointment"):
    st.sidebar.subheader("Appointment")
    st.sidebar.json(ctx["appointment"])
