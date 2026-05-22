import streamlit as st
import requests
import time

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

# --- Display Chat ---
st.title("💙 Medical Appointment Assistant")

for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        st.markdown(f"<div class='user-bubble'>{msg['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bot-bubble'>{msg['content']}</div>", unsafe_allow_html=True)

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

# --- Input Box ---
query = st.chat_input("Type your concern or request...")

if query:
    # Show user message immediately
    st.session_state["messages"].append({"role": "user", "content": query})

    # Spinner while backend processes
    with st.spinner("🤔 Assistant is thinking..."):
        try:
            params = {"query": query}
            patient_info = st.session_state["context"].get("patient_info") or {}
            if patient_info.get("patient_id"):
                params["patient_id"] = patient_info["patient_id"]

            response = requests.post(
                "http://localhost:8000/rag",
                params=params
            )
            data = response.json()
            reply = data.get("reply", "⚠️ No reply")
            st.session_state["messages"].append({"role": "bot", "content": reply})
            st.session_state["context"] = data.get("context", {})
        except Exception as e:
            st.session_state["messages"].append({"role": "bot", "content": f"⚠️ Backend error: {e}"})
