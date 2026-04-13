"""
AI Interviewer — Streamlit Frontend
Production-grade interview interface with voice, text, workspace, and live scoring.
"""

import os
import streamlit as st
import requests
import json
import base64
import time
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
API = {
    "start": f"{BACKEND_URL}/api/start-interview",
    "start_json": f"{BACKEND_URL}/api/start-interview-json",
    "respond": f"{BACKEND_URL}/api/submit-response",
    "submit_code": f"{BACKEND_URL}/api/submit-code",
    "run_code": f"{BACKEND_URL}/api/run-code",
    "end": f"{BACKEND_URL}/api/end-interview",
    "state": f"{BACKEND_URL}/api/state",
    "tts": f"{BACKEND_URL}/api/tts",
    "health": f"{BACKEND_URL}/api/health",
}

# ────────────────────────────────────────────
# Page Config & Custom CSS
# ────────────────────────────────────────────
st.set_page_config(
    page_title="AI Interviewer | Eightfold-Style Adaptive Interview",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Global ── */
    :root {
        --navy: #0a0f1c;
        --navy-light: #111827;
        --navy-mid: #1e293b;
        --accent: #3b82f6;
        --accent-glow: rgba(59, 130, 246, 0.3);
        --green: #10b981;
        --amber: #f59e0b;
        --red: #ef4444;
        --text: #e2e8f0;
        --text-dim: #94a3b8;
        --border: #334155;
    }

    .stApp {
        font-family: 'Inter', sans-serif !important;
    }

    .main .block-container {
        padding-top: 1.5rem;
        max-width: 1400px;
    }

    /* ── Header ── */
    .hero-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1.5rem 2rem;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }

    .hero-header::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--accent), var(--green), var(--accent));
    }

    .hero-header h1 {
        margin: 0;
        font-size: 1.6rem;
        font-weight: 700;
        background: linear-gradient(135deg, #60a5fa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-header p {
        margin: 0.25rem 0 0 0;
        color: var(--text-dim);
        font-size: 0.85rem;
    }

    /* ── Chat Transcript ── */
    .chat-container {
        background: var(--navy-light);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1rem;
        max-height: 520px;
        overflow-y: auto;
        margin-bottom: 1rem;
    }

    .msg-bubble {
        padding: 0.75rem 1rem;
        border-radius: 12px;
        margin-bottom: 0.75rem;
        max-width: 85%;
        line-height: 1.5;
        font-size: 0.9rem;
        animation: fadeSlideIn 0.3s ease-out;
    }

    @keyframes fadeSlideIn {
        from { opacity: 0; transform: translateY(8px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    .msg-interviewer {
        background: linear-gradient(135deg, #1e3a5f, #1e293b);
        border: 1px solid #2563eb33;
        color: var(--text);
        margin-right: auto;
        border-bottom-left-radius: 4px;
    }

    .msg-interviewer::before {
        content: '🎯 Interviewer';
        display: block;
        font-size: 0.7rem;
        font-weight: 600;
        color: var(--accent);
        margin-bottom: 0.35rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .msg-candidate {
        background: linear-gradient(135deg, #064e3b, #1e293b);
        border: 1px solid #10b98133;
        color: var(--text);
        margin-left: auto;
        border-bottom-right-radius: 4px;
    }

    .msg-candidate::before {
        content: '👤 You';
        display: block;
        font-size: 0.7rem;
        font-weight: 600;
        color: var(--green);
        margin-bottom: 0.35rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* ── Score bars ── */
    .score-row {
        display: flex;
        align-items: center;
        margin-bottom: 0.4rem;
        font-size: 0.8rem;
    }

    .score-label {
        width: 90px;
        color: var(--text-dim);
        font-weight: 500;
    }

    .score-bar-bg {
        flex: 1;
        height: 8px;
        background: var(--navy);
        border-radius: 4px;
        overflow: hidden;
        margin: 0 8px;
    }

    .score-bar-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .score-value {
        width: 40px;
        text-align: right;
        font-weight: 600;
        font-size: 0.75rem;
    }

    /* ── Status badges ── */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .badge-conversation {
        background: rgba(59, 130, 246, 0.15);
        color: var(--accent);
        border: 1px solid rgba(59, 130, 246, 0.3);
    }

    .badge-workspace {
        background: rgba(245, 158, 11, 0.15);
        color: var(--amber);
        border: 1px solid rgba(245, 158, 11, 0.3);
    }

    .badge-complete {
        background: rgba(16, 185, 129, 0.15);
        color: var(--green);
        border: 1px solid rgba(16, 185, 129, 0.3);
    }

    /* ── Workspace ── */
    .workspace-panel {
        background: var(--navy-light);
        border: 1px solid var(--amber);
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }

    .workspace-panel h3 {
        color: var(--amber);
        margin-top: 0;
    }

    /* ── Report card ── */
    .report-card {
        background: linear-gradient(135deg, #0f172a, #1e293b);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 2rem;
    }

    .report-card h2 {
        background: linear-gradient(135deg, #60a5fa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .rec-badge {
        display: inline-block;
        padding: 0.5rem 1.5rem;
        border-radius: 25px;
        font-weight: 700;
        font-size: 1rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .rec-hire { background: rgba(16, 185, 129, 0.2); color: #34d399; border: 2px solid #10b981; }
    .rec-strong { background: rgba(16, 185, 129, 0.3); color: #6ee7b7; border: 2px solid #34d399; }
    .rec-borderline { background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 2px solid #f59e0b; }
    .rec-nohire { background: rgba(239, 68, 68, 0.2); color: #f87171; border: 2px solid #ef4444; }

    /* ── Progress ring ── */
    .progress-ring {
        text-align: center;
        margin: 1rem 0;
    }

    /* ── Sidebar tweaks ── */
    [data-testid="stSidebar"] {
        background: var(--navy-light);
    }

    /* ── Fix text area for code ── */
    .stTextArea textarea {
        font-family: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace !important;
        font-size: 0.85rem !important;
        line-height: 1.5 !important;
    }
</style>
""", unsafe_allow_html=True)


# ────────────────────────────────────────────
# Session state initialization
# ────────────────────────────────────────────
def init_session():
    defaults = {
        "session_id": None,
        "interview_state": None,
        "interview_started": False,
        "voice_mode": False,
        "voice_started": False,
        "last_voice_event": None,
        "last_state_poll": 0.0,
        "audio_queue": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_session()


# ────────────────────────────────────────────
# Helper functions
# ────────────────────────────────────────────
def call_api(endpoint: str, method: str = "POST", data: dict = None, files: dict = None, form_data: dict = None):
    """Make API call to backend."""
    try:
        if method == "POST":
            if files or form_data:
                resp = requests.post(endpoint, data=form_data, files=files, timeout=120)
            else:
                resp = requests.post(endpoint, json=data, timeout=120)
        else:
            resp = requests.get(endpoint, timeout=30)

        if resp.status_code == 200:
            return resp.json()
        else:
            st.error(f"API Error {resp.status_code}: {resp.text}")
            return None
    except requests.ConnectionError:
        st.error(f"❌ Cannot connect to backend at {BACKEND_URL}. Make sure the FastAPI server is running.")
        return None
    except Exception as e:
        st.error(f"❌ API Error: {str(e)}")
        return None


def maybe_poll_state():
    if not st.session_state.interview_started:
        return
    if not st.session_state.session_id:
        return
    now = time.time()
    last_poll = st.session_state.get("last_state_poll", 0.0)
    if (now - last_poll) < 0.8:
        return
    st.session_state.last_state_poll = now
    result = call_api(f"{API['state']}/{st.session_state.session_id}", method="GET")
    if result and isinstance(result, dict) and result.get("state"):
        new_state = result["state"]
        prev_state = st.session_state.get("interview_state") or {}
        prev_sig = (
            prev_state.get("current_step"),
            (prev_state.get("cumulative_scores") or {}).get("num_evaluations"),
            prev_state.get("mode")
        )
        new_sig = (
            new_state.get("current_step"),
            (new_state.get("cumulative_scores") or {}).get("num_evaluations"),
            new_state.get("mode")
        )
        st.session_state.interview_state = new_state
        return new_sig != prev_sig
    return False


def get_score_color(score: float) -> str:
    if score >= 0.7:
        return "#10b981"
    elif score >= 0.4:
        return "#f59e0b"
    else:
        return "#ef4444"


def render_score_bar(label: str, score: float):
    color = get_score_color(score)
    pct = int(score * 100)
    st.markdown(f"""
    <div class="score-row">
        <span class="score-label">{label}</span>
        <div class="score-bar-bg">
            <div class="score-bar-fill" style="width: {pct}%; background: {color};"></div>
        </div>
        <span class="score-value" style="color: {color};">{pct}%</span>
    </div>
    """, unsafe_allow_html=True)


def render_mode_badge(mode: str):
    badge_map = {
        "conversation": ("💬 Conversation", "badge-conversation"),
        "workspace": ("🔧 Workspace", "badge-workspace"),
        "complete": ("✅ Complete", "badge-complete"),
    }
    label, cls = badge_map.get(mode, ("⏳ Initializing", "badge-conversation"))
    st.markdown(f'<span class="status-badge {cls}">{label}</span>', unsafe_allow_html=True)


# ────────────────────────────────────────────
# Header
# ────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <h1>🎯 AI Interviewer</h1>
    <p>Eightfold-Style Adaptive Technical Interview System</p>
</div>
""", unsafe_allow_html=True)


# ────────────────────────────────────────────
# Sidebar — Setup & Controls
# ────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎛️ Interview Setup")

    if not st.session_state.interview_started:
        role = st.text_input(
            "Target Role",
            value="Machine Learning Engineer",
            placeholder="e.g. Senior Backend Engineer",
            key="role_input"
        )

        resume_file = st.file_uploader(
            "📄 Upload Resume (Optional)",
            type=["pdf", "txt"],
            key="resume_upload"
        )

        st.markdown("---")

        if st.button("🚀 Start Interview", type="primary", use_container_width=True, key="start_btn"):
            with st.spinner("🧠 Analyzing profile & building interview plan..."):
                form_data = {"role": role}
                files = {}

                if resume_file:
                    files["resume"] = (resume_file.name, resume_file.getvalue(), resume_file.type)

                result = call_api(API["start"], form_data=form_data, files=files if files else None)

                if result:
                    st.session_state.session_id = result["session_id"]
                    st.session_state.interview_state = result["state"]
                    st.session_state.interview_started = True
                    st.session_state.voice_started = False
                    st.rerun()
    else:
        state = st.session_state.interview_state
        if state:
            mode = state.get("mode", "init")

            # Mode badge
            render_mode_badge(mode)

            # Progress
            plan = state.get("interview_plan", [])
            current = state.get("current_step", 0)
            total = len(plan)

            if total > 0:
                progress = min(current / total, 1.0)
                st.progress(progress, text=f"Step {min(current + 1, total)} of {total}")

            st.markdown("---")

            # Cumulative scores
            st.markdown("### 📊 Live Scores")
            scores = state.get("cumulative_scores", {})
            if scores.get("num_evaluations", 0) > 0:
                render_score_bar("Correctness", scores.get("correctness", 0))
                render_score_bar("Depth", scores.get("depth", 0))
                render_score_bar("Clarity", scores.get("clarity", 0))
                render_score_bar("Confidence", scores.get("confidence", 0))
                st.markdown("---")
                render_score_bar("Overall", scores.get("overall", 0))
            else:
                st.caption("Scores will appear after your first response")

            st.markdown("---")

            # Workspace evaluation summary
            workspace = state.get("workspace", {})
            ws_eval = workspace.get("evaluation", {}) if isinstance(workspace, dict) else {}
            if isinstance(ws_eval, dict) and any(k in ws_eval for k in ("correctness", "code_quality", "overall", "feedback")):
                st.markdown("### 🧪 Workspace Review")
                if "correctness" in ws_eval:
                    render_score_bar("Correctness", ws_eval.get("correctness", 0))
                if "code_quality" in ws_eval:
                    render_score_bar("Quality", ws_eval.get("code_quality", 0))
                if "overall" in ws_eval:
                    render_score_bar("Overall", ws_eval.get("overall", 0))
                if ws_eval.get("feedback"):
                    st.caption(ws_eval.get("feedback"))
                st.markdown("---")

            # Profile summary
            profile = state.get("profile", {})
            if profile.get("summary"):
                with st.expander("📋 Candidate Profile", expanded=False):
                    st.markdown(f"**Level**: {profile.get('experience_level', 'N/A')}")
                    st.markdown(f"**Summary**: {profile.get('summary', 'N/A')}")

                    skills = profile.get("skills", [])
                    if skills:
                        st.markdown("**Skills:**")
                        for s in skills[:8]:
                            name = s.get("name", "?")
                            level = s.get("level", "?")
                            conf = s.get("confidence", 0)
                            emoji = "🟢" if conf > 0.7 else "🟡" if conf > 0.4 else "🔴"
                            st.markdown(f"- {emoji} {name} ({level}, {int(conf*100)}%)")

            # Interview plan
            if plan:
                with st.expander("📝 Interview Plan", expanded=False):
                    for i, step in enumerate(plan):
                        marker = "✅" if i < current else "➡️" if i == current else "⬜"
                        diff_emoji = {"easy": "🟢", "medium": "🟡", "hard": "🔴"}.get(step.get("difficulty"), "⚪")
                        st.markdown(f"{marker} {diff_emoji} **{step.get('topic', '?')}** ({step.get('type', '?')})")

            st.markdown("---")

            # Controls
            col1, col2 = st.columns(2)
            with col1:
                if st.button("⏹️ End Interview", use_container_width=True, key="end_btn"):
                    with st.spinner("Generating final report..."):
                        result = call_api(API["end"], data={"session_id": st.session_state.session_id})
                        if result:
                            st.session_state.interview_state = result["state"]
                            st.rerun()
            with col2:
                if st.button("🔄 New Interview", use_container_width=True, key="new_btn"):
                    st.session_state.interview_started = False
                    st.session_state.session_id = None
                    st.session_state.interview_state = None
                    st.rerun()


# ────────────────────────────────────────────
# Main content area
# ────────────────────────────────────────────
if not st.session_state.interview_started:
    # Welcome screen
    st.markdown("""
    <div style="text-align: center; padding: 3rem 0;">
        <h2 style="color: var(--text); font-weight: 300;">Welcome to the AI Interviewer</h2>
        <p style="color: var(--text-dim); font-size: 1.1rem; max-width: 600px; margin: 0 auto;">
            Experience an adaptive technical interview powered by AI.
            Upload your resume, choose a role, and begin a personalized interview
            that adapts to your strengths and weaknesses in real-time.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Feature cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        #### 🧠 Adaptive Questions
        Questions adjust dynamically based on your performance. No static scripts.
        """)
    with col2:
        st.markdown("""
        #### 🔧 Live Coding
        Hands-on workspace tasks with real code execution and evaluation.
        """)
    with col3:
        st.markdown("""
        #### 📊 Detailed Report
        Comprehensive skill breakdown with hire/no-hire recommendation.
        """)

else:
    state = st.session_state.interview_state
    mode = state.get("mode", "conversation") if state else "conversation"

    # ── CONVERSATION MODE ──
    if mode in ["conversation", "planning", "init", "profiling", "workspace"]:
        left_col, right_col = st.columns([1, 1.2])
        with left_col:
            # Chat transcript
            st.markdown("### 💬 Interview Transcript")

            history = state.get("conversation_history", []) if state else []

            # Render chat messages
            if not st.session_state.voice_started:
                st.info("Click 'Start Voice Interview' to begin. The interviewer will start after voice is enabled.")
            else:
                chat_html = '<div class="chat-container">'
                if history:
                    for msg in history:
                        if isinstance(msg, dict):
                            role = msg.get("role")
                            content = msg.get("content", "")
                            if role == "interviewer":
                                chat_html += f'<div class="msg-bubble msg-interviewer">{content}</div>'
                            elif role == "candidate":
                                chat_html += f'<div class="msg-bubble msg-candidate">{content}</div>'
                chat_html += '</div>'
                st.markdown(chat_html, unsafe_allow_html=True)

            # Current question highlight
            current_q = state.get("current_question", "") if state else ""
            if current_q and (not history or history[-1].get("role") != "candidate"):
                # TTS playback for current question
                pass  # TTS handled via voice mode

            # Response input
            st.markdown("---")
            with st.form("response_form", clear_on_submit=True):
                user_response = st.text_area(
                    "Your Response",
                    placeholder="Type your answer here...",
                    height=100,
                    key="response_input"
                )

                col1, col2 = st.columns([3, 1])
                with col1:
                    submitted = st.form_submit_button(
                        "📤 Send Response",
                        type="primary",
                        use_container_width=True
                    )
                with col2:
                    tts_btn = st.form_submit_button("🔊 Read Question", use_container_width=True)

                if submitted and user_response.strip():
                    with st.spinner("🧠 Evaluating response..."):
                        result = call_api(API["respond"], data={
                            "session_id": st.session_state.session_id,
                            "response_text": user_response
                        })
                        if result:
                            st.session_state.interview_state = result["state"]
                            st.session_state.voice_started = True
                            st.rerun()

            # Continuous Voice Chat (WebRTC/WebSocket directly to FastAPI)
            st.markdown("---")
            st.markdown("### 🎙️ Live Voice Interview")
            st.info("Click 'Start Voice Interview' below to enable your microphone. Speak naturally. The system will automatically detect when you finish speaking and reply.")

            ws_url = f"{BACKEND_URL.replace('http', 'ws')}/ws/voice/{st.session_state.session_id}"

            voice_client_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <style>
                    body {{ font-family: 'Inter', sans-serif; background: #0a0f1c; color: white; display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100vh; margin: 0; padding: 20px; box-sizing: border-box; }}
                    button {{ background: #10b981; color: white; border: none; padding: 12px 24px; border-radius: 8px; font-weight: 600; cursor: pointer; font-size: 16px; transition: background 0.2s; }}
                    button:hover {{ background: #059669; }}
                    button.stop {{ background: #ef4444; }}
                    button.stop:hover {{ background: #dc2626; }}
                    button.submit {{ background: #3b82f6; display: none; margin-left: 10px; }}
                    button.submit:hover {{ background: #2563eb; }}
                    .status {{ margin-top: 15px; font-size: 14px; color: #94a3b8; text-align: center; }}
                    .pulse {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; background: #94a3b8; margin-right: 8px; transition: 0.2s; }}
                    .pulse.listening {{ background: #10b981; box-shadow: 0 0 8px #10b981; }}
                    .pulse.speaking {{ background: #3b82f6; box-shadow: 0 0 8px #3b82f6; }}
                </style>
            </head>
            <body>
                <div style="display: flex; align-items: center; gap: 15px;">
                    <button id="toggleBtn">Start Voice Interview</button>
                    <button id="flushBtn" class="submit" style="display:none;">Done Speaking</button>
                </div>
                <div class="status">
                    <span id="pulseBadge" class="pulse"></span>
                    <span id="statusText">Ready</span>
                </div>

                <script>
                    let ws = null;
                    let stream = null;
                    let audioContext = null;
                    let processor = null;
                    let isRecording = false;
                    let isProcessing = false;
                    let awaitingResponse = false;
                    let lastTurnId = null;
                    let currentAudio = null;
                    let isSpeaking = false;
                    let pendingState = null;

                    const btn = document.getElementById('toggleBtn');
                    const flushBtn = document.getElementById('flushBtn');
                    const status = document.getElementById('statusText');
                    const pulse = document.getElementById('pulseBadge');

                    async function startVoice() {{
                        try {{
                            stream = await navigator.mediaDevices.getUserMedia({{ audio: {{
                                echoCancellation: true,
                                noiseSuppression: true,
                                autoGainControl: true
                            }} }});
                            audioContext = new (window.AudioContext || window.webkitAudioContext)({{sampleRate: 16000}});
                            const source = audioContext.createMediaStreamSource(stream);
                            // ScriptProcessor is deprecated but universally supported for raw PCM
                            processor = audioContext.createScriptProcessor(4096, 1, 1);

                            ws = new WebSocket('{ws_url}');

                            ws.onopen = () => {{
                                status.innerText = "Listening...";
                                btn.innerText = "Stop Voice Interview";
                                btn.className = "stop";
                                flushBtn.style.display = "inline-block";
                                pulse.className = "pulse listening";
                                window.parent.postMessage({{isStreamlitMessage: true, type: 'streamlit:setComponentValue', value: JSON.stringify({{type: 'voice_started', session_id: '{st.session_state.session_id}'}})}}, '*');
                                ws.send(JSON.stringify({{action: 'set_manual_flush_only', enabled: true}}));

                                source.connect(processor);
                                processor.connect(audioContext.destination);

                                processor.onaudioprocess = (e) => {{
                                    if (isProcessing || isSpeaking) return;
                                    if (ws.readyState === WebSocket.OPEN) {{
                                        const floatData = e.inputBuffer.getChannelData(0);
                                        // Convert to 16-bit PCM
                                        const pcmData = new Int16Array(floatData.length);
                                        for (let i = 0; i < floatData.length; i++) {{
                                            let s = Math.max(-1, Math.min(1, floatData[i]));
                                            pcmData[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
                                        }}
                                        ws.send(pcmData.buffer);
                                    }}
                                }};
                            }};

                            flushBtn.onclick = () => {{
                                if (isProcessing) return;
                                if (ws && ws.readyState === WebSocket.OPEN) {{
                                    status.innerText = "Evaluating...";
                                    pulse.className = "pulse";
                                    isProcessing = true;
                                    awaitingResponse = true;
                                    flushBtn.disabled = true;
                                    ws.send(JSON.stringify({{action: 'flush'}}));
                                }}
                            }};

                            ws.onmessage = async (event) => {{
                                const data = JSON.parse(event.data);
                                if (data.type === 'processing') {{
                                    status.innerText = "Thinking...";
                                }}
                                if (data.type === 'turn_complete') {{
                                    if (!awaitingResponse) {{
                                        return;
                                    }}
                                    if (data.turn_id && data.turn_id === lastTurnId) {{
                                        return;
                                    }}
                                    lastTurnId = data.turn_id || Date.now();
                                    pendingState = data.state || null;
                                    awaitingResponse = false;

                                    if (currentAudio) {{
                                        currentAudio.pause();
                                        currentAudio.currentTime = 0;
                                    }}

                                    if (!data.audio) {{
                                        status.innerText = "Listening...";
                                        pulse.className = "pulse listening";
                                        isProcessing = false;
                                        awaitingResponse = false;
                                        flushBtn.disabled = false;
                                        if (pendingState) {{
                                            window.parent.postMessage({{isStreamlitMessage: true, type: 'streamlit:setComponentValue', value: JSON.stringify({{type: 'voice_state', state: pendingState, session_id: '{st.session_state.session_id}'}})}}, '*');
                                            pendingState = null;
                                        }}
                                        return;
                                    }}

                                    status.innerText = "Interviewer speaking...";
                                    pulse.className = "pulse speaking";
                                    isSpeaking = true;

                                    // Play received base64 WAV audio
                                    currentAudio = new Audio("data:audio/wav;base64," + data.audio);
                                    await currentAudio.play();

                                    currentAudio.onended = () => {{
                                        status.innerText = "Listening...";
                                        pulse.className = "pulse listening";
                                        isProcessing = false;
                                        flushBtn.disabled = false;
                                        isSpeaking = false;
                                        if (pendingState) {{
                                            window.parent.postMessage({{isStreamlitMessage: true, type: 'streamlit:setComponentValue', value: JSON.stringify({{type: 'voice_state', state: pendingState, session_id: '{st.session_state.session_id}'}})}}, '*');
                                            pendingState = null;
                                        }}
                                    }};
                                }}
                            }};

                            ws.onerror = (e) => {{
                                status.innerText = "Connection error";
                                stopVoice(false);
                            }};

                            ws.onclose = () => {{
                                status.innerText = "Disconnected";
                                stopVoice(false);
                            }};

                            isRecording = true;
                        }} catch (e) {{
                            status.innerText = "Microphone access denied or error";
                            console.error(e);
                        }}
                    }}

                    function stopVoice(sendFlush = true) {{
                        const activeWs = ws;
                        if (processor) {{ processor.disconnect(); processor = null; }}
                        if (stream) {{ stream.getTracks().forEach(t => t.stop()); stream = null; }}
                        ws = null;
                        isProcessing = false;
                        lastTurnId = null;
                        isSpeaking = false;
                        pendingState = null;
                        if (currentAudio) {{ currentAudio.pause(); currentAudio = null; }}

                        if (activeWs && activeWs.readyState === WebSocket.OPEN && sendFlush) {{
                            try {{ activeWs.send(JSON.stringify({{action: 'flush'}})); }} catch (e) {{}}
                            setTimeout(() => {{
                                try {{ activeWs.close(); }} catch (e) {{}}
                            }}, 200);
                        }} else if (activeWs) {{
                            try {{ activeWs.close(); }} catch (e) {{}}
                        }}

                        if (audioContext) {{ audioContext.close(); audioContext = null; }}

                        isRecording = false;
                        btn.innerText = "Start Voice Interview";
                        btn.className = "";
                        flushBtn.style.display = "none";
                        flushBtn.disabled = false;
                        pulse.className = "pulse";
                        status.innerText = "Ready";
                    }}

                    btn.onclick = () => {{
                        if (isRecording) stopVoice();
                        else startVoice();
                    }};
                </script>
            </body>
            </html>
            """
            import streamlit.components.v1 as components

            # We use a custom component just to trigger reruns when the JS sends a message
            # The JS handles all audio streaming natively over WS without blocking Streamlit!
            voice_event = components.html(voice_client_html, height=150)
            if voice_event:
                try:
                    raw_event = voice_event
                    payload = json.loads(voice_event) if isinstance(voice_event, str) else voice_event
                    if isinstance(payload, dict):
                        if payload.get("session_id") == st.session_state.session_id:
                            if raw_event != st.session_state.last_voice_event:
                                st.session_state.last_voice_event = raw_event
                                if payload.get("type") == "voice_state":
                                    new_state = payload.get("state")
                                    if isinstance(new_state, dict):
                                        st.session_state.interview_state = new_state
                                        st.session_state.voice_started = True
                                        st.rerun()
                                elif payload.get("type") == "voice_started":
                                    st.session_state.voice_started = True
                                    st.rerun()
                except Exception:
                    pass

        # ── WORKSPACE MODE ──
        with right_col:
            workspace = state.get("workspace", {})
            problem = workspace.get("problem", {})
            plan = state.get("interview_plan", []) if state else []
            show_workspace = bool(workspace.get("active")) or any(
                isinstance(step, dict) and step.get("type") == "coding" for step in plan
            )
            if not show_workspace:
                st.empty()
            else:
                st.markdown("### 🔧 Coding Workspace")

                # Problem description
                st.markdown(f"""
                <div class="workspace-panel">
                    <h3>💡 {problem.get('title', 'Coding Challenge')}</h3>
                    <p><strong>Difficulty:</strong> {problem.get('difficulty', 'medium').upper()} &nbsp;|&nbsp;
                    <strong>Time Limit:</strong> {problem.get('time_limit_minutes', 10)} minutes</p>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(problem.get("description", "No description available"))

            # Test cases preview
                test_cases = problem.get("test_cases", [])
                if test_cases:
                    with st.expander("📋 Test Cases", expanded=True):
                        for i, tc in enumerate(test_cases):
                            st.markdown(f"**Test {i+1}**: {tc.get('description', '')}")
                            st.code(f"Input:    {tc.get('input', '')}\nExpected: {tc.get('expected_output', '')}", language="text")

            # Code editor
                st.markdown("#### ✏️ Your Solution")
                starter_code = workspace.get("user_code", problem.get("starter_code", "# Write your code here\n"))

                user_code = st.text_area(
                    "Code Editor",
                    value=starter_code,
                    height=300,
                    key="code_editor",
                    label_visibility="collapsed"
                )

            # Action buttons
                col1, col2, col3 = st.columns(3)

                with col1:
                    if st.button("▶️ Run Code", use_container_width=True, key="run_code_btn"):
                        with st.spinner("Running..."):
                            result = call_api(API["run_code"], data={
                                "session_id": st.session_state.session_id,
                                "code": user_code
                            })
                            if result:
                                exec_result = result.get("execution_result", {})
                                if exec_result.get("success"):
                                    st.success("✅ Code executed successfully!")
                                else:
                                    st.error("❌ Execution failed")

                                if exec_result.get("stdout"):
                                    st.markdown("**Output:**")
                                    st.code(exec_result["stdout"], language="text")
                                if exec_result.get("stderr"):
                                    st.markdown("**Errors:**")
                                    st.code(exec_result["stderr"], language="text")
                                if exec_result.get("safety_error"):
                                    st.error(f"🔒 Safety: {exec_result['safety_error']}")

                                # Show test results
                                test_results = exec_result.get("test_results", [])
                                if test_results:
                                    passed = sum(1 for t in test_results if t.get("passed"))
                                    total = len(test_results)
                                    st.markdown(f"**Tests: {passed}/{total} passed**")
                                    for tr in test_results:
                                        icon = "✅" if tr.get("passed") else "❌"
                                        st.markdown(f"{icon} Test {tr.get('test', '?')}: {tr.get('description', '')}")
                                        if not tr.get("passed"):
                                            st.caption(f"Expected: `{tr.get('expected')}` | Got: `{tr.get('actual', tr.get('error', 'N/A'))}`")

                with col2:
                    if st.button("📤 Submit Solution", type="primary", use_container_width=True, key="submit_code_btn"):
                        with st.spinner("🧠 Evaluating solution..."):
                            result = call_api(API["submit_code"], data={
                                "session_id": st.session_state.session_id,
                                "code": user_code
                            })
                            if result:
                                st.session_state.interview_state = result["state"]

                                exec_result = result.get("execution_result", {})
                                passed = exec_result.get("passed_tests", 0)
                                total = exec_result.get("total_tests", 0)

                                if passed == total and total > 0:
                                    st.balloons()
                                    st.success(f"🎉 All {total} tests passed!")
                                elif passed > 0:
                                    st.warning(f"Passed {passed}/{total} tests")
                                else:
                                    st.error("Tests failed — but your approach matters too!")

                                time.sleep(2)
                                st.rerun()

        if st.session_state.interview_started:
            changed = maybe_poll_state()
            if changed:
                st.rerun()
            time.sleep(0.8)
            st.rerun()

            with col3:
                if st.button("⏭️ Skip Challenge", use_container_width=True, key="skip_code_btn"):
                    result = call_api(API["submit_code"], data={
                        "session_id": st.session_state.session_id,
                        "code": "# Skipped by candidate"
                    })
                    if result:
                        st.session_state.interview_state = result["state"]
                        st.rerun()

            # Hints
            hints = problem.get("hints", [])
            if hints:
                with st.expander("💡 Hints (try without first!)"):
                    for i, hint in enumerate(hints):
                        st.markdown(f"**Hint {i+1}:** {hint}")

        # ── COMPLETE MODE ──
    elif mode == "complete":
        report = state.get("final_report", {})

        st.markdown("""
        <div class="report-card">
            <h2>📋 Interview Assessment Report</h2>
        </div>
        """, unsafe_allow_html=True)

        # Overall score
        overall = report.get("overall_score", 0)
        recommendation = report.get("recommendation", "borderline")

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            # Big score display
            score_color = get_score_color(overall)
            st.markdown(f"""
            <div style="text-align: center; padding: 1.5rem;">
                <div style="font-size: 4rem; font-weight: 800; color: {score_color};">
                    {int(overall * 100)}%
                </div>
                <div style="font-size: 0.9rem; color: var(--text-dim); margin-bottom: 1rem;">Overall Score</div>
            </div>
            """, unsafe_allow_html=True)

            # Recommendation badge
            rec_class = {
                "strong_hire": "rec-strong",
                "hire": "rec-hire",
                "borderline": "rec-borderline",
                "no_hire": "rec-nohire"
            }.get(recommendation, "rec-borderline")

            rec_label = recommendation.replace("_", " ").upper()

            st.markdown(f"""
            <div style="text-align: center; margin-bottom: 2rem;">
                <span class="rec-badge {rec_class}">{rec_label}</span>
            </div>
            """, unsafe_allow_html=True)

        # Detailed breakdown
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 💪 Strengths")
            for s in report.get("strengths", []):
                st.markdown(f"✅ {s}")

            st.markdown("#### 📈 Skill Breakdown")
            skills = report.get("skill_breakdown", {})
            if skills:
                for skill_name, skill_data in skills.items():
                    if isinstance(skill_data, dict):
                        score = skill_data.get("score", 0)
                        render_score_bar(skill_name[:14], score)
                        if skill_data.get("notes"):
                            st.caption(f"→ {skill_data['notes']}")

        with col2:
            st.markdown("#### ⚠️ Areas for Improvement")
            for w in report.get("weaknesses", []):
                st.markdown(f"🔸 {w}")

            st.markdown("#### 📝 Suggested Next Steps")
            if report.get("suggested_next_steps"):
                st.markdown(report["suggested_next_steps"])

        # Detailed feedback
        if report.get("detailed_feedback"):
            st.markdown("---")
            st.markdown("#### 📄 Detailed Assessment")
            st.markdown(report["detailed_feedback"])

        # Download transcript
        st.markdown("---")
        history = state.get("conversation_history", [])
        if history:
            transcript_text = "\n\n".join([
                f"{'INTERVIEWER' if msg.get('role')=='interviewer' else 'CANDIDATE'}: {msg.get('content', '')}"
                for msg in history if isinstance(msg, dict)
            ])

            full_report = f"""
AI INTERVIEW REPORT
{'='*50}
Role: {state.get('role', 'N/A')}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}
Overall Score: {int(overall * 100)}%
Recommendation: {rec_label}

DETAILED FEEDBACK:
{report.get('detailed_feedback', 'N/A')}

STRENGTHS:
{chr(10).join('- ' + s for s in report.get('strengths', []))}

WEAKNESSES:
{chr(10).join('- ' + w for w in report.get('weaknesses', []))}

TRANSCRIPT:
{'-'*50}
{transcript_text}
"""
            st.download_button(
                "📥 Download Full Report",
                full_report,
                file_name=f"interview_report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain",
                use_container_width=True,
                key="download_report"
            )

# ────────────────────────────────────────────
# Last eval display (bottom of page, conversation mode)
# ────────────────────────────────────────────
if st.session_state.interview_started and state:
    mode = state.get("mode", "")
    if mode == "conversation":
        eval_data = state.get("evaluation", {})
        if eval_data and eval_data.get("feedback"):
            with st.expander("🔍 Last Response Evaluation (Internal)", expanded=False):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    render_score_bar("Correct", eval_data.get("correctness", 0))
                with col2:
                    render_score_bar("Depth", eval_data.get("depth", 0))
                with col3:
                    render_score_bar("Clarity", eval_data.get("clarity", 0))
                with col4:
                    render_score_bar("Confid.", eval_data.get("confidence", 0))
                st.caption(f"💡 {eval_data.get('feedback', '')}")
