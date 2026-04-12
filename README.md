# 🎯 AI Interviewer System

A production-grade, real-time AI interviewer system inspired by Eightfold-style adaptive interviewing. This system conducts personalized technical interviews with dynamic question adaptation, live evaluation, hands-on coding challenges, and comprehensive final assessments.

## 🏗️ Architecture

```
User (Mic + Resume Upload)
→ Profile Extraction Agent (Gemini)
→ Interview Planner Agent (Gemini)
→ Combined Responder/Evaluator Agent (Gemini)
→ LangGraph Orchestrator
  ├── Task Generator Agent
  └── Workspace Module (Code Execution)
→ Response (Text + TTS Audio)
```

### Tech Stack
- **Backend**: FastAPI + LangGraph + WebSockets
- **AI**: Google Gemini (LLM reasoning, STT, TTS)
- **Frontend**: Streamlit
- **State Machine**: LangGraph StateGraph with conditional routing

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.11+
- Google Gemini API key

### 2. Setup

```bash
# Clone and navigate
cd eightfold

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

### 3. Run

```bash
# Terminal 1: Start backend
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: Start frontend
streamlit run frontend/app.py --server.port 8501
```

### 4. Use
1. Open http://localhost:8501
2. Enter a role (e.g., "Machine Learning Engineer")
3. Upload a resume (PDF or TXT) — optional
4. Click "Start Interview"
5. Answer questions in the text box
6. Complete coding challenges when presented
7. View your final assessment report

## 🧠 System Components

### Agents
| Agent | Purpose |
|-------|---------|
| **Profile Extractor** | Parses resume → structured profile with skill depth inference |
| **Interview Planner** | Generates adaptive interview plan based on profile |
| **Responder/Evaluator** | Scores each response and generates the next interviewer turn in one step |
| **Task Generator** | Creates coding challenges (diagnostic or challenge mode) |

### LangGraph Flow
```
Init:     START → Profile → Plan → First Question → END
Turn:     START → Respond+Evaluate → Route →
            ├── END (next question already generated)
            ├── Task → END
            └── Final Report → END
Workspace: START → Evaluate Code → Advance → Question → END
```

### Modes
- **Conversation**: Q&A interview flow
- **Workspace**: Hands-on coding challenge
- **Complete**: Final report generated

## 📊 Evaluation Dimensions
- **Correctness** (35%): Factual accuracy
- **Depth** (30%): Beyond surface-level understanding
- **Clarity** (20%): Communication quality
- **Confidence** (15%): Assuredness (inferred)

## 🔒 Safety
- Code sandbox with blocked imports (os, sys, subprocess, etc.)
- Execution timeout (5 seconds)
- Dangerous pattern detection (e.g., eval/exec/open)
- Isolated execution directory and Python isolated mode (`python -I`)
- Note: this is a hardened interview sandbox, not a fully isolated container sandbox

## 📁 Project Structure
```
eightfold/
├── backend/
│   ├── main.py              # FastAPI + WebSocket endpoints
│   ├── config.py             # Settings & configuration
│   ├── state.py              # LangGraph state definition
│   ├── graph.py              # LangGraph orchestrator
│   ├── agents/
│   │   ├── profile_extractor.py
│   │   ├── interview_planner.py
│   │   ├── conversation.py
│   │   ├── evaluator.py
│   │   └── task_generator.py
│   ├── voice/
│   │   ├── stt.py            # Speech-to-text
│   │   └── tts.py            # Text-to-speech
│   ├── workspace/
│   │   └── executor.py       # Safe code execution
│   └── utils/
│       └── resume_parser.py  # PDF/text parsing
├── frontend/
│   └── app.py                # Streamlit UI
├── requirements.txt
├── .env.example
└── README.md
```

## ⚙️ Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | — | Your Gemini API key (required) |
| `GEMINI_MODEL` | `gemini-2.5-flash-preview-04-17` | LLM model for reasoning |
| `TTS_MODEL` | `gemini-2.5-flash-preview-tts` | TTS model |
| `BACKEND_PORT` | `8000` | FastAPI server port |
| `BACKEND_URL` | `http://localhost:8000` | Frontend target URL for backend API |
| `CORS_ORIGINS` | `http://localhost:8501,http://127.0.0.1:8501` | Allowed CORS origins |
| `VOICE_SILENCE_SECONDS` | `1.1` | Silence window before auto-transcribe in voice mode |
| `SESSION_STORE_ENABLED` | `true` | Persist session state to disk across backend restarts |
| `SESSION_STORE_PATH` | `.session_store.json` | Session storage file path |
| `CODE_EXECUTION_TIMEOUT` | `5` | Max seconds for code execution |
| `MAX_INTERVIEW_STEPS` | `12` | Maximum interview steps |
| `MIN_INTERVIEW_STEPS` | `8` | Minimum interview steps |

## 🧪 Demo Scenario
1. Upload an ML Engineer resume
2. System extracts: Python (expert), TensorFlow (advanced), SQL (intermediate)
3. Plan: Intro → ML Concepts → Deep Learning → Coding → System Design → Wrap-up
4. Adaptive flow: weak on transformers → probing follow-up → coding task on attention
5. Strong on optimization → harder challenge on distributed training
6. Final report: Overall 72%, Hire recommendation
