"""
FastAPI Application — Main entry point.
Provides REST endpoints and WebSocket for the AI Interviewer system.
"""

import json
import base64
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.config import settings
from backend.state import create_initial_state
from backend.graph import init_graph, conversation_graph, workspace_graph
from backend.workspace.executor import execute_code
from backend.voice.stt import transcribe
from backend.voice.tts import synthesize

# ────────────────────────────────────────────
# App setup
# ────────────────────────────────────────────
app = FastAPI(
    title="AI Interviewer System",
    description="Production-grade, real-time AI interviewer with adaptive questioning",
    version="1.0.0"
)

allowed_origins = [origin.strip() for origin in settings.cors_origins.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store with optional disk persistence.
_session_store_path = Path(settings.session_store_path)


def _load_sessions() -> dict[str, dict]:
    if not settings.session_store_enabled:
        return {}
    try:
        if _session_store_path.exists():
            data = json.loads(_session_store_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    except Exception as exc:
        print(f"Session load warning: {exc}")
    return {}


def _save_sessions() -> None:
    if not settings.session_store_enabled:
        return
    try:
        _session_store_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = _session_store_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(sessions), encoding="utf-8")
        tmp_path.replace(_session_store_path)
    except Exception as exc:
        print(f"Session save warning: {exc}")


sessions: dict[str, dict] = _load_sessions()


def _generate_session_id() -> str:
    """Create a collision-resistant interview session id."""
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    return f"interview_{stamp}_{uuid4().hex[:10]}"


# ────────────────────────────────────────────
# Request/Response models
# ────────────────────────────────────────────
class StartInterviewRequest(BaseModel):
    role: str
    resume_text: Optional[str] = ""
    session_id: Optional[str] = None


class SubmitResponseRequest(BaseModel):
    session_id: str
    response_text: str


class SubmitCodeRequest(BaseModel):
    session_id: str
    code: str


class EndInterviewRequest(BaseModel):
    session_id: str


# ────────────────────────────────────────────
# Health check
# ────────────────────────────────────────────
@app.get("/api/health")
async def health():
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


# ────────────────────────────────────────────
# Start Interview
# ────────────────────────────────────────────
@app.post("/api/start-interview")
async def start_interview_endpoint(
    role: str = Form(...),
    resume: Optional[UploadFile] = File(None),
    session_id: Optional[str] = Form(None)
):
    """
    Start a new interview session.
    Accepts role + optional resume file upload.
    Triggers: Profile Extraction → Interview Planning → First Question
    """
    resume_data = ""
    resume_mime = "text/plain"
    if resume:
        file_bytes = await resume.read()
        resume_data = base64.b64encode(file_bytes).decode('utf-8')
        resume_mime = resume.content_type or "application/pdf"

    # Generate session ID
    if not session_id:
        session_id = _generate_session_id()

    # Create initial state
    state = create_initial_state(role=role, resume_data=resume_data, resume_mime_type=resume_mime)

    # Run initialization graph: profile → plan → first question
    try:
        result = await init_graph.ainvoke(state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Interview initialization failed: {str(e)}")

    # Store session
    sessions[session_id] = result
    _save_sessions()

    return {
        "session_id": session_id,
        "state": _sanitize_state(result)
    }


@app.post("/api/start-interview-json")
async def start_interview_json(req: StartInterviewRequest):
    """Alternative JSON-based start endpoint (no file upload)."""
    state = create_initial_state(role=req.role, resume_data=base64.b64encode((req.resume_text or "").encode()).decode(), resume_mime_type="text/plain")
    session_id = req.session_id or _generate_session_id()

    try:
        result = await init_graph.ainvoke(state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Interview initialization failed: {str(e)}")

    sessions[session_id] = result
    _save_sessions()
    return {"session_id": session_id, "state": _sanitize_state(result)}


# ────────────────────────────────────────────
# Submit Response (Text)
# ────────────────────────────────────────────
@app.post("/api/submit-response")
async def submit_response(req: SubmitResponseRequest):
    """
    Submit a candidate response and get the next interview action.
    Triggers: Input Processing → Evaluation → Routing → Next Question/Task/End
    """
    if req.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    state = sessions[req.session_id]

    if state.get("mode") == "complete":
        return {
            "session_id": req.session_id,
            "state": _sanitize_state(state),
            "message": "Interview already completed"
        }

    # Update state with user response
    state["last_user_response"] = req.response_text

    # Run conversation graph
    try:
        result = await conversation_graph.ainvoke(state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

    # Update session
    sessions[req.session_id] = result
    _save_sessions()

    return {
        "session_id": req.session_id,
        "state": _sanitize_state(result)
    }


# ────────────────────────────────────────────
# Submit Code (Workspace)
# ────────────────────────────────────────────
@app.post("/api/submit-code")
async def submit_code(req: SubmitCodeRequest):
    """
    Submit code from the workspace for execution and evaluation.
    Triggers: Code Execution → Workspace Evaluation → Next Question
    """
    if req.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    state = sessions[req.session_id]
    workspace = state.get("workspace", {})

    if not workspace.get("active", False):
        raise HTTPException(status_code=400, detail="No active workspace task")

    # Execute code
    problem = workspace.get("problem", {})
    test_cases = problem.get("test_cases", [])
    exec_result = execute_code(req.code, test_cases)

    # Update workspace state
    state["workspace"]["user_code"] = req.code
    state["workspace"]["result"] = exec_result

    # Run workspace evaluation graph
    try:
        result = await workspace_graph.ainvoke(state)
        # After coding challenge, generate final report immediately
        from backend.graph import generate_final_report
        result = {**state, **result}
        final_result = await generate_final_report(result)
        result.update(final_result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Workspace evaluation failed: {str(e)}")

    sessions[req.session_id] = result
    _save_sessions()

    return {
        "session_id": req.session_id,
        "state": _sanitize_state(result),
        "execution_result": exec_result
    }


# ────────────────────────────────────────────
# Run Code (No evaluation, just execute)
# ────────────────────────────────────────────
@app.post("/api/run-code")
async def run_code(req: SubmitCodeRequest):
    """Run code without triggering evaluation — for testing during workspace mode."""
    if req.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    state = sessions[req.session_id]
    workspace = state.get("workspace", {})

    if not workspace.get("active", False):
        raise HTTPException(status_code=400, detail="No active workspace task")

    problem = workspace.get("problem", {})
    test_cases = problem.get("test_cases", [])

    exec_result = execute_code(req.code, test_cases)

    return {
        "session_id": req.session_id,
        "execution_result": exec_result
    }


# ────────────────────────────────────────────
# End Interview
# ────────────────────────────────────────────
@app.post("/api/end-interview")
async def end_interview(req: EndInterviewRequest):
    """Force-end the interview and generate final report."""
    if req.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    state = sessions[req.session_id]

    if state.get("mode") == "complete":
        return {"session_id": req.session_id, "state": _sanitize_state(state)}

    # Force step to end
    state["current_step"] = len(state.get("interview_plan", []))
    state["last_user_response"] = "[Interview ended by candidate]"

    try:
        from backend.graph import generate_final_report
        result = await generate_final_report(state)
        state.update(result)
        result = state
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")

    sessions[req.session_id] = result
    _save_sessions()
    return {"session_id": req.session_id, "state": _sanitize_state(result)}


# ────────────────────────────────────────────
# Get State
# ────────────────────────────────────────────
@app.get("/api/state/{session_id}")
async def get_state(session_id: str):
    """Get current interview state for a session."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"session_id": session_id, "state": _sanitize_state(sessions[session_id])}


# ────────────────────────────────────────────
# WebSocket: Voice Pipeline
# ────────────────────────────────────────────
@app.websocket("/ws/voice/{session_id}")
async def websocket_voice(websocket: WebSocket, session_id: str):
    """
    Bidirectional voice streaming WebSocket.

    Client sends: binary audio chunks (16kHz, 16-bit PCM)
    Server sends: JSON messages with transcription + TTS audio

    Protocol:
    1. Client sends audio chunks
    2. Server accumulates and transcribes
    3. Server runs LangGraph conversation turn
    4. Server generates TTS response
    5. Server sends back: {"transcript": str, "response": str, "audio": base64, "state": dict}
    """
    await websocket.accept()

    if session_id not in sessions:
        await websocket.send_json({"error": "Session not found"})
        await websocket.close()
        return

    from backend.voice.stt import AudioAccumulator
    accumulator = AudioAccumulator(
        silence_threshold_rms=500,
        silence_duration_sec=settings.voice_silence_seconds,
        min_chunk_seconds=settings.voice_min_chunk_seconds,
    )
    turn_id = 0
    manual_flush_only = True
    is_processing_turn = False

    try:
        while True:
            # Receive data from client
            data = await websocket.receive()

            if "bytes" in data:
                # Audio chunk received
                audio_chunk = data["bytes"]
                is_ready = accumulator.add_chunk(audio_chunk)

                if is_ready and not manual_flush_only and not is_processing_turn:
                    # Silence detected → Transcribe accumulated audio
                    audio_bytes = accumulator.get_audio_and_reset()
                    is_processing_turn = True
                    transcript = await transcribe(audio_bytes)

                    if transcript and transcript not in ["[silence]", "[inaudible]"]:
                        # Send acknowledgment
                        await websocket.send_json({"type": "processing", "transcript": transcript})
                        
                        # Process through conversation graph
                        state = sessions[session_id]
                        state["last_user_response"] = transcript

                        result = await conversation_graph.ainvoke(state)
                        sessions[session_id] = result
                        _save_sessions()

                        # Generate TTS for response
                        response_text = result.get("current_question", "")
                        audio_response = await synthesize(response_text)
                        turn_id += 1

                        await websocket.send_json({
                            "type": "turn_complete",
                            "transcript": transcript,
                            "response": response_text,
                            "audio": base64.b64encode(audio_response).decode() if audio_response else "",
                            "turn_id": turn_id,
                            "state": _sanitize_state(result)
                        })
                    is_processing_turn = False

            elif "text" in data:
                # JSON command from client
                try:
                    cmd = json.loads(data["text"])
                    if cmd.get("action") == "set_manual_flush_only":
                        manual_flush_only = bool(cmd.get("enabled", True))

                    elif cmd.get("action") == "flush":
                        if is_processing_turn:
                            continue
                        # Flush remaining audio
                        remaining = accumulator.flush()
                        if remaining:
                            is_processing_turn = True
                            transcript = await transcribe(remaining)
                            if transcript and transcript not in ["[silence]", "[inaudible]"]:
                                state = sessions[session_id]
                                state["last_user_response"] = transcript
                                result = await conversation_graph.ainvoke(state)
                                sessions[session_id] = result
                                _save_sessions()

                                response_text = result.get("current_question", "")
                                audio_response = await synthesize(response_text)
                                turn_id += 1

                                await websocket.send_json({
                                    "type": "turn_complete",
                                    "transcript": transcript,
                                    "response": response_text,
                                    "audio": base64.b64encode(audio_response).decode() if audio_response else "",
                                    "turn_id": turn_id,
                                    "state": _sanitize_state(result)
                                })
                            is_processing_turn = False

                    elif cmd.get("action") == "text_input":
                        # Direct text input (fallback mode)
                        text = cmd.get("text", "")
                        if text:
                            if is_processing_turn:
                                continue
                            is_processing_turn = True
                            state = sessions[session_id]
                            state["last_user_response"] = text
                            result = await conversation_graph.ainvoke(state)
                            sessions[session_id] = result
                            _save_sessions()

                            response_text = result.get("current_question", "")
                            audio_response = await synthesize(response_text)
                            turn_id += 1

                            await websocket.send_json({
                                "type": "turn_complete",
                                "transcript": text,
                                "response": response_text,
                                "audio": base64.b64encode(audio_response).decode() if audio_response else "",
                                "turn_id": turn_id,
                                "state": _sanitize_state(result)
                            })
                            is_processing_turn = False

                except json.JSONDecodeError:
                    pass

    except WebSocketDisconnect:
        print(f"Voice WebSocket disconnected: {session_id}")
    except Exception as e:
        print(f"Voice WebSocket error: {e}")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


# ────────────────────────────────────────────
# TTS endpoint (REST fallback)
# ────────────────────────────────────────────
@app.post("/api/tts")
async def text_to_speech(text: str = Form(...)):
    """Generate TTS audio for given text. Returns base64 WAV audio."""
    audio_bytes = await synthesize(text)
    if not audio_bytes:
        raise HTTPException(status_code=500, detail="TTS generation failed")
    return {
        "audio": base64.b64encode(audio_bytes).decode(),
        "format": "wav"
    }


# ────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────
def _sanitize_state(state: dict) -> dict:
    """
    Sanitize state for client consumption.
    Removes raw resume text and internal metadata.
    """
    sanitized = {}
    for key, value in state.items():
        if key in ["resume_text", "resume_data"]:
            sanitized[key] = "[uploaded]" if value else ""
        elif key == "conversation_history":
            # Clean conversation history for serialization
            sanitized[key] = _clean_conversation_history(value)
        else:
            try:
                json.dumps(value)  # Test if serializable
                sanitized[key] = value
            except (TypeError, ValueError):
                sanitized[key] = str(value)
    return sanitized


def _clean_conversation_history(history) -> list:
    """Clean conversation history entries for JSON serialization."""
    clean = []
    for entry in history:
        if isinstance(entry, dict):
            clean.append({
                "role": entry.get("role", "unknown"),
                "content": entry.get("content", ""),
                "timestamp": entry.get("timestamp", ""),
                "step": entry.get("step", 0)
            })
    return clean


# ────────────────────────────────────────────
# Run
# ────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=settings.backend_host,
        port=settings.backend_port,
        reload=True
    )
