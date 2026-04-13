"""
Configuration module for the AI Interviewer system.
Loads settings from environment variables with sensible defaults.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Gemini API
    gemini_api_key: str = Field(default="", env="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-2.5-flash", env="GEMINI_MODEL")
    tts_model: str = Field(default="gemini-2.5-flash-preview-tts", env="TTS_MODEL")
    live_model: str = Field(default="gemini-2.5-flash", env="LIVE_MODEL")
    gemini_thinking_budget: int = Field(default=0, env="GEMINI_THINKING_BUDGET")
    gemini_top_p: float = Field(default=0.9, env="GEMINI_TOP_P")
    gemini_top_k: int = Field(default=20, env="GEMINI_TOP_K")
    gemini_json_max_tokens: int = Field(default=900, env="GEMINI_JSON_MAX_TOKENS")
    gemini_text_max_tokens: int = Field(default=320, env="GEMINI_TEXT_MAX_TOKENS")

    # Groq API (for fast STT)
    groq_api_key: str = Field(default="", env="GROQ_API_KEY")

    # LLM provider selection
    llm_provider: str = Field(default="gemini", env="LLM_PROVIDER")
    groq_llm_model: str = Field(default="llama-3.1-8b-instant", env="GROQ_LLM_MODEL")

    # Server
    backend_host: str = Field(default="0.0.0.0", env="BACKEND_HOST")
    backend_port: int = Field(default=8000, env="BACKEND_PORT")
    backend_url: str = Field(default="http://localhost:8000", env="BACKEND_URL")
    cors_origins: str = Field(
        default="http://localhost:8501,http://127.0.0.1:8501",
        env="CORS_ORIGINS"
    )

    # Sandbox
    code_execution_timeout: int = Field(default=5, env="CODE_EXECUTION_TIMEOUT")
    max_output_length: int = Field(default=10000, env="MAX_OUTPUT_LENGTH")

    # Voice
    stt_provider: str = Field(default="groq", env="STT_PROVIDER")
    stt_model: str = Field(default="whisper-large-v3-turbo", env="STT_MODEL")
    stt_language: str = Field(default="en", env="STT_LANGUAGE")

    tts_provider: str = Field(default="piper", env="TTS_PROVIDER")
    piper_binary: str = Field(default="piper", env="PIPER_BINARY")
    piper_model_path: str = Field(default="models/piper/en_US-lessac-medium.onnx", env="PIPER_MODEL_PATH")
    piper_model_config_path: str = Field(default="models/piper/en_US-lessac-medium.onnx.json", env="PIPER_MODEL_CONFIG_PATH")
    piper_speaker_id: int = Field(default=0, env="PIPER_SPEAKER_ID")
    piper_noise_scale: float = Field(default=0.667, env="PIPER_NOISE_SCALE")
    piper_length_scale: float = Field(default=1.0, env="PIPER_LENGTH_SCALE")
    piper_noise_w: float = Field(default=0.8, env="PIPER_NOISE_W")

    voice_silence_seconds: float = Field(default=0.6, env="VOICE_SILENCE_SECONDS")
    voice_min_chunk_seconds: float = Field(default=0.4, env="VOICE_MIN_CHUNK_SECONDS")

    # Session persistence
    session_store_enabled: bool = Field(default=True, env="SESSION_STORE_ENABLED")
    session_store_path: str = Field(default=".session_store.json", env="SESSION_STORE_PATH")

    # Interview
    max_interview_steps: int = Field(default=5, env="MAX_INTERVIEW_STEPS")
    min_interview_steps: int = Field(default=5, env="MIN_INTERVIEW_STEPS")

    # Profile extraction
    fast_profile_mode: bool = Field(default=True, env="FAST_PROFILE_MODE")

    # Blocked imports for sandbox
    blocked_imports: list[str] = [
        "os", "sys", "subprocess", "shutil", "pathlib",
        "socket", "http", "urllib", "requests", "ftplib",
        "smtplib", "ctypes", "signal", "multiprocessing",
        "threading", "importlib", "__import__"
    ]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
