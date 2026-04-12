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

    # Server
    backend_host: str = Field(default="0.0.0.0", env="BACKEND_HOST")
    backend_port: int = Field(default=8000, env="BACKEND_PORT")
    cors_origins: str = Field(
        default="http://localhost:8501,http://127.0.0.1:8501",
        env="CORS_ORIGINS"
    )

    # Sandbox
    code_execution_timeout: int = Field(default=5, env="CODE_EXECUTION_TIMEOUT")
    max_output_length: int = Field(default=10000, env="MAX_OUTPUT_LENGTH")

    # Voice
    voice_silence_seconds: float = Field(default=1.1, env="VOICE_SILENCE_SECONDS")

    # Session persistence
    session_store_enabled: bool = Field(default=True, env="SESSION_STORE_ENABLED")
    session_store_path: str = Field(default=".session_store.json", env="SESSION_STORE_PATH")

    # Interview
    max_interview_steps: int = Field(default=12, env="MAX_INTERVIEW_STEPS")
    min_interview_steps: int = Field(default=8, env="MIN_INTERVIEW_STEPS")

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
