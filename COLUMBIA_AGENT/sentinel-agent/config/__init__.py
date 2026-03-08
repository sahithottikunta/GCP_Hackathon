"""
SENTINEL — Configuration Management
"""

from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True
    cors_origins: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8000",
    ]

    # Anthropic
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-20250514"

    # OpenAI
    openai_api_key: str = ""
    openai_tts_model: str = "tts-1"
    openai_tts_voice: str = "onyx"
    openai_stt_model: str = "whisper-1"

    # Google Gemini
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"
    google_service_account_file: str = ""
    google_cloud_project: str = ""

    # Agent
    voice_interrupt_threshold_ms: int = 300
    max_concurrent_agents: int = 5
    agent_response_timeout_s: int = 30

    # Market Data
    market_data_interval_ms: int = 500
    anomaly_scan_interval_ms: int = 2000
    max_history_points: int = 5000

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    @property
    def has_anthropic(self) -> bool:
        return bool(self.anthropic_api_key and self.anthropic_api_key != "sk-ant-xxxxx")

    @property
    def has_openai(self) -> bool:
        return bool(self.openai_api_key and self.openai_api_key != "sk-xxxxx")

    @property
    def has_gemini(self) -> bool:
        has_api_key = bool(self.gemini_api_key and self.gemini_api_key != "your-gemini-api-key-here")
        has_service_account = bool(self.google_service_account_file and self.google_cloud_project)
        return has_api_key or has_service_account


settings = Settings()
