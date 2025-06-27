import os

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="APP_")

    # Server settings
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    httpx_timeout: int = 30
    api_prefix: str = "/api"

    @property
    def api_url(self) -> str:
        return f"http://{self.host}:{self.port}{self.api_prefix}"

    # LLM settings
    ollama_url: str = "http://localhost:11434/v1"
    model_name: str = "llama3.2:1b"
    temperature: float = 0.7
    mock_llm: bool = False  # DEV: Use a mock LLM for testing purposes
    # TODO: conversation memory settings

    # RAG Settings
    # TODO: vectorstore settings, chunking, etc
    use_rag: bool = True
    top_k: int = 5
    vector_db_url: str = ""

    documents_path: str = "./data/documents"
    mock_rag_response: bool = False  # DEV: Use a mock RAG response for testing
    mock_documents: bool = False  # DEV: Use mock documents for testing

    # Frontend settings
    dev_mode: bool = False  # Enable additional options in the UI
    use_rag: bool = True  # Use RAG context in the chat

    @property
    def chroma_url(self) -> str:
        return f"http://{self.chroma_host}:{self.chroma_port}"

    # Logging settings
    log_level: str = "INFO"
    dependency_log_level: str = "WARNING"

    @property
    def logging_config(self) -> dict:
        return {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M",
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "standard",
                    "level": self.log_level,
                }
            },
            "root": {
                "handlers": ["console"],
                "level": self.log_level,
            },
            "loggers": {
                "uvicorn": {
                    "level": self.dependency_log_level,
                    "handlers": ["console"],
                    "propagate": False,
                },
                "openai": {
                    "level": self.dependency_log_level,
                    "handlers": ["console"],
                    "propagate": False,
                },
                "httpx": {
                    "level": self.dependency_log_level,
                    "handlers": ["console"],
                    "propagate": False,
                },
                "httpcore": {
                    "level": self.dependency_log_level,
                    "handlers": ["console"],
                    "propagate": False,
                },
                "uvicorn.access": {
                    "level": self.dependency_log_level,
                    "handlers": ["console"],
                    "propagate": False,
                },
                "watchfiles": {
                    "level": self.dependency_log_level,
                    "handlers": ["console"],
                    "propagate": False,
                },
            },
        }


def reset_settings(**overrides):
    """Override default settings."""
    keys_to_remove = [key for key in os.environ.keys() if key.startswith("APP_")]
    for key in keys_to_remove:
        print(f"Removing environment variable: {key}")
        del os.environ[key]

    for key, value in overrides.items():
        env_key = f"APP_{key.upper()}"
        os.environ[env_key] = str(value)

    global settings
    settings = Settings()


settings = Settings()
