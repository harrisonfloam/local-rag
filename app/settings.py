from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="APP_")

    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    OLLAMA_URL: str = "http://ollama:11434/v1"
    VECTOR_DB_URL: str = ""

    DOCUMENTS_PATH: str = "./data/documents"
    API_PREFIX: str = "/api"

    LLM_NAME: str = "mistral:latest"

    LOGGING_CONFIG: dict = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {"format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"}
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "standard",
                "level": "DEBUG",
            }
        },
        "root": {
            "handlers": ["console"],
            "level": "INFO",
        },
    }


settings = Settings()
