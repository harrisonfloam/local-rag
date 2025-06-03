from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="APP_")

    debug: bool = False
    log_level: str = "INFO"
    host: str = "0.0.0.0"
    port: int = 8000

    ollama_url: str = "http://ollama:11434/v1"
    vector_db_url: str = ""

    documents_path: str = "./data/documents"
    api_prefix: str = "/api"

    llm_name: str = "mistral:latest"

    logging_config: dict = {
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
