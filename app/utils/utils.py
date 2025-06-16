import logging
import logging.config

from app.settings import settings


def init_logging(log_level: str = settings.log_level):
    """
    Initialize logging using the configuration defined in settings.logging_config.
    """
    log_level = log_level.upper()
    config = settings.logging_config.copy()
    config["root"]["level"] = log_level
    for handler in config["handlers"].values():
        handler["level"] = log_level
    logging.config.dictConfig(config)
