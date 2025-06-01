import logging
import logging.config

from app.settings import settings


def init_logging():
    """
    Initialize logging using the configuration defined in settings.LOGGING_CONFIG.
    """
    logging.config.dictConfig(settings.LOGGING_CONFIG)
