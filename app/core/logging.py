import logging
import sys

from app.core.config import get_settings


def configure_logging() -> None:
    """
    Call once at startup. Keeps formatting consistent across every
    module that does `logging.getLogger(__name__)`.
    """
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        stream=sys.stdout,
    )