import logging
from functools import wraps

import streamlit as st

logger = logging.getLogger(__name__)


class StreamlitRerunNeeded(Exception):
    """Raised when a UI refresh is needed after an operation. Optionally, display a message after the rerun."""

    def __init__(
        self,
        message="Operation completed, refresh needed",
        show_after_rerun=False,
        pending_message_kwargs={},
    ):
        self.message = message

        self.show_after_rerun = show_after_rerun
        self.pending_message_kwargs = pending_message_kwargs
        if show_after_rerun:
            st.session_state.pending_message = {
                "type": self.pending_message_kwargs.get("type", "toast"),
                "message": self.message,
                "icon": self.pending_message_kwargs.get("icon", "ℹ️"),
            }
        super().__init__(self.message)


class StreamlitToastMessage(Exception):
    """Raised when a toast message should be shown."""

    def __init__(self, message, details=None, icon="ℹ️"):
        self.message = message
        self.details = details
        self.icon = icon
        super().__init__(self.message)


class StreamlitErrorMessage(Exception):
    """Raised when an error message should be shown."""

    def __init__(
        self, message, details=None, icon="❌", style="toast", stop_execution=False
    ):
        self.message = message
        self.details = details  # Only shown in dev_mode
        self.icon = icon
        self.style = style  # "toast" or "error"
        self.stop_execution = stop_execution
        super().__init__(self.message)


def handle_streamlit_exceptions(func):
    """Decorator to handle all Streamlit-specific exceptions at the page level."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)

        except StreamlitRerunNeeded as e:
            logger.debug(f"Streamlit rerun for: {e.message}")
            st.rerun()

        except StreamlitToastMessage as e:
            st.toast(e.message, icon=e.icon)

        except StreamlitErrorMessage as e:
            # Handle dev_mode detail logic
            dev_mode = st.session_state.get("dev_mode", False)
            display_message = (
                f"{e.message}: {e.details}" if e.details and dev_mode else e.message
            )
            logger.debug(f"Streamlit error: {display_message}")

            if e.style == "error":
                st.error(display_message)
            else:
                st.toast(display_message, icon=e.icon)

            if e.stop_execution:
                st.stop()

        except Exception as e:
            # Catch-all for unexpected errors
            dev_mode = st.session_state.get("dev_mode", False)
            error_detail = (
                f"Unexpected error: {str(e)}"
                if dev_mode
                else "An unexpected error occurred"
            )
            logger.debug(f"Unexpected Streamlit error: {error_detail}")
            st.error(error_detail)
            if dev_mode:
                st.exception(e)  # Show full traceback in dev mode

    return wrapper
