import logging
from datetime import datetime

from .settings import LOGGING_LEVEL


def create_timestamp() -> str:
    now = datetime.now()
    return now.strftime("%Y_%m_%d %H_%M_%S")


class BaseLogger(logging.Logger):

    def __init__(self, name):
        super().__init__(name, LOGGING_LEVEL)

        self.extra = {}

        # Add handlers (e.g., ConsoleHandler, FileHandler, etc.)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s --- %(levelname)s --- %(name)s --- %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        handler.setLevel(LOGGING_LEVEL)
        self.addHandler(handler)

    def info(self, msg: object, *args, **kwargs):
        if self.isEnabledFor(logging.INFO):
            self._log(logging.INFO, msg, args, extra=self.extra, **kwargs)

    def debug(self, msg: object, *args, **kwargs) -> None:
        if self.isEnabledFor(logging.DEBUG):
            super().debug(logging.DEBUG, msg, args, extra=self.extra, **kwargs)

    def warning(self, msg: object, *args, **kwargs) -> None:
        if self.isEnabledFor(logging.WARNING):
            self._log(logging.WARNING, msg, args, extra=self.extra, **kwargs)
    
    def error(self, msg: object, *args, **kwargs) -> None:
        if self.isEnabledFor(logging.ERROR):
            self._log(logging.ERROR, msg, args, extra=self.extra, **kwargs)

    def exception(self, msg: object, *args, **kwargs) -> None:
        self.error(msg, *args, **kwargs)
