import logging
import queue


class UILogHandler(logging.Handler):
    """
    Logging handler that sends log records to a UI queue.
    """

    def __init__(self, log_queue: queue.Queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        try:
            message = self.format(record)

            self.log_queue.put(
                message
            )

        except Exception:
            self.handleError(record)