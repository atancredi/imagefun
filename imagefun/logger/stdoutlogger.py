import logging

RESET = "\033[0m"
COLORS = {
    logging.DEBUG: "\033[90m",
    logging.INFO: "\033[36m",
    logging.WARNING: "\033[33m",
    logging.ERROR: "\033[31m",
    logging.CRITICAL: "\033[1;31m",
}
_STANDARD_ATTRS = set(logging.makeLogRecord({}).__dict__)


class PrettyFormatter(logging.Formatter):
    def format(self, record):
        color = COLORS.get(record.levelno, "")
        level = f"{color}{record.levelname:<8}{RESET}"

        prefix = (
            f"[{self.formatTime(record, '%H:%M:%S')}] "
            f"{level} │ "
            f"{record.filename}:{record.lineno} │ "
        )

        lines = [prefix + record.getMessage()]

        extra = {k: v for k, v in record.__dict__.items() if k not in _STANDARD_ATTRS}

        if extra:
            indent = " " * len(
                f"[{self.formatTime(record, '%H:%M:%S')}] " f"{record.levelname:<8} │ "
            )
            for key, value in extra.items():
                lines.append(f"{indent}{key}: {value}")

        return "\n".join(lines)

