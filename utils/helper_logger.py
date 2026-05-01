import logging
import os

def create_logger(
    name,
    log_file,
    level=logging.DEBUG,
    # format='[%(asctime)s] %(message)s',
    format='%(message)s',
    overwrite=True
):
    # Remove existing handlers for this logger (prevents duplicate logs)
    logger = logging.getLogger(name)
    if logger.handlers:
        for h in logger.handlers[:]:
            logger.removeHandler(h)
            h.close()

    logger.setLevel(level)

    # If overwrite=True, remove the old file first
    if overwrite and os.path.exists(log_file):
        os.remove(log_file)

    # Create a file handler; mode="a" (append) or "w" (overwrite)
    mode = "w" if overwrite else "a"
    file_handler = logging.FileHandler(log_file, mode=mode)
    file_handler.setLevel(level)

    formatter = logging.Formatter(format)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.propagate = False

    return logger

