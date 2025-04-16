import logging

def _log_message(message: str, log_path: str, message_level : str ) -> None:
    """
    Logs details to log file.

    Args:
        log_path: path of the log file
        message_level: "error" or "info". Default level is info

    Returns:

    """
    import datetime
    time_prefix = datetime.datetime.now().astimezone().strftime("%m/%d/%Y %H:%M:%S") + ' : '
    message = time_prefix + message

    if message_level.strip().lower() == "error":
        logging.error(message)
    else:
        logging.info(message)

    try:
        with open(log_path, 'a') as f:
            f.write(f"{message}\n")
            f.flush()  # Ensure data is written to the file
    except IOError as e:
        logging.error(f"Failed to write to log file {log_path}: {e}")
