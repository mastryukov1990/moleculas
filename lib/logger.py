import logging

from colorama import Fore


class MultiLevelFormatter(logging.Formatter):
    def __init__(self, info_fmt: str, warning_fmt: str, error_fmt: str):
        super().__init__(fmt='%(levelno)d: %(msg)s', datefmt=None, style='%')
        self.info_fmt = info_fmt
        self.warning_fmt = warning_fmt
        self.error_fmt = error_fmt

    def format(self, record):

        # Save the original format configured by the user
        # when the logger formatter was instantiated
        format_orig = self._style._fmt

        # Replace the original format with one customized by logging level
        if record.levelno == logging.INFO:
            self._style._fmt = self.info_fmt

        elif record.levelno == logging.WARNING:
            self._style._fmt = self.info_fmt

        elif record.levelno == logging.ERROR:
            self._style._fmt = self.error_fmt

        result = logging.Formatter.format(self, record)
        self._style._fmt = format_orig

        return result


def configure_logger(logger_name: str, logger_level=logging.INFO, log_dir: str = '/tmp'):
    logger = logging.getLogger(logger_name)
    if len(logger.handlers):
        return logger
    logger.setLevel(logger_level)
    formatter = MultiLevelFormatter(
        info_fmt=f'{Fore.LIGHTBLUE_EX}%(asctime)s{Fore.GREEN}  %(message)s{Fore.RESET}',
        warning_fmt=f'{Fore.LIGHTBLUE_EX}%(asctime)s{Fore.YELLOW}  %(message)s{Fore.RESET}',
        error_fmt=f'{Fore.LIGHTBLUE_EX}%(asctime)s{Fore.RED}  %(message)s{Fore.RESET}',
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logger_level)
    stream_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)

    return logger


class Logger:

    def __init__(self, name: str):
        self.logger = configure_logger(name)

    def info(self, text: str):
        self.logger.info(text)
