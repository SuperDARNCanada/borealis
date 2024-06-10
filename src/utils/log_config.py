"""
    This config script sets up the `logging` and `structlog` modules for message logging to
    console and file. It simply needs to be imported into any module where logging is needed.
    In order to capture exceptions and crashes main() should be in a try/except block.
    See example.

    :copyright: 2023 SuperDARN Canada
    :author: Adam Lozinsky

    :example:
        if __name__ == '__main__':
            from utils import log_config
            log = log_config.log(log_level='INFO')
            log.info(f"Example info text {[1, 2, 3]}", example_key=[1, 2, 3])
            try:
                main()
            except Exception as exec:
                log.exception("Example crashed", exception=exec)

    :notes:
        Setting up structlog is very tricky, but after it is done it should just work
        (or so they promise). Regardless, the comments herein should help explain how this
        works should we ever need to update it.

        See documentation.
        https://docs.python.org/3/library/logging.handlers.html#timedrotatingfilehandler
        https://www.structlog.org/en/stable/standard-library.html
        https://www.structlog.org/en/stable/processors.html#chains
"""

import inspect
from pathlib import Path
import sys
from .options import Options

# We need these two handlers from logging to print to a file and stdout
import logging
from logging import StreamHandler
from logging.handlers import TimedRotatingFileHandler
import structlog
import graypy

# We need rich to make the console look pretty (Requires Python 3.7+)
import rich

rich.pretty.install()


VERBOSE = logging.INFO - 5  #: New logging level, set between `DEBUG` and `INFO`.


class VerboseLogger(structlog.stdlib.BoundLogger):
    """
    Wrapper class that adds a new logging method `verbose` between `debug` and `info` levels.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        setattr(
            self._logger, "verbose", self._verbose
        )  # Create the new `verbose` method.

    def verbose(self, *args, **kwargs):
        return self._proxy_to_logger("verbose", *args, **kwargs)

    def _verbose(self, msg, *args, **kwargs):
        """Log msg at the `VERBOSE` level"""
        return self._logger._log(VERBOSE, msg, args, **kwargs)


def add_logging_level(level_name, level_num, method_name=None):
    """
    Modified from https://stackoverflow.com/a/35804945

    Comprehensively adds a new logging level to the `logging` module and the
    currently configured logging class.

    `levelName` becomes an attribute of the `logging` module with the value
    `levelNum`. `methodName` becomes a convenience method for both `logging`
    itself and the class returned by `logging.getLoggerClass()` (usually just
    `logging.Logger`). If `methodName` is not specified, `levelName.lower()` is
    used.

    To avoid accidental clobberings of existing attributes, this method will
    raise an `AttributeError` if the level name is already an attribute of the
    `logging` module or if the method name is already present

    Example
    -------
    >>> add_logging_level('TRACE', logging.DEBUG - 5)
    >>> logging.getLogger(__name__).setLevel("TRACE")
    >>> logging.getLogger(__name__).trace('that worked')
    >>> logging.trace('so did this')
    >>> logging.TRACE
    5

    """
    if not method_name:
        method_name = level_name.lower()

    if hasattr(logging, level_name):
        raise AttributeError(f"{level_name} already defined in logging module")
    if hasattr(logging, method_name):
        raise AttributeError(f"{method_name} already defined in logging module")
    if hasattr(logging.getLoggerClass(), method_name):
        raise AttributeError(f"{method_name} already defined in logger class")

    logging.addLevelName(level_num, level_name.upper())


def swap_logger_name(_, __, event_dict):
    """
    Swaps the kw 'logger_name' value with the 'module_name' and 'func_name' values then removes them
    from the 'event_dict' for nicer representation. This is done to hack ConsoleRenderer to somewhat
    match our past format. This is intended only to be used for console rendering and not
    JSON rendering (prints nice but does not appear in file).
    """
    event_dict["logger"] = event_dict["module"] + " " + event_dict["func_name"]
    del event_dict["module"]
    del event_dict["func_name"]

    return event_dict


def format_floats(_, __, event_dict):
    """
    Truncate all floating-point field to three decimal places. This is done only for ConsoleRenderer
    to reduce the size of logs in the screen when running the radar.
    """
    for k, v in event_dict.items():
        if isinstance(v, float):
            event_dict[k] = f"{v:.3f}"
    return event_dict


def log(
    console_log_level=None,
    logfile_log_level=None,
    aggregator_log_level=None,
    console=None,
    logfile=None,
    aggregator=None,
    json_to_console=False,
):
    """
    :param console_log_level: Logging threshold for console renderer [CRITICAL, ERROR, WARNING, INFO, VERBOSE, DEBUG, NOTSET]
    :type console_log_level: str | int
    :param logfile_log_level: Logging threshold for logfile renderer [CRITICAL, ERROR, WARNING, INFO, VERBOSE, DEBUG, NOTSET]
    :type logfile_log_level: str | int
    :param aggregator_log_level: Logging threshold for aggregator renderer [CRITICAL, ERROR, WARNING, VERBOSE, INFO, DEBUG, NOTSET]
    :type aggregator_log_level: str | int
    :param console: Enable (True) or Disable (False) console logging override
    :type console: bool
    :param logfile: Enable (True) or Disable (False) JSON file logging override
    :type logfile: bool
    :param aggregator: Enable (True) or Disable (False) aggregator log forwarding override
    :type aggregator: bool
    :param json_to_console: Enable (True) or Disable (False) a logger that converts JSON logs to console-style
    :type json_to_console: bool

    :notes:
    There are three parts to logging; processors, renderers, and handlers.
        processors - modify, add, or clean up the log message dict
        renderers - make the log message a string, json, dict, etc. with fancy styling
        handlers -  print the rendered data to stdout, file, stream, etc.
    """
    if json_to_console and (console or logfile or aggregator):
        raise RuntimeError("Cannot convert a JSON logfile while simultaneously logging")

    # Ensure that add_logging_level is only called once
    if "verbose" not in vars(logging):
        # Create a new logging level in between DEBUG and INFO
        add_logging_level("verbose", logging.INFO - 5)

    # Obtain the module name that imported this log_config
    caller = Path(inspect.stack()[-1].filename)
    module_name = caller.name.split(".")[0]

    # Gather the borealis configuration information
    options = Options()

    # If no override log level is set load the config log level
    if console_log_level is None:
        console_log_level = options.console_log_level
    if logfile_log_level is None:
        logfile_log_level = options.logfile_log_level
    if aggregator_log_level is None:
        aggregator_log_level = options.aggregator_log_level

    if console is None:
        console = options.log_console_bool
    if logfile is None:
        logfile = options.log_logfile_bool
    if aggregator is None:
        aggregator = options.log_aggregator_bool

    # Processors are a list of functions that sequentially modify the event_dict (log message)
    shared_processors = [
        structlog.stdlib.add_logger_name,  # Add the name of the logger to event dict
        structlog.stdlib.add_log_level,  # Add log level to event dict
        structlog.stdlib.add_log_level_number,  # Add numeric log level to event dict
        structlog.processors.TimeStamper(fmt="iso", utc=True),  # Add ISO-8601 timestamp
        structlog.processors.UnicodeDecoder(),  # Decode byte strings to unicode strings
    ]

    processors = [
        structlog.stdlib.add_logger_name,  # Add the name of the logger
        structlog.stdlib.add_log_level,  # Add log level to event dict
        structlog.processors.TimeStamper(
            fmt="iso", utc=True
        ),  # Add timestamps to the log
        structlog.processors.UnicodeDecoder(),  # Decode byte strings to unicode strings
        structlog.processors.StackInfoRenderer(),  # Move "stack_info" in event_dict to "stack" and render
        structlog.processors.CallsiteParameterAdder(  # Add items from the call enum
            {
                structlog.processors.CallsiteParameter.FUNC_NAME,  # function name
                structlog.processors.CallsiteParameter.MODULE,  # module name
                structlog.processors.CallsiteParameter.PROCESS,  # process ID
                # structlog.processors.CallsiteParameter.THREAD,  # thread ID
                # structlog.processors.CallsiteParameter.FILENAME,  # file name
                # structlog.processors.CallsiteParameter.LINENO,  # line number
            },
            # Ignore any function from this module when attributing a log to a function
            additional_ignores=["utils.log_config", "src.utils.log_config"],
        ),
        # The last processor has to be a renderer to render the log to file, stream, etc. in some style
        # This wrapper lets us decide on the renderer later. This is needed to have two renderers.
        structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
    ]

    # Remove some processors when simply converting a JSON logfile to console rendering
    if json_to_console:
        processors.pop(-2)  # CallsiteParameterAdder
        processors.pop(2)  # Timestamper
        processors.pop(0)  # add_logger_name

    # Configure structlog here once for everything so that every log is uniformly formatted
    # noinspection PyTypeChecker
    structlog.configure(
        processors=processors,
        wrapper_class=VerboseLogger,  # Structlog wrapper class to imitate `logging.Logger`
        logger_factory=structlog.stdlib.LoggerFactory(),  # Creates the wrapped loggers
        cache_logger_on_first_use=True,  # Freeze the configuration (no tampering!)
    )

    # Get the logging logger object and attach both handlers
    root_logger = logging.getLogger()
    # Set the logging level that was configured by the start options
    root_logger.setLevel(logging.NOTSET)

    # Set up the first handler to pipe logs to stdout
    if console:
        console_handler = StreamHandler(sys.stdout)
        console_handler.setLevel(console_log_level)
        styles = structlog.dev.ConsoleRenderer.get_default_level_styles(colors=True)
        # Use the info style when logging a verbose message
        styles["verbose"] = styles["info"]
        console_handler.setFormatter(
            structlog.stdlib.ProcessorFormatter(
                foreign_pre_chain=shared_processors,  # These run on logs that do not come from structlog
                processors=[
                    swap_logger_name,
                    structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                    format_floats,
                    structlog.dev.ConsoleRenderer(
                        sort_keys=False, colors=True, level_styles=styles
                    ),
                ],
            )
        )
        root_logger.addHandler(console_handler)

    # Set up the second handler to pipe logs to a JSON file that rotates at midnight
    if logfile:
        # Set the log file and dir path. The time tag will be appended at the midnight
        # roll over by the TimedRotatingLogHandler.
        log_file = f"{options.log_directory}/{module_name}"
        logfile_handler = TimedRotatingFileHandler(
            filename=log_file, when="midnight", utc=True
        )
        logfile_handler.setLevel(logfile_log_level)
        logfile_handler.setFormatter(
            structlog.stdlib.ProcessorFormatter(
                foreign_pre_chain=shared_processors,  # These run on logs that do not come from structlog
                processors=[
                    structlog.processors.TimeStamper(
                        key="unix_timestamp", fmt=None, utc=True
                    ),
                    # Add Unix timestamp
                    structlog.processors.dict_tracebacks,  # Makes tracebacks dict rather than str
                    structlog.stdlib.ProcessorFormatter.remove_processors_meta,  # Removes _records
                    structlog.processors.JSONRenderer(sort_keys=False),
                ],
            )
        )
        root_logger.addHandler(logfile_handler)
        # Note: the foreign_pre_chain= option can be used to add more processors to just one handler

    # Set up the third handler to pipe logs to the log aggregator (Graylogs). See further logging documentation
    # to set up the log aggregator server and the extractors on the server.
    if aggregator:
        aggregator_handler = graypy.GELFUDPHandler(
            options.log_aggregator_addr, options.log_aggregator_port
        )
        aggregator_handler.setLevel(aggregator_log_level)
        aggregator_handler.setFormatter(
            structlog.stdlib.ProcessorFormatter(
                foreign_pre_chain=shared_processors,  # These run on logs that do not come from structlog
                processors=[
                    structlog.stdlib.ProcessorFormatter.remove_processors_meta,  # Removes _records
                    structlog.processors.JSONRenderer(sort_keys=False),
                ],
            )
        )
        root_logger.addHandler(aggregator_handler)

    if json_to_console:
        json_to_console_handler = StreamHandler(sys.stdout)
        styles = structlog.dev.ConsoleRenderer.get_default_level_styles(colors=True)
        # Use the info style when logging a verbose message
        styles["verbose"] = styles["info"]
        json_to_console_handler.setFormatter(
            structlog.stdlib.ProcessorFormatter(
                processors=[
                    structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                    format_floats,
                    structlog.dev.ConsoleRenderer(
                        sort_keys=False, colors=True, level_styles=styles
                    ),
                ]
            )
        )
        root_logger.addHandler(json_to_console_handler)

    # Create a local logger for performance reasons (see https://www.structlog.org/en/stable/performance.html)
    logger = structlog.get_logger(module_name).new()
    return logger
