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
from rich import pretty
rich.pretty.install()


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


def log(log_level=None, console=None, logfile=None, aggregator=None, status=False):
    """
    :param log_level: Logging threshold [CRITICAL, ERROR, WARNING, INFO, DEBUG, NOTSET]
    :type log_level: str
    :param console: Enable (True) or Disable (False) console logging override
    :type console: bool
    :param logfile: Enable (True) or Disable (False) JSON file logging override
    :type logfile: bool
    :param aggregator: Enable (True) or Disable (False) aggregator log forwarding override
    :type aggregator: bool
    :param status: Enable (True) or Disable (False) radar_status log observations
    :type status: bool

    :notes:
    There are three parts to logging; processors, renderers, and handlers.
        processors - modify, add, or clean up the log message dict
        renderers - make the log message a string, json, dict, etc. with fancy styling
        handlers -  print the rendered data to stdout, file, stream, etc.
    """

    # Obtain the module name that imported this log_config
    caller = Path(inspect.stack()[-1].filename)
    module_name = caller.name.split('.')[0]

    # Gather the borealis configuration information
    options = Options()

    # If no override log level is set load the config log level
    if log_level is None:
        log_level = options.log_level
    if console is None:
        console = options.log_console_bool
    if logfile is None:
        logfile = options.log_logfile_bool
        # Set the log file and dir path. The time tag will be appended at the midnight
        # roll over by the TimedRotatingLogHandler.
        log_file = f"{options.log_directory}/{module_name}"
    if aggregator is None:
        aggregator = options.log_aggregator_bool

    # Processors are a list of functions that sequentially modify the event_dict (log message)
    shared_processors = [
        structlog.stdlib.add_logger_name,  # Add the name of the logger to event dict
        structlog.stdlib.add_log_level,  # Add log level to event dict
        structlog.processors.TimeStamper(fmt='iso', utc=True),  # Add ISO-8601 timestamp
        structlog.processors.UnicodeDecoder(),  # Decode byte strings to unicode strings
        ]

    # Configure structlog here once for everything so that every log is uniformly formatted
    # noinspection PyTypeChecker
    structlog.configure(
        processors=shared_processors + [
            structlog.stdlib.filter_by_level,  # Abort pipeline on log levels lower than threshold
            structlog.processors.StackInfoRenderer(),  # Move "stack_info" in event_dict to "stack" and render
            structlog.processors.CallsiteParameterAdder(  # Add items from the call enum
                {
                    structlog.processors.CallsiteParameter.FUNC_NAME,  # function name
                    structlog.processors.CallsiteParameter.MODULE,  # module name
                    structlog.processors.CallsiteParameter.PROCESS,  # process ID
                    # structlog.processors.CallsiteParameter.THREAD,  # thread ID
                    # structlog.processors.CallsiteParameter.FILENAME,  # file name
                    # structlog.processors.CallsiteParameter.LINENO,  # line number

                }),
            # The last processor has to be a renderer to render the log to file, stream, etc. in some style
            # This wrapper lets us decide on the renderer later. This is needed to have two renderers.
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter
        ],
        wrapper_class=structlog.stdlib.BoundLogger,  # Structlog wrapper class to imitate 'logging.Logger`
        logger_factory=structlog.stdlib.LoggerFactory(),  # Creates the wrapped loggers
        cache_logger_on_first_use=True  # Freeze the configuration (no tampering!)
    )

    # Get the logging logger object and attach both handlers
    root_logger = logging.getLogger()
    # Set the logging level that was configured by the start options
    root_logger.setLevel(log_level)

    # Set up the first handler to pipe logs to stdout
    if console:
        console_handler = StreamHandler(sys.stdout)
        console_handler.setFormatter(structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=shared_processors,  # These run on logs that do not come from structlog
            processors=[swap_logger_name,
                        structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                        structlog.dev.ConsoleRenderer(sort_keys=False, colors=True)]))
        root_logger.addHandler(console_handler)

    # Set up the second handler to pipe logs to a JSON file that rotates at midnight
    if logfile:
        logfile_handler = TimedRotatingFileHandler(filename=log_file, when='midnight', utc=True)
        logfile_handler.setFormatter(structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=shared_processors,  # These run on logs that do not come from structlog
            processors=[structlog.processors.TimeStamper(key='unix_timestamp', fmt=None, utc=True),  # Add Unix timestamp
                        structlog.processors.dict_tracebacks,  # Makes tracebacks dict rather than str
                        structlog.stdlib.ProcessorFormatter.remove_processors_meta,  # Removes _records
                        structlog.processors.JSONRenderer(sort_keys=False)]))
        root_logger.addHandler(logfile_handler)
        # Note: the foreign_pre_chain= option can be used to add more processors to just one handler

    # Set up the third handler to pipe logs to the log aggregator (Graylogs). See further logging documentation
    # to set up the log aggregator server and the extractors on the server.
    if aggregator:
        aggregator_handler = graypy.GELFUDPHandler(options.log_aggregator_addr,
                                                   options.log_aggregator_port)
        aggregator_handler.setFormatter(structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=shared_processors,  # These run on logs that do not come from structlog
            processors=[structlog.stdlib.ProcessorFormatter.remove_processors_meta,  # Removes _records
                        structlog.processors.JSONRenderer(sort_keys=False)]))
        root_logger.addHandler(aggregator_handler)

    # Set up the fourth handler to pipe logs through a socket to radar_status
    # TODO (Adam): Finish making this work
    if status:
        status_handler = logging.handlers.SocketHandler(host="localhost", port=10514)
        status_handler.setFormatter(structlog.stdlib.ProcessorFormatter(
            processors=[swap_logger_name,
                        structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                        structlog.dev.ConsoleRenderer(sort_keys=False, pad_event=80, colors=True)]))
        root_logger.addHandler(status_handler)

    # Apply the configuration
    structlog.configure()

    return structlog.getLogger(module_name)
