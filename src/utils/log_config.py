"""
I am structuring this config file so that it simply needs to be imported to any module
where we want logging.
"""
import os
import json
import inspect
from pathlib import Path
import datetime
import logging
from logging.handlers import TimedRotatingFileHandler
import sys
import structlog
import rich  # requires python 3.7+
from rich import pretty
rich.pretty.install()


def log(log_level='INFO'):
    """
    Logging Levels = [CRITICAL, ERROR, WARNING, INFO, DEBUG, NOTSET]
    """

    # Obtain the module name that imported this log_config
    if __name__ != '__main__':
        caller = Path(inspect.stack()[-1].filename)
        module_name = caller.name.split('.')[0]

    # Gather the configuration information.
    config_path = os.environ["BOREALISPATH"] + "/config.ini"
    try:
        with open(config_path, 'r') as config_data:
            raw_config = json.load(config_data)
    except IOError:
        errmsg = f'Cannot open config file at {config_path}'
        raise IOError(errmsg)

    # Determine to the log file and path to write logs to
    logfile_timestamp = datetime.datetime.utcnow().strftime("%Y.%m.%d")
    log_file = f"{raw_config['log_directory']}/{logfile_timestamp}-{module_name}.log"
    log_file = f"{raw_config['log_directory']}/{module_name}"

    structlog.configure(
        processors=[
            # If log level is too low, abort pipeline and throw away log entry.
            structlog.stdlib.filter_by_level,
            # Add the name of the logger to event dict.
            structlog.stdlib.add_logger_name,
            # Add log level to event dict.
            structlog.stdlib.add_log_level,
            # Perform %-style formatting.
            # structlog.stdlib.PositionalArgumentsFormatter(),
            # Add a timestamp in ISO 8601 format.
            structlog.processors.TimeStamper(fmt='iso', utc=True),
            structlog.processors.TimeStamper(key='unix_timestamp', fmt=None, utc=True),
            # If the "stack_info" key in the event dict is true, remove it and
            # render the current stack trace in the "stack" key.
            structlog.processors.StackInfoRenderer(),
            # If the "exc_info" key in the event dict is either true or a
            # sys.exc_info() tuple, remove "exc_info" and render the exception
            # with traceback into the "exception" key.
            structlog.processors.format_exc_info,
            # If some value is in bytes, decode it to a unicode str.
            structlog.processors.UnicodeDecoder(),
            # Add callsite parameters.
            structlog.processors.CallsiteParameterAdder(
                {
                    structlog.processors.CallsiteParameter.FILENAME,
                    structlog.processors.CallsiteParameter.FUNC_NAME,
                    structlog.processors.CallsiteParameter.LINENO,
                }
            ),
            # Render the final event dict as JSON.
            # structlog.processors.JSONRenderer()
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter
        ],
        # `wrapper_class` is the bound logger that you get back from
        # get_logger(). This one imitates the API of `logging.Logger`.
        wrapper_class=structlog.stdlib.BoundLogger,
        # `logger_factory` is used to create wrapped loggers that are used for
        # OUTPUT. This one returns a `logging.Logger`. The final value (a JSON
        # string) from the final processor (`JSONRenderer`) will be passed to
        # the method of the same name as that you've called on the bound logger.
        logger_factory=structlog.stdlib.LoggerFactory(),
        # Effectively freeze configuration after creating the first bound
        # logger.
        cache_logger_on_first_use=True,
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(structlog.stdlib.ProcessorFormatter(
        processor=structlog.dev.ConsoleRenderer(colors=True),
        keep_exc_info=True,
        keep_stack_info=True,
    ))

    file_handler = TimedRotatingFileHandler(filename=log_file, when='midnight', utc=True)
    file_handler.setFormatter(structlog.stdlib.ProcessorFormatter(
        processor=structlog.processors.JSONRenderer(),
        keep_exc_info=True,
        keep_stack_info=True,
    ))

    root_logger = logging.getLogger()
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    root_logger.setLevel(log_level)

    structlog.configure()

    return structlog.getLogger(module_name)
