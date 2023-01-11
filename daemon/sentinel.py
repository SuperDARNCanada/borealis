"""
Purposed radar_status module. A daemon for controlling, monitoring, and managing borealis.

This module is named after the large horse that watches over you at the PGR radar site.
"""
import time
import datetime
import logging
import sys
import structlog
import rich  # requires python 3.7+
from rich import pretty
from rich.console import Console
from rich.table import Table
from rich.progress import track
rich.pretty.install()


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
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.TimeStamper(),
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

now = datetime.datetime.utcnow()
logfile_timestamp = now.strftime("%Y.%m.%d.%H:%M")

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(structlog.stdlib.ProcessorFormatter(processor=structlog.dev.ConsoleRenderer(colors=True)))

file_handler = logging.FileHandler(f'testing.log')
file_handler.setFormatter(structlog.stdlib.ProcessorFormatter(processor=structlog.processors.JSONRenderer()))

root_logger = logging.getLogger()
root_logger.addHandler(console_handler)
root_logger.addHandler(file_handler)

root_logger.setLevel('INFO')


console = Console()
table = Table(show_header=True, header_style="bold magenta")

structlog.configure()
log = structlog.getLogger()
log.info(f"hello", key="value", another_key=[1, 2, 3])
log.warning(f"hello", key="value", another_key=[1, 2, 3])
log.error(f"hello", key="value", another_key=[1, 2, 3])

table.add_column("Date", style="dim", width=12)
table.add_column("Title")
table.add_column("Production Budget", justify="right")
table.add_column("Box Office", justify="right")
table.add_row(
    "Dec 20, 2019", "Star Wars: The Rise of Skywalker", "$275,000,000", "$375,126,118"
)
table.add_row(
    "May 25, 2018",
    "[red]Solo[/red]: A Star Wars Story",
    "$275,000,000",
    "$393,151,347",
)
table.add_row(
    "Dec 15, 2017",
    "Star Wars Ep. VIII: The Last Jedi",
    "$262,000,000",
    "[bold]$1,332,539,889[/bold]",
)

console.print(table)

for step in track(range(100)):
    time.sleep(0.05)

tasks = [f"task {n}" for n in range(1, 11)]

with console.status("[bold green]Working on tasks...", spinner='pong') as status:
    while tasks:
        task = tasks.pop(0)
        time.sleep(1)
        console.log(f"{task} complete")

