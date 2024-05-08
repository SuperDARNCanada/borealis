#!/usr/bin/env python3

"""
    usrp_driver
    ~~~~~~~~~~~
    A python wrapper which launches the usrp_driver, captures logs into structlog,
    and enables python controllers and managers to interact with the drivers.

    :copyright: 2023 SuperDARN Canada
    :author: Adam Lozinsky
"""
import os
import subprocess
import faulthandler
import argparse as ap
import pty
import re
import time


def main():
    """
    Given the parsed arguments this runs the USRP driver in a subproccess.
    The subprocess stdout and stderr are streamed through a pty to be read in realtime,
    parsed, then logged using structlog.
    """

    # TODO (Adam): - console should not clear on crash
    #              - debug mode does not work
    faulthandler.enable()
    parser = ap.ArgumentParser(description="Wrapper to the USRP driver")
    parser.add_argument(
        "run_mode",
        help="The mode to run, switches scons builds and some arguments to "
        "modules based on this mode. Commonly 'release'.",
    )
    parser.add_argument(
        "--c_debug_opts", help="A C debug run options string", default=""
    )
    args = parser.parse_args()

    path = os.environ["BOREALISPATH"]
    cmd = f"source {path}/mode {args.run_mode}; {args.c_debug_opts} usrp_driver"
    log.debug("usrp_driver start command", command=cmd)

    # If you are here to work on the code below this comment I bid you good luck!
    # This will only work on LINUX!

    # First we open a pseudoterminal and get the file descriptors for the stdout stream
    stdout_master_fd, stdout_slave_fd = pty.openpty()

    # Then start usrp_driver executable via subprocess.Popen
    # We need text=False and universal_newlines=False to ensure byte mode (default)
    # We need shell=True, even though it is less secure so that we can 'source mode'
    # We use close_fds=True so that the process does not hang open printing empty log lines
    # We need to send stderr to stdout to have one stream, UHD sends logs to stderr, we do stdout
    driver = subprocess.Popen(
        [cmd],
        shell=True,
        close_fds=True,
        stdout=stdout_slave_fd,
        stderr=stdout_slave_fd,
    )
    # Set up a dict to map the UHD log levels to structlog functions
    uhd_log_level = {
        "DEBUG": log.debug,
        "INFO": log.info,
        "WARNING": log.warning,
        "ERROR": log.error,
        "FATAL": log.critical,
    }
    # Compile an ANSI escape character regex stripper
    ansi_escape_regex = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    # driver.poll() checks if the subprocess is still running or has returned an exit code
    while driver.poll() is None:
        # os.read is used because it is non-blocking
        # 1024 bytes means we won't likely cut a printed line midway
        output = os.read(stdout_master_fd, 1024)
        # Switch the output from bytes to a string split by newline,
        # this is needed to ensure rapid messages go to independent logs
        output = output.decode("utf-8").split("\r\n")
        for o in output:
            result = ansi_escape_regex.sub("", o)
            if result == "":
                continue
            else:
                # Split result by enclosed brackets [...] to get log level and device
                # Example: "[INFO] [GPS] No gps lock..." becomes ["", "INFO", " ", "GPS", "No gps lock..."]
                result = re.split("\[|\]", result)
                if len(result) > 1:
                    # Log UHD logs with correct level
                    log_func = uhd_log_level[result[1]]
                    log_func(result[3] + result[4], device=result[3])
                else:
                    # Log our messages and the UHD firmware messages (L, U, D, S, etc)
                    # Some further parsing may be needed in the future to handle our debugs
                    log.info(result[0])
        # Sample the pty every 10 ms
        time.sleep(0.01)
    else:
        # Check if there was a clean exit (returncode=0) or not; if not raise exception
        if driver.returncode != 0:
            raise subprocess.CalledProcessError(cmd=cmd, returncode=driver.returncode)


if __name__ == "__main__":
    from utils import log_config

    log = log_config.log()
    log.info(f"USRP_DRIVER BOOTED")
    try:
        main()
        log.info(f"USRP_DRIVER EXITED")
    except Exception as main_exception:
        log.critical("USRP_DRIVER CRASHED", error=main_exception)
        log.exception("USRP_DRIVER CRASHED", exception=main_exception)
