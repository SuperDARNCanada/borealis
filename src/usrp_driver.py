#!/usr/bin/env python3

"""
    usrp_driver
    ~~~~~~~~~~~
    A python wrapper to the usrp_driver which is used to capture logs and enable python
    controllers and managers to interact with the drivers.

    :copyright: 2023 SuperDARN Canada
    :author: Adam Lozinsky
"""

import subprocess
import faulthandler
import argparse as ap


def main():
    faulthandler.enable()
    parser = ap.ArgumentParser(description='Wrapper to the USRP driver')
    parser.add_argument("run_mode", help="The mode to run, switches scons builds and some arguments to "
                                         "modules based on this mode. Commonly 'release'.")
    parser.add_argument('--c_debug_opts', help='A C debug run options string',
                        default='')
    args = parser.parse_args()

    cmd = f"source mode {args.run_mode}; {args.c_debug_opts} usrp_driver"
    with subprocess.Popen([cmd], bufsize=1, universal_newlines=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE) as driver:
        for out, err in zip(iter(driver.stdout.readline, b''), iter(driver.stderr.readline, b'')):
            if out is not None:
                log.info(out)
            if err is not None:
                log.error(err)

        if driver.poll() is not None:
            log.exception("usrp error", err=subprocess.CalledProcessError(cmd=cmd, returncode=driver.returncode))
            raise


if __name__ == '__main__':
    from utils import log_config

    log = log_config.log(logfile=False, aggregator=False)
    log.info(f"USRP_DRIVER BOOTED")
    try:
        main()
    except Exception as main_exception:
        log.critical("USRP_DRIVER CRASHED", error=main_exception)
        log.exception("USRP_DRIVER CRASHED", exception=main_exception)
