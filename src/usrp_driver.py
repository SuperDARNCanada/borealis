#!/usr/bin/env python3

"""
    usrp_driver
    ~~~~~~~~~~~
    A python wrapper to the usrp_driver which is used to capture logs and enable python
    controllers and managers to interact with the drivers.

    :copyright: 2023 SuperDARN Canada
    :author: Adam Lozinsky
"""
import os
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

    path = os.environ["BOREALISPATH"]
    cmd = f"source {path}mode {args.run_mode}; {args.c_debug_opts} usrp_driver"
    log.info('usrp_driver start command', command=cmd)
    with subprocess.Popen([cmd], shell=True, bufsize=1, text=True, universal_newlines=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE) as driver:
        # TODO: - the screen is clearing after crash which is not what i want
        #       - messages are buffering on launch
        #       - need to simulate L, U, to see how it logs
        for out, err in zip(iter(driver.stdout.readline, ''), iter(driver.stderr.readline, '')):
            if out is not None:
                log.info(out)
            if err is not None:
                # UHD sends INFO level logs to STDERR, so we need to capture it here.
                if 'INFO' in err:
                    log.info(err)
                else:
                    log.error(err)

        if driver.poll() is not None:
            raise subprocess.CalledProcessError(cmd=cmd, returncode=driver.returncode)


if __name__ == '__main__':
    from utils import log_config

    log = log_config.log()
    log.info(f"USRP_DRIVER BOOTED")
    try:
        main()
        log.info(f"USRP_DRIVER EXITED")
    except Exception as main_exception:
        # print(main_exception)
        log.critical("USRP_DRIVER CRASHED", error=main_exception)
        log.exception("USRP_DRIVER CRASHED", exception=main_exception)