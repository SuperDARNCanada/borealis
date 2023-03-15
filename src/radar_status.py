"""
TODO (Adam):
    read logs on control the radar when oopsies happen (daemon)
    gather misc system health information and log it (radar_status)
    mini aggregator of nice clean logs before sending to reduce volume (radar_status)
    console based viewer w/ plotext so we don't have to start 6 segments any more (radar_status)
"""

import time
import subprocess
import json


def execute_cmd(cmd):
    try:
        output = subprocess.check_output(cmd, shell=True)
    except subprocess.CalledProcessError as err:
        output = {'cmd_error': err.output}
    return output.decode('utf-8')


def inxi_cli():
    """
    General system health information:
    """
    cmd = f"inxi --tty --no-sudo --swap --disk --info --processes --sensors --output json --output-file print"
    msg = execute_cmd(cmd)
    msg = json.loads(msg)
    log.info("system health", **msg)

    return


def main():
    while True:
        inxi_cli()
        time.sleep(5)


if __name__ == '__main__':
    from utils import log_config

    log = log_config.log(logfile=False, aggregator=False)
    log.info(f"RADAR_STATUS BOOTED")
    try:
        main()
    except Exception as main_exception:
        log.critical("RADAR_STATUS CRASHED", error=main_exception)
        log.exception("RADAR_STATUS CRASHED", exception=main_exception)
