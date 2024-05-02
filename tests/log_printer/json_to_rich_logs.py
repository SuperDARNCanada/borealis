"""
    json_to_rich_logs
    ~~~~~~~~~~~~~~~~~
    A script for converting JSON-rendered Borealis logfiles to console-rendered logfiles.
    This script will replicate the logs that display in screens when steamed_hams.py is run.
"""

import argparse
import os
import sys

import ijson


def add_item(container, k, v):
    if isinstance(container, dict):
        key = k.split(".")[-1]
        container[key] = v
    elif isinstance(container, list):
        container.append(v)
    else:
        raise RuntimeError(
            f"Unable to add to current object: unsupported type {type(nest_list[-1])}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("json_file", type=str, help="Path to JSON log file")
    args = parser.parse_args()

    sys.path.append(os.environ["BOREALISPATH"])
    from src.utils import log_config

    log = log_config.log(
        console=False,
        logfile=False,
        aggregator=False,
        json_to_console=True,
    )

    with open(args.json_file, "rb") as stream:
        json_parser = ijson.parse(stream, use_float=True, multiple_values=True)

        nest_list = []
        prefix_list = []
        for prefix, event, value in json_parser:
            if event == "start_map":
                nest_list.append({})
            elif event == "end_map":
                if len(nest_list) == 0:
                    raise RuntimeError("JSON closing } found without opening {")
                current_entry = nest_list.pop()
                if len(nest_list) > 0:
                    add_item(nest_list[-1], prefix, current_entry)
                else:
                    level = current_entry.pop("level", None)
                    msg = current_entry.pop("event", "")
                    if level == "info":
                        log.info(msg, **current_entry)
                    elif level == "verbose":
                        log.verbose(msg, **current_entry)
                    elif level == "debug":
                        log.debug(msg, **current_entry)
                    elif level == "warning":
                        log.warning(msg, **current_entry)
                    elif level == "critical":
                        log.critical(msg, **current_entry)
                    else:
                        raise ValueError(f"Unknown log level: {level}")
            elif event == "start_array":
                nest_list.append([])
            elif event == "end_array":
                current_entry = nest_list.pop()
                nest_list[-1][prefix] = current_entry
            elif event not in ["", "map_key"] and prefix != "":
                if len(nest_list) == 0:
                    add_item(nest_list, prefix, value)
                else:
                    add_item(nest_list[-1], prefix, value)
