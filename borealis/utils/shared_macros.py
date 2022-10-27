#Copyright 2019 SuperDARN

from datetime import datetime

def COLOR(color, msg):
    """
    @brief      Wraps a string to print with ANSI terminal colors

    @param      color  The color to use
    @param      msg    The message to wrap

    @return     New colored text
    """
    if color == "black":
        return "\033[30m{}\033[0m".format(msg)
    elif color == "red":
        return "\033[31m{}\033[0m".format(msg)
    elif color == "green":
        return "\033[32m{}\033[0m".format(msg)
    elif color == "yellow":
        return "\033[33m{}\033[0m".format(msg)
    elif color == "blue":
        return "\033[34m{}\033[0m".format(msg)
    elif color == "magenta":
        return "\033[35m{}\033[0m".format(msg)
    elif color == "cyan":
        return "\033[36m{}\033[0m".format(msg)
    elif color == "white":
        return "\033[37m{}\033[0m".format(msg)
    else:
        return msg

def MODULE_PRINT(module_name, color):
    """
    @brief      Function generator for a formatted message printer

    @param      module_name  The module name to print from
    @param      color        The color to wrap the module name with.

    @return     New print function to use.
    """
    module_name_upper = module_name.upper()
    colored_name = COLOR(color, module_name_upper + ": ")

    def pprint(msg):
        print('{} - '.format(datetime.utcnow().strftime("%Y%m%d.%H%M%S.%f"))+colored_name+msg)

    return pprint

