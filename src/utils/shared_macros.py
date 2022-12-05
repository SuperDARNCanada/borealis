#Copyright 2019 SuperDARN

from datetime import datetime

def COLOR(color, msg):
    """
    Wraps a string to print with ANSI terminal colors

    :param      color:  The color to use
    :type       color:  str
    :param      msg:    The message to wrap
    :type       msg:    str

    :return:    New colored text
    :rtype:     str
    """
    if color == "black":
        return f"\x1b[30m{msg}\x1b[0m"
    elif color == "red":
        return f"\x1b[31m{msg}\x1b[0m"
    elif color == "green":
        return f"\x1b[32m{msg}\x1b[0m"
    elif color == "yellow":
        return f"\x1b[33m{msg}\x1b[0m"
    elif color == "blue":
        return f"\x1b[34m{msg}\x1b[0m"
    elif color == "magenta":
        return f"\x1b[35m{msg}\x1b[0m"
    elif color == "cyan":
        return f"\x1b[36m{msg}\x1b[0m"
    elif color == "white":
        return f"\x1b[37m{msg}\x1b[0m"
    else:
        return msg

def MODULE_PRINT(module_name, color):
    """
    Function generator for a formatted message printer

    :param      module_name:    The module name to print from
    :type       module_name:    str
    :param      color:          The color to wrap the module name with.
    :type       color:          str

    :return:    New print function to use.
    :rtype:     function
    """
    module_name_upper = module_name.upper()
    colored_name = COLOR(color, module_name_upper + ": ")

    def pprint(msg):
        print(f'{datetime.utcnow().strftime("%Y%m%d.%H%M%S.%f")} - {colored_name}{msg}')

    return pprint

