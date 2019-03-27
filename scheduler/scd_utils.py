import logging
import datetime as dt


class SCDUtils(object):
    """docstring for SCDUtils"""
    def __init__(self, scd_filename):
        super(SCDUtils, self).__init__()

        self.scd_filename = scd_filename
        self.scd_dt_fmt = "%H:%M%x"

    def read_scd(self):
        try:
            with open(self.scd_filename, "r") as f:
                raw_scd = f.readlines()
        except:
            #TODO possible error
            pass

        raw_scd = [line.split() for line in raw_scd]

        scd_lines = []
        for line in raw_scd:
            if len(line) != 4:
                #TODO error
                pass

            try:
                time = dt.datetime.strptime(line[0]+line[1], self.scd_dt_fmt)
            except:
                # TODO error
                pass

            prio = int(line[2])
            if not (0 <= prio <= 20):
                pass
                #TODO raise error

            experiment = line[3]

            scd_lines.append({"time" : time, "prio" : prio, "experiment" : experiment})

        return scd_lines


    def add_line(self, hhmm, mmddyy, prio, experiment):

        scd_lines = self.read_scd()
        try:
            # create datetime from args to see if valid
            time = dt.datetime.strptime(hhmm+mmddyy, self.scd_dt_fmt)




