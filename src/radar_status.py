"""
Okay what is this for?

read logs on control the radar when oopsies happen (daemon)

gather misc system health information and log it (radar_status)

mini aggregator of nice clean logs before sending to reduce volume (radar_status)

console based viewer w/ plotext so we don't have to start 6 segments any more (radar_status)

"""


def statustype():
    errors = ('EXPNEEDED', 'NOERROR', 'WARNING', 'EXITERROR')
    return errors

def errortype():
    return {}  # TODO


class RadarStatus():  # TODO finish the class when we determine how to log and what information
    # from the driver that we would like to pass back to the experiment / user.
    # Suggested information: confirmed ctrfreq's and sampling rate's from the driver
    # third-stage sampling rate (rate of result data)
    """Class to define transmit specifications of a certain frequency, beam, and pulse sequence.
    
    errors = ('EXPNEEDED', 'NOERROR', 'WARNING', 'EXITERROR')
    
    Probably will be phased out once administrator is working
    """

    def __init__(self):
        """
        A RadarStatus is only initiated on startup of radar_control so we need an experiment 
        at this point. 
        """
        self.status = 'EXPNEEDED'# needs a control program.
        self.errorcode = None
        self.logs_for_user = []


    # def warning(self, warning_data):
    #     self.status = 'WARNING'
    #     self.logs_for_user.append(warning_data)
