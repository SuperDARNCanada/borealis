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
