def statustype():
    errors = ('EXPNEEDED', 'NOERROR', 'WARNING', 'EXITERROR')
    return errors

def errortype():
    return {}  # REVIEW #6 TODO?


class RadarStatus():  # REVIEW #6 TODO to finish the class?
    """Class to define transmit specifications of a certain frequency, 
    beam, and pulse sequence.
    """

    def __init__(self):
        self.status = 'CPNEEDED'# needs a control program.
        self.errorcode = None
