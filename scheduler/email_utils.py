import email
import smtplib


class Emailer(object):
    """Utilities used to send logs during scheduling.

    Attributes:
        emails (TYPE): list of emails to send to.
        sender (str): Name of the sender.
        smtp (TYPE): SMTP client to send emails from. Going through localhost instead of some known
        email client.
    """
    def __init__(self, file_of_emails):
        super(Emailer, self).__init__()

        with open(file_of_emails, 'r') as f:
            self.emails = f.readlines()

        self.smtp = smtplib.SMTP('localhost')
        self.sender = "borealis"


    def email_log(self, subject, log_file):
        """Send a log to the emails.

        Args:
            subject (TYPE): Subject line for the log email.
            log_file (TYPE): File name of the log.
        """
        try:
            with open(log_file, 'r') as f:
                body = f.read()
        except Exception as e:
            body = "Unable to open log file {} with error:\n{}".format(log_file, str(e))


        em = email.mime.text.MIMEText(body)
        em['subject'] = subject
        em['From'] = self.sender
        em['To'] = ", ".join(self.emails)

        self.smtp.sendmail(self.sender, self.emails, em.as_string())