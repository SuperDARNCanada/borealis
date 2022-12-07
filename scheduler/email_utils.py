#!/usr/bin/python3

"""
    email_utils.py
    ~~~~~~~~~~~~~~
    Utilites for sending emails via scripts.

    :copyright: 2019 SuperDARN Canada
"""

import email
import smtplib
import email.mime.text
import email.mime.multipart
import email.mime.base

class Emailer(object):
    """
    Utilities used to send logs during scheduling.

    :param  emails: list of emails to send to.
    :type   emails: list
    :param  sender: Name of the sender
    :type   sender: str
    :param  smtp:   SMTP client to send emails from. Going through localhost instead of some known
                    email client.
    :type   smtp:   smtplib.SMTP

    """
    def __init__(self, file_of_emails):
        """
        Initializes the Emailer object.

        :param  file_of_emails: a file containing a list of emails.
        :type   file_of_emails: str
        """
        super(Emailer, self).__init__()

        try:
            with open(file_of_emails, 'r') as f:
                self.emails = f.readlines()
        except:
            self.emails = []



    def email_log(self, subject, log_file, attachments=[]):
        """
        Send a log to the emails.

        :param  subject: Subject line for the log email.
        :type   subject: str
        :param  log_file: File name of the log.
        :type   log_file: str
        :param  attachments: List of paths to email attachments. (Default value = [])
        :type   attachments: list
        """
        try:
            with open(log_file, 'r') as f:
                body = f.read()
        except Exception as e:
            body = "Unable to open log file {} with error:\n{}".format(log_file, str(e))


        self.smtp = smtplib.SMTP('localhost')
        self.sender = "borealis"
        
        em = email.mime.multipart.MIMEMultipart()

        em['subject'] = subject
        em['From'] = self.sender
        em['To'] = ", ".join(self.emails)

        em.attach(email.mime.text.MIMEText(body))

        if attachments:
            for attachment in attachments:
                with open(attachment, 'rb') as f:
                    payload = email.mime.base.MIMEBase('application', 'octet-stream')
                    payload.set_payload(f.read())
                    email.encoders.encode_base64(payload)

                    attachment_header = "attachment; filename={}".format(attachment)
                    payload.add_header('Content-Disposition', attachment_header)
                    em.attach(payload)


        self.smtp.sendmail(self.sender, self.emails, em.as_string())
        self.smtp.quit()
