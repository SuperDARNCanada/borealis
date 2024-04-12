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
        super().__init__()
        self.smtp = smtplib.SMTP('localhost')
        self.sender = "borealis"

        try:
            with open(file_of_emails, 'r') as emails_file:
                self.emails = emails_file.readlines()
        except OSError as err:
            # File can't be opened
            self.emails = []
            print(f"OSError opening emails text file: {err}")
        except ValueError as err:
            # Encoding error
            self.emails = []
            print(f"ValueError opening emails text file: {err}")
        except Exception as err:
            # Unknown error
            self.emails = []
            print(f"Error opening emails text file: {err}")

        if not self.emails:
            raise ValueError("No email addresses to send to")

    def email_log(self, subject, log_filename, attachments=None):
        """
        Send a log to the emails.

        :param  subject: Subject line for the log email.
        :type   subject: str
        :param  log_filename: File name of the log.
        :type   log_filename: str
        :param  attachments: List of paths to email attachments. Default None
        :type   attachments: list
        """
        try:
            with open(log_filename, 'r') as log_file:
                body = log_file.read()
        except Exception as err:
            body = f"Unable to open log file {log_filename} with error:\n{str(err)}"

        em = email.mime.multipart.MIMEMultipart()

        em['subject'] = str(subject)
        em['From'] = self.sender
        em['To'] = ", ".join(self.emails)

        em.attach(email.mime.text.MIMEText(body))
        
        # Print the email contents to the console
        print(f"From: {self.sender}")
        print(f"To: {', '.join(self.emails)}")
        print(f"Subject: {str(subject)}")
        print(body)

        if attachments:
            for attachment in attachments:
                with open(attachment, 'rb') as f:
                    payload = email.mime.base.MIMEBase('application', 'octet-stream')
                    payload.set_payload(f.read())
                    email.encoders.encode_base64(payload)

                    attachment_header = f"attachment; filename={attachment}"
                    payload.add_header('Content-Disposition', attachment_header)
                    em.attach(payload)

        self.smtp.sendmail(self.sender, self.emails, em.as_string())
        self.smtp.quit()
