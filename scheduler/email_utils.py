#!/usr/bin/python3

# Copyright 2019 SuperDARN Canada
#
# email_utils.py
# 2019-04-18
# Utilities for sending emails via scripts.
#
import email
import smtplib
import email.mime.text
import email.mime.multipart
import email.mime.base


class Emailer(object):
    """Utilities used to send logs during scheduling.

    Attributes:
        emails (list): list of emails to send to.
        sender (str): Name of the sender.
        smtp (smtplib.SMTP): SMTP client to send emails from. Going through localhost instead of some known
        email client.
    """
    def __init__(self, file_of_emails):
        """Initializes the Emailer object.

        Args:
            file_of_emails (str): a file containing a list of emails.
        """
        super(Emailer, self).__init__()
        self.smtp = None
        self.sender = None

        try:
            with open(file_of_emails, 'r') as f:
                self.emails = f.readlines()
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

    def email_log(self, subject, log_file, attachments=None):
        """Send a log to the emails.

        Args:
            subject (str): Subject line for the log email.
            log_file (str): File name of the log.
            attachments(list) : List of paths to email attachments. Default None
        """
        try:
            with open(log_file, 'r') as f:
                body = f.read()
        except Exception as e:
            body = f"Unable to open log file {log_file} with error:\n{str(e)}"

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

                    attachment_header = f"attachment; filename={attachment}"
                    payload.add_header('Content-Disposition', attachment_header)
                    em.attach(payload)

        self.smtp.sendmail(self.sender, self.emails, em.as_string())
        self.smtp.quit()
