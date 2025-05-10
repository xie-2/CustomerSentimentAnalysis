# Standard library imports
import os
import email
from email.header import decode_header

# Third-party imports
import pandas as pd
from imapclient import IMAPClient
from dotenv import load_dotenv

# Load credentials from .env
load_dotenv()
IMAP_SERVER = os.getenv('IMAP_SERVER', 'imap.gmail.com')  # Default to Gmail
EMAIL_USER = os.getenv('EMAIL_USER')
EMAIL_PASS = os.getenv('EMAIL_PASS')
MAILBOX = 'INBOX'

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# Debug prints
print(f"IMAP_SERVER: {IMAP_SERVER}")
print(f"EMAIL_USER: {EMAIL_USER}")
print(f"EMAIL_PASS: {'*' * len(EMAIL_PASS) if EMAIL_PASS else None}")


def fetch_emails(limit=100):
    """Connect to IMAP server and fetch email bodies."""
    with IMAPClient(IMAP_SERVER) as client:
        client.login(EMAIL_USER, EMAIL_PASS)
        client.select_folder(MAILBOX)
        messages = client.search(['NOT', 'DELETED'])[:limit]
        records = []
        for uid, message_data in client.fetch(messages, 'RFC822').items():
            msg = email.message_from_bytes(message_data[b'RFC822'])
            subject = msg['Subject']
            if subject:
                subject, encoding = decode_header(subject)[0]
                if isinstance(subject, bytes):
                    subject = subject.decode(encoding or 'utf-8', errors='ignore')
            else:
                subject = '(No Subject)'
            body = ''
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == 'text/plain':
                        body += part.get_payload(decode=True).decode(errors='ignore')
            else:
                body = msg.get_payload(decode=True).decode(errors='ignore')
            records.append({'uid': uid, 'subject': subject, 'body': body})
        df = pd.DataFrame(records)
        df.to_csv('data/raw_emails.csv', index=False)
        print(f"Fetched {len(df)} emails to data/raw_emails.csv")


if __name__ == '__main__':
    fetch_emails()