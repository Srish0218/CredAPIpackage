import zulip

import time

# Initialize the Zulip client once
client = zulip.Client(config_file="zuliprc")


def send_zulip_message(content: str):
    """
    Sends a message to the specified Zulip stream.

    Parameters:
        content (str): The message content.

    Returns:
        dict: The response from Zulip API.
    """
    message = {
        "type": "stream",
        "to": "Speech data trigger update",  # The Zulip stream to send the message to. Defaults to "Speech data
        # trigger update".
        "subject": "Testing Update",  # The message subject.
        "content": content
    }

    try:
        response = client.send_message(message)
    except Exception as e:
        response = {"result": "error", "error": str(e)}

    return response


def reportSuccessMsgBRCP(uid: str, created_on: str):
    """Reports success for BRCP upload."""
    content = f"Cred Brcp data for {uid}_{created_on} uploaded successfully at {time.strftime('%Y-%m-%d %H:%M:%S')}"
    return send_zulip_message(content)


def reportSuccessMsgSoftSkill(date: str):
    """Reports success for SoftSkill upload."""
    content = f"Cred SoftSkill data for {date} uploaded successfully at {time.strftime('%Y-%m-%d %H:%M:%S')}"
    return send_zulip_message(content)


def reportError(error: str):
    """Reports an error message."""
    content = f"Error: {error} occurred at {time.strftime('%Y-%m-%d %H:%M:%S')}"
    return send_zulip_message(content)


def reportTranscriptGenerated(uid: str):
    """Reports that transcripts have been generated."""
    content = f"All transcripts generated for {uid} at {time.strftime('%Y-%m-%d %H:%M:%S')}"
    return send_zulip_message(content)


def reportStatus(status: str, ):
    """Report Current Status"""
    content = f"Current Status: {status}"
    return send_zulip_message(content)
