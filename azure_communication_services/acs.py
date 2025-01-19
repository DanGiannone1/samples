from azure.communication.email import EmailClient
import os
from datetime import datetime

def create_email_message(sender: str, recipient: str, subject: str, message: str) -> dict:
    """
    Create an email message with basic HTML formatting
    
    Args:
        sender (str): Sender email address
        recipient (str): Recipient email address
        subject (str): Email subject
        message (str): Email message content
    
    Returns:
        dict: Formatted email message
    """
    return {
        "senderAddress": sender,
        "recipients": {
            "to": [{"address": recipient}]
        },
        "content": {
            "subject": f"[Demo] {subject}",
            "plainText": message,
            "html": f"""
            <html>
                <body style="font-family: sans-serif; padding: 20px;">
                    <div style="max-width: 600px; margin: 0 auto;">
                        <h2>{subject}</h2>
                        <div style="padding: 15px; background-color: #f8f9fa;">
                            {message}
                        </div>
                        <div style="margin-top: 20px; color: #666; font-size: 14px;">
                            Sent via Azure Communication Services Demo<br>
                            {datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")}
                        </div>
                    </div>
                </body>
            </html>
            """
        }
    }

def send_email(email_client: EmailClient, sender: str, recipient: str, 
               subject: str, message: str) -> dict:
    """
    Send an email using Azure Communication Services
    
    Args:
        email_client (EmailClient): Initialized Azure Communication Services email client
        sender (str): Sender email address
        recipient (str): Recipient email address
        subject (str): Email subject
        message (str): Email message content
    
    Returns:
        dict: Result of the email sending operation
    """
    try:
        # Create email message
        email_message = create_email_message(sender, recipient, subject, message)
        
        # Send email
        poller = email_client.begin_send(email_message)
        result = poller.result()
        
        print(f"Email sent successfully to {recipient}")
        print(f"Message ID: {result.message_id}")
        
        return {
            "status": "success",
            "message_id": result.message_id,
            "recipient": recipient,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        error_msg = f"Failed to send email: {str(e)}"
        print(f"Error: {error_msg}")
        return {
            "status": "error",
            "error": error_msg,
            "recipient": recipient,
            "timestamp": datetime.now().isoformat()
        }

def main():
    """Main function to demonstrate Azure Communication Services email functionality"""
    try:
        # Get environment variables
        connection_string = os.environ.get("COMMUNICATION_SERVICES_CONNECTION_STRING")
        sender_email = os.environ.get("SENDER_EMAIL")
        
        if not connection_string or not sender_email:
            raise ValueError("Missing required environment variables")
        
        # Initialize email client
        email_client = EmailClient.from_connection_string(connection_string)
        
        # Example message
        recipient = "test@example.com"  # Replace with actual recipient
        subject = "Test Email"
        message = """
        This is a test message sent via Azure Communication Services.
        It demonstrates basic HTML email formatting and error handling.
        """
        
        # Send test email
        result = send_email(
            email_client=email_client,
            sender=sender_email,
            recipient=recipient,
            subject=subject,
            message=message
        )
        
        print("\nSend result:", result)
        
    except Exception as e:
        print(f"Demo failed: {str(e)}")

if __name__ == "__main__":
    main()