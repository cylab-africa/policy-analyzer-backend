"""
Email service for sending batch query notifications.

This module handles email sending using Python's smtplib with support for:
- HTML and plain text emails
- Batch completion notifications
- PDF download links
"""

import html as html_lib
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional, Dict, Any
from datetime import datetime

from app.core.config import settings


class EmailService:
    """Service for sending email notifications."""

    def __init__(self):
        self.smtp_server = settings.SMTP_SERVER
        self.smtp_port = settings.SMTP_PORT
        self.smtp_username = settings.SMTP_USERNAME
        self.smtp_password = settings.SMTP_PASSWORD
        # Send from the account that logs in unless another address is configured
        self.sender_email = settings.SENDER_EMAIL or settings.SMTP_USERNAME
        self.sender_name = settings.SENDER_NAME

    def _create_connection(self):
        try:
            if self.smtp_port == 465:
                import ssl
                context = ssl.create_default_context()
                server = smtplib.SMTP_SSL(
                    self.smtp_server,
                    self.smtp_port,
                    timeout=30,
                    context=context
                )
                server.set_debuglevel(0)  # Set to 1 for debugging
                server.ehlo()
            else:
                server = smtplib.SMTP(self.smtp_server, self.smtp_port, timeout=30)
                server.set_debuglevel(0)  
                server.ehlo()
                if self.smtp_port == 587:
                    server.starttls()
                    server.ehlo()  
            if self.smtp_username and self.smtp_password:
                server.login(self.smtp_username, self.smtp_password)

            return server
        except smtplib.SMTPAuthenticationError as e:
            print(f"SMTP Authentication Error: {e}")
            print(f"Please check SMTP_USERNAME and SMTP_PASSWORD in your .env file")
            print(f"For Gmail, you need an App Password, not your regular password")
            print(f"Generate one at: https://myaccount.google.com/apppasswords")
            raise
        except smtplib.SMTPConnectError as e:
            print(f"SMTP Connection Error: {e}")
            print(f"Could not connect to {self.smtp_server}:{self.smtp_port}")
            raise
        except smtplib.SMTPServerDisconnected as e:
            print(f"SMTP Server Disconnected: {e}")
            print(f"The server unexpectedly closed the connection")
            print(f"This often happens with incorrect port configuration:")
            print(f"  - Port 465: Use for SSL connections")
            print(f"  - Port 587: Use for STARTTLS connections")
            raise
        except Exception as e:
            print(f"Error connecting to SMTP server: {e}")
            raise

    def _format_plain_text_email(
        self,
        recipient_email: str,
        job_summary: Dict[str, Any],
        pdf_download_url: Optional[str] = None
    ) -> str:
        """Format plain text version of batch completion email."""
        job_id = job_summary.get('job_id', 'N/A')
        total = job_summary.get('total_queries', 0)
        completed = job_summary.get('completed_queries', 0)
        failed = job_summary.get('failed_queries', 0)
        success_rate = job_summary.get('success_rate', 0)
        processing_time = job_summary.get('processing_time', 0)
        completed_at = job_summary.get('completed_at', datetime.now())

        text = f"""
        Batch Query Processing Complete
        ================================

        Your batch query job has been completed!

        Job Details:
        -----------
        Job ID: {job_id}
        Total Questions: {total}
        Successfully Processed: {completed}
        Failed: {failed}
        Success Rate: {success_rate:.1f}%
        Processing Time: {processing_time:.1f} seconds
        Completed At: {completed_at.strftime("%B %d, %Y at %I:%M %p")}

        """
        if pdf_download_url:
            text += f"""
            Download Your Report:
            --------------------
            {pdf_download_url}

            The download link will be valid for 24 hours.

            """
        else:
            text += """
            Your report is ready and can be downloaded from your dashboard.
            """

        text += """
            Thank you for using PDP Africa Analyzer!

            ---
            This is an automated email. Please do not reply to this message.
            """
        return text

    def _format_html_email(
        self,
        recipient_email: str,
        job_summary: Dict[str, Any],
        pdf_download_url: Optional[str] = None
    ) -> str:
        """Format HTML version of batch completion email (simple and professional)."""
        # Values go into HTML, so they are escaped (the link also sits in an attribute)
        job_id = html_lib.escape(str(job_summary.get('job_id', 'N/A')))
        total = job_summary.get('total_queries', 0)
        completed = job_summary.get('completed_queries', 0)
        failed = job_summary.get('failed_queries', 0)
        success_rate = job_summary.get('success_rate', 0)
        processing_time = job_summary.get('processing_time', 0)
        completed_at = job_summary.get('completed_at', datetime.now())

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 600px;
                    margin: 0 auto;
                    padding: 20px;
                    background-color: #ffffff;
                }}
                .container {{
                    border: 1px solid #ddd;
                    border-radius: 4px;
                }}
                .header {{
                    background: #f8f9fa;
                    padding: 20px;
                    border-bottom: 2px solid #333;
                }}
                .header h1 {{
                    margin: 0;
                    font-size: 20px;
                    font-weight: 600;
                    color: #333;
                }}
                .content {{
                    padding: 30px 20px;
                }}
                .message {{
                    margin-bottom: 25px;
                    padding-bottom: 20px;
                    border-bottom: 1px solid #eee;
                }}
                .stats-table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }}
                .stats-table td {{
                    padding: 10px 0;
                    border-bottom: 1px solid #f0f0f0;
                }}
                .stats-table tr:last-child td {{
                    border-bottom: none;
                }}
                .stats-label {{
                    font-weight: 500;
                    color: #666;
                    width: 50%;
                }}
                .stats-value {{
                    text-align: right;
                    color: #333;
                    font-weight: 500;
                }}
                .download-section {{
                    text-align: center;
                    margin: 30px 0;
                    padding: 20px;
                    background: #f8f9fa;
                    border-radius: 4px;
                }}
                .download-button {{
                    display: inline-block;
                    background: #333;
                    color: white !important;
                    padding: 12px 30px;
                    text-decoration: none;
                    border-radius: 4px;
                    font-weight: 500;
                    margin: 10px 0;
                }}
                .footer {{
                    text-align: center;
                    color: #999;
                    font-size: 12px;
                    padding: 20px;
                    border-top: 1px solid #eee;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Batch Query Report Ready</h1>
                </div>
                <div class="content">
                    <div class="message">
                        <p>Your batch query job has been completed successfully.</p>
                    </div>

                    <table class="stats-table">
                        <tr>
                            <td class="stats-label">Job ID</td>
                            <td class="stats-value">{job_id}</td>
                        </tr>
                        <tr>
                            <td class="stats-label">Total Questions</td>
                            <td class="stats-value">{total}</td>
                        </tr>
                        <tr>
                            <td class="stats-label">Successfully Processed</td>
                            <td class="stats-value">{completed}</td>
                        </tr>
                        <tr>
                            <td class="stats-label">Failed</td>
                            <td class="stats-value">{failed}</td>
                        </tr>
                        <tr>
                            <td class="stats-label">Success Rate</td>
                            <td class="stats-value">{success_rate:.1f}%</td>
                        </tr>
                        <tr>
                            <td class="stats-label">Processing Time</td>
                            <td class="stats-value">{processing_time:.1f}s</td>
                        </tr>
                        <tr>
                            <td class="stats-label">Completed At</td>
                            <td class="stats-value">{completed_at.strftime("%B %d, %Y at %I:%M %p")}</td>
                        </tr>
                    </table>
        """
        if pdf_download_url:
            html += f"""
            <div class="download-section">
                <a href="{html_lib.escape(pdf_download_url, quote=True)}" class="download-button">Download Report (PDF)</a>
                <p style="color: #999; font-size: 12px; margin-top: 10px;">Link valid for 24 hours</p>
            </div>
            """
        else:
            html += """
            <div class="download-section">
                <p style="margin: 0; color: #666;">Your report is ready in your dashboard.</p>
            </div>
            """

        html += """
        </div>
        <div class="footer">
            <p><strong>PDP Africa Analyzer</strong></p>
            <p>This is an automated message. Please do not reply.</p>
        </div>
            </div>
        </body>
        </html>
        """
        return html

    def send_batch_complete_email(
        self,
        recipient_email: str,
        job_summary: Dict[str, Any],
        pdf_download_url: Optional[str] = None
    ) -> bool:
        try:
            if not self.smtp_username or not self.smtp_password:
                print("Warning: SMTP credentials not configured. Email not sent.")
                return False

            # Create message
            message = MIMEMultipart('alternative')
            message['Subject'] = f"Batch Query Report Ready - {job_summary.get('total_queries', 0)} Questions Processed"
            message['From'] = f"{self.sender_name} <{self.sender_email}>"
            message['To'] = _one_line(recipient_email)

            # Create plain text and HTML versions
            text_content = self._format_plain_text_email(recipient_email, job_summary, pdf_download_url)
            html_content = self._format_html_email(recipient_email, job_summary, pdf_download_url)

            # Attach parts
            part1 = MIMEText(text_content, 'plain')
            part2 = MIMEText(html_content, 'html')
            message.attach(part1)
            message.attach(part2)

            # Send email with proper connection cleanup
            server = None
            try:
                server = self._create_connection()
                server.sendmail(self.sender_email, recipient_email, message.as_string())
                print(f"Batch completion email sent successfully to {recipient_email}")
                return True
            finally:
                # Ensure connection is closed even if there's an error
                if server:
                    try:
                        server.quit()
                    except Exception as quit_error:
                        print(f"Error closing SMTP connection: {quit_error}")

        except Exception as e:
            print(f"Error sending email: {e}")
            return False

    def send_batch_failed_email(
        self,
        recipient_email: str,
        job_id: str,
        error_message: str
    ) -> bool:

        try:
            # Skip if no SMTP credentials configured
            if not self.smtp_username or not self.smtp_password:
                print("Warning: SMTP credentials not configured. Email not sent.")
                return False

            # Create message
            message = MIMEMultipart('alternative')
            message['Subject'] = f"Batch Query Job Failed - {_one_line(job_id)}"
            message['From'] = f"{self.sender_name} <{self.sender_email}>"
            message['To'] = _one_line(recipient_email)

            # Plain text version
            text_content = f"""
                Batch Query Processing Failed
                ==============================

                Unfortunately, your batch query job has failed.

                Job ID: {job_id}
                Error: {error_message}

                Please try again or contact support if the problem persists.

                ---
                This is an automated email. Please do not reply to this message.
                """

            # HTML version (simple and professional)
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <style>
                    body {{
                        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
                        line-height: 1.6;
                        color: #333;
                        max-width: 600px;
                        margin: 0 auto;
                        padding: 20px;
                        background-color: #ffffff;
                    }}
                    .container {{
                        border: 1px solid #ddd;
                        border-radius: 4px;
                    }}
                    .header {{
                        background: #f8f9fa;
                        padding: 20px;
                        border-bottom: 2px solid #333;
                    }}
                    .header h1 {{
                        margin: 0;
                        font-size: 20px;
                        font-weight: 600;
                        color: #333;
                    }}
                    .content {{
                        padding: 30px 20px;
                    }}
                    .error-box {{
                        background: #f8f9fa;
                        padding: 15px;
                        border-left: 3px solid #333;
                        margin: 20px 0;
                    }}
                    .footer {{
                        text-align: center;
                        color: #999;
                        font-size: 12px;
                        padding: 20px;
                        border-top: 1px solid #eee;
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>Batch Processing Failed</h1>
                    </div>
                    <div class="content">
                        <p>Unfortunately, your batch query job encountered an error and could not be completed.</p>
                        <div class="error-box">
                            <p><strong>Job ID:</strong> {html_lib.escape(str(job_id))}</p>
                            <p><strong>Error:</strong> {html_lib.escape(str(error_message))}</p>
                        </div>
                        <p>Please try again or contact support if the problem persists.</p>
                    </div>
                    <div class="footer">
                        <p><strong>PDP Africa Analyzer</strong></p>
                        <p>This is an automated message. Please do not reply.</p>
                    </div>
                </div>
            </body>
            </html>
            """
            part1 = MIMEText(text_content, 'plain')
            part2 = MIMEText(html_content, 'html')
            message.attach(part1)
            message.attach(part2)
            server = None
            try:
                server = self._create_connection()
                server.sendmail(self.sender_email, recipient_email, message.as_string())
                print(f"Batch failure email sent successfully to {recipient_email}")
                return True
            finally:
                
                if server:
                    try:
                        server.quit()
                    except Exception as quit_error:
                        print(f"Error closing SMTP connection: {quit_error}")

        except Exception as e:
            print(f"Error sending failure email: {e}")
            return False

def _one_line(value: Any) -> str:
    """A header value on one line: carriage returns and line feeds would start new headers."""
    return " ".join(str(value).splitlines())


_email_service: Optional[EmailService] = None


def get_email_service() -> EmailService:
    """Get or create the singleton email service instance."""
    global _email_service
    if _email_service is None:
        _email_service = EmailService()
    return _email_service
