import os
import uuid
import logging
import base64
import boto3
from botocore.exceptions import ClientError
from typing import Optional

logger = logging.getLogger('browser-use-api')

class S3Uploader:
    """Utility class for uploading files to S3"""
    
    def __init__(self):
        # Get S3 configuration from environment variables
        self.aws_access_key = os.environ.get('AWS_ACCESS_KEY_ID')
        self.aws_secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
        self.aws_region = os.environ.get('AWS_REGION', 'us-east-1')
        self.bucket_name = os.environ.get('AWS_S3_BUCKET')
        self.s3_prefix = os.environ.get('S3_PREFIX', 'browser-use-api/tasks')
        
        # Check if S3 is configured
        self.is_configured = all([self.aws_access_key, self.aws_secret_key, self.bucket_name])
        
        if self.is_configured:
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=self.aws_access_key,
                aws_secret_access_key=self.aws_secret_key,
                region_name=self.aws_region
            )
            logger.info(f"S3 uploader initialized with bucket: {self.bucket_name}")
        else:
            logger.warning("S3 uploader not fully configured. Screenshots will be included in the response.")
    
    async def upload_screenshot(self, base64_screenshot: str, task_id: str, step_number: int) -> Optional[str]:
        """Upload a base64 screenshot to S3 and return the URL"""
        if not self.is_configured or not base64_screenshot:
            return None
        
        try:
            # Decode the base64 string
            if base64_screenshot.startswith('data:image'):
                # Handle data URLs (e.g., data:image/png;base64,iVBORw0KGgo...)
                _, base64_data = base64_screenshot.split(',', 1)
                image_data = base64.b64decode(base64_data)
            else:
                # Handle raw base64 strings
                image_data = base64.b64decode(base64_screenshot)
            
            # Generate a unique filename
            filename = f"{task_id}/{step_number}_{uuid.uuid4().hex}.png"
            s3_key = f"{self.s3_prefix}/{filename}"
            
            # Upload to S3
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=image_data,
                ContentType='image/png'
            )
            
            # Generate URL
            url = f"https://{self.bucket_name}.s3.{self.aws_region}.amazonaws.com/{s3_key}"
            logger.debug(f"Uploaded screenshot to S3: {url}")
            return url
        
        except ClientError as e:
            logger.error(f"Error uploading screenshot to S3: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error uploading screenshot: {e}")
            return None

# Create a singleton instance
s3_uploader = S3Uploader()
