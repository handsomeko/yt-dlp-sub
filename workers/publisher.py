"""
PublishWorker - Distributes generated content to various platforms
Handles the final step in the content pipeline
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum

from workers.base import BaseWorker, WorkerStatus
from config.settings import get_settings
from core.storage_paths_v2 import get_storage_paths_v2

logger = logging.getLogger(__name__)


class PublishTarget(Enum):
    """Supported publishing targets"""
    LOCAL = "local"
    WEBHOOK = "webhook"
    EMAIL = "email"
    API = "api"
    # Phase 2
    TWITTER = "twitter"
    LINKEDIN = "linkedin"
    FACEBOOK = "facebook"
    MEDIUM = "medium"
    WORDPRESS = "wordpress"
    # Phase 3
    BUFFER = "buffer"
    HOOTSUITE = "hootsuite"
    ZAPIER = "zapier"


class PublishWorker(BaseWorker):
    """
    Worker that handles content distribution to various platforms
    
    Phase 1: Local storage and webhooks
    Phase 2: Social media platforms
    Phase 3: Full omnichannel publishing
    """
    
    def __init__(self):
        super().__init__("publisher")
        self.settings = get_settings()
        self.storage_paths = get_storage_paths_v2()
        
        # Phase 1 supported targets
        self.supported_targets = {
            PublishTarget.LOCAL,
            PublishTarget.WEBHOOK,
            PublishTarget.EMAIL,
            PublishTarget.API
        }
        
        # Publishing results tracking
        self.publish_results = []
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input has required fields"""
        required = ['content_id', 'content_type', 'publish_targets']
        
        if not all(field in input_data for field in required):
            self.log_with_context(
                logging.ERROR,
                f"Missing required fields. Required: {required}",
                extra={"input_fields": list(input_data.keys())}
            )
            return False
        
        # Validate publish targets
        targets = input_data.get('publish_targets', [])
        if not targets:
            self.log_with_context(
                logging.ERROR,
                "No publish targets specified"
            )
            return False
        
        # Check if targets are supported
        for target in targets:
            target_enum = PublishTarget(target) if isinstance(target, str) else target
            if target_enum not in self.supported_targets:
                self.log_with_context(
                    logging.WARNING,
                    f"Unsupported publish target: {target} (Phase 1 only supports local, webhook, email, api)"
                )
        
        return True
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute content publishing to specified targets
        
        Args:
            input_data: Must contain:
                - content_id: ID of the content to publish
                - content_type: Type of content (blog, social, newsletter, etc.)
                - content_path: Path to the content file (optional)
                - content_data: Actual content data (optional)
                - publish_targets: List of publishing targets
                - publish_options: Target-specific options
        
        Returns:
            Dict with publishing results
        """
        content_id = input_data['content_id']
        content_type = input_data['content_type']
        publish_targets = input_data['publish_targets']
        publish_options = input_data.get('publish_options', {})
        
        self.log_with_context(
            logging.INFO,
            f"Starting content publishing for {content_id}",
            extra={
                'content_type': content_type,
                'targets': publish_targets
            }
        )
        
        # Get content data
        content_data = self._get_content_data(input_data)
        if not content_data:
            return {
                'status': WorkerStatus.FAILED,
                'error': 'Could not retrieve content data'
            }
        
        # Publish to each target
        self.publish_results = []
        success_count = 0
        failed_count = 0
        
        for target in publish_targets:
            try:
                target_enum = PublishTarget(target) if isinstance(target, str) else target
                target_options = publish_options.get(target, {})
                
                if target_enum == PublishTarget.LOCAL:
                    result = self._publish_local(content_data, content_type, target_options)
                elif target_enum == PublishTarget.WEBHOOK:
                    result = self._publish_webhook(content_data, content_type, target_options)
                elif target_enum == PublishTarget.EMAIL:
                    result = self._publish_email(content_data, content_type, target_options)
                elif target_enum == PublishTarget.API:
                    result = self._publish_api(content_data, content_type, target_options)
                else:
                    result = {
                        'status': 'skipped',
                        'message': f'{target} not supported in Phase 1'
                    }
                
                self.publish_results.append({
                    'target': target,
                    'result': result,
                    'timestamp': datetime.utcnow().isoformat()
                })
                
                if result.get('status') == 'success':
                    success_count += 1
                else:
                    failed_count += 1
                    
            except Exception as e:
                self.log_with_context(
                    logging.ERROR,
                    f"Failed to publish to {target}: {str(e)}"
                )
                self.publish_results.append({
                    'target': target,
                    'result': {'status': 'failed', 'error': str(e)},
                    'timestamp': datetime.utcnow().isoformat()
                })
                failed_count += 1
        
        # Determine overall status
        if success_count > 0 and failed_count == 0:
            status = WorkerStatus.SUCCESS
        elif success_count > 0 and failed_count > 0:
            status = WorkerStatus.PARTIAL
        else:
            status = WorkerStatus.FAILED
        
        return {
            'status': status,
            'content_id': content_id,
            'publish_results': self.publish_results,
            'summary': {
                'total_targets': len(publish_targets),
                'successful': success_count,
                'failed': failed_count
            }
        }
    
    def _get_content_data(self, input_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Retrieve content data from input or file"""
        # Direct content data provided
        if 'content_data' in input_data:
            return input_data['content_data']
        
        # Load from file path
        if 'content_path' in input_data:
            content_path = Path(input_data['content_path'])
            if content_path.exists():
                try:
                    if content_path.suffix == '.json':
                        with open(content_path, 'r') as f:
                            return json.load(f)
                    else:
                        with open(content_path, 'r') as f:
                            return {'content': f.read()}
                except Exception as e:
                    self.log_with_context(
                        logging.ERROR,
                        f"Failed to load content from {content_path}: {str(e)}"
                    )
        
        # TODO: Phase 2 - Load from database using content_id
        
        return None
    
    def _publish_local(self, content_data: Dict[str, Any], content_type: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """
        Publish content to local storage
        
        Phase 1: Save to output directory
        """
        try:
            output_dir = Path(options.get('output_dir', self.settings.storage_path / 'published'))
            output_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            filename = f"{content_type}_{timestamp}.json"
            output_path = output_dir / filename
            
            with open(output_path, 'w') as f:
                json.dump({
                    'content_type': content_type,
                    'published_at': datetime.utcnow().isoformat(),
                    'data': content_data
                }, f, indent=2)
            
            self.log_with_context(
                logging.INFO,
                f"Published content locally to {output_path}"
            )
            
            return {
                'status': 'success',
                'output_path': str(output_path)
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    def _publish_webhook(self, content_data: Dict[str, Any], content_type: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """
        Publish content via webhook
        
        Phase 1: Send POST request to webhook URL
        """
        webhook_url = options.get('webhook_url')
        if not webhook_url:
            return {
                'status': 'failed',
                'error': 'No webhook URL provided'
            }
        
        # TODO: Implement webhook publishing with httpx
        # For Phase 1, we'll simulate this
        self.log_with_context(
            logging.INFO,
            f"Would publish to webhook: {webhook_url}"
        )
        
        return {
            'status': 'success',
            'message': f'Webhook publishing to {webhook_url} simulated (Phase 1)'
        }
    
    def _publish_email(self, content_data: Dict[str, Any], content_type: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """
        Publish content via email
        
        Phase 1: Prepare email content (actual sending in Phase 2)
        """
        email_to = options.get('email_to')
        if not email_to:
            return {
                'status': 'failed',
                'error': 'No email recipient provided'
            }
        
        # Prepare email content
        email_subject = options.get('subject', f'New {content_type} content ready')
        email_body = self._format_email_body(content_data, content_type)
        
        # TODO: Phase 2 - Actual email sending
        self.log_with_context(
            logging.INFO,
            f"Email prepared for {email_to}: {email_subject}"
        )
        
        return {
            'status': 'success',
            'message': f'Email to {email_to} prepared (actual sending in Phase 2)',
            'email_data': {
                'to': email_to,
                'subject': email_subject,
                'body_preview': email_body[:200] + '...' if len(email_body) > 200 else email_body
            }
        }
    
    def _publish_api(self, content_data: Dict[str, Any], content_type: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """
        Publish content via API endpoint
        
        Phase 1: Prepare API request (actual sending in Phase 2)
        """
        api_endpoint = options.get('api_endpoint')
        if not api_endpoint:
            return {
                'status': 'failed',
                'error': 'No API endpoint provided'
            }
        
        # Prepare API payload
        api_payload = {
            'content_type': content_type,
            'data': content_data,
            'metadata': {
                'published_at': datetime.utcnow().isoformat(),
                'source': 'yt-dl-sub'
            }
        }
        
        # TODO: Phase 2 - Actual API call with httpx
        self.log_with_context(
            logging.INFO,
            f"API request prepared for {api_endpoint}"
        )
        
        return {
            'status': 'success',
            'message': f'API publishing to {api_endpoint} prepared (actual sending in Phase 2)',
            'payload_size': len(json.dumps(api_payload))
        }
    
    def _format_email_body(self, content_data: Dict[str, Any], content_type: str) -> str:
        """Format content for email body"""
        if isinstance(content_data, dict):
            if 'content' in content_data:
                return str(content_data['content'])
            else:
                return json.dumps(content_data, indent=2)
        return str(content_data)
    
    def handle_error(self, error: Exception) -> Dict[str, Any]:
        """Categorize and handle different error types"""
        error_str = str(error).lower()
        
        if 'network' in error_str or 'connection' in error_str:
            return {
                'error_code': 'E001',
                'error_type': 'network',
                'message': 'Network error during publishing',
                'recoverable': True,
                'retry_delay': 60
            }
        elif 'unauthorized' in error_str or '401' in error_str:
            return {
                'error_code': 'E002',
                'error_type': 'auth',
                'message': 'Authentication failed',
                'recoverable': False,
                'action_required': 'Check API credentials'
            }
        elif 'rate' in error_str or '429' in error_str:
            return {
                'error_code': 'E003',
                'error_type': 'rate_limit',
                'message': 'Rate limit exceeded',
                'recoverable': True,
                'retry_delay': 300
            }
        else:
            return {
                'error_code': 'E999',
                'error_type': 'unknown',
                'message': str(error),
                'recoverable': True
            }


# Convenience function for direct use
async def publish_content(
    content_id: str,
    content_type: str,
    publish_targets: List[str],
    content_data: Optional[Dict[str, Any]] = None,
    content_path: Optional[str] = None,
    publish_options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Convenience function to publish content
    
    Args:
        content_id: Unique identifier for the content
        content_type: Type of content (blog, social, newsletter, etc.)
        publish_targets: List of publishing targets
        content_data: Optional actual content data
        content_path: Optional path to content file
        publish_options: Optional target-specific options
    
    Returns:
        Dict with publishing results
    """
    worker = PublishWorker()
    return worker.run({
        'content_id': content_id,
        'content_type': content_type,
        'publish_targets': publish_targets,
        'content_data': content_data,
        'content_path': content_path,
        'publish_options': publish_options or {}
    })