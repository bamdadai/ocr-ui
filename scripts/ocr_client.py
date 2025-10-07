#!/usr/bin/env python3
"""
OCR Client

A client to submit files to the OCR service and retrieve results.
Supports both webhook-based async processing and polling-based retrieval.

Usage:
    # Submit with webhook (async)
    python ocr_client.py submit image.png --webhook http://localhost:8080/webhook/ocr

    # Submit and poll for results (sync)
    python ocr_client.py submit image.png --poll

    # Submit multiple files
    python ocr_client.py submit file1.png file2.pdf --poll

    # Check status of a task
    python ocr_client.py status <task_id>

    # Submit with metadata
    python ocr_client.py submit image.png --guid my-custom-guid --webhook http://localhost:8080/webhook/ocr
"""

import argparse
import base64
import json
import sys
import time
import uuid
from pathlib import Path
from typing import List, Dict, Optional

import requests


class OCRClient:
    """Client for interacting with the OCR API."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize the OCR client.

        Args:
            base_url: Base URL of the OCR API (default: http://localhost:8000)
        """
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()

    def submit_files(
        self,
        file_paths: List[str],
        webhook_url: Optional[str] = None,
        metadata: Optional[List[Dict]] = None
    ) -> List[Dict]:
        """
        Submit one or more files for OCR processing.

        Args:
            file_paths: List of file paths to submit
            webhook_url: Optional webhook URL for async result delivery
            metadata: Optional list of metadata dicts (one per file)

        Returns:
            List of task queue items with task_id and status for each file
        """
        url = f"{self.base_url}/v3/ocr"

        # Prepare files
        files = []
        for file_path in file_paths:
            path = Path(file_path)
            if not path.exists():
                print(f"Error: File not found: {file_path}", file=sys.stderr)
                continue

            files.append(
                ('files', (path.name, open(path, 'rb'), self._get_content_type(path)))
            )

        if not files:
            raise ValueError("No valid files to submit")

        # Prepare form data
        data = {}
        if webhook_url:
            data['webhook_url'] = webhook_url

        if metadata:
            data['metadata'] = json.dumps(metadata)

        # Submit request
        try:
            response = self.session.post(url, files=files, data=data)
            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            print(f"Error submitting files: {e}", file=sys.stderr)
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response: {e.response.text}", file=sys.stderr)
            raise

        finally:
            # Close file handles
            for _, file_tuple in files:
                file_tuple[1].close()

    def get_task_status(self, task_id: str) -> Dict:
        """
        Get the status of a specific task.

        Args:
            task_id: The task ID to check

        Returns:
            Task status response
        """
        url = f"{self.base_url}/v3/ocr/tasks/{task_id}"

        try:
            response = self.session.get(url)
            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            print(f"Error getting task status: {e}", file=sys.stderr)
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response: {e.response.text}", file=sys.stderr)
            raise

    def poll_until_complete(
        self,
        task_id: str,
        interval: float = 2.0,
        timeout: float = 300.0
    ) -> Dict:
        """
        Poll a task until it completes or times out.

        Args:
            task_id: The task ID to poll
            interval: Polling interval in seconds (default: 2.0)
            timeout: Maximum time to wait in seconds (default: 300.0)

        Returns:
            Final task result
        """
        start_time = time.time()
        print(f"Polling task {task_id}...", file=sys.stderr)

        while True:
            elapsed = time.time() - start_time
            if elapsed > timeout:
                raise TimeoutError(f"Task did not complete within {timeout} seconds")

            status_response = self.get_task_status(task_id)
            status = status_response.get('status', '').upper()

            print(f"Status: {status} (elapsed: {elapsed:.1f}s)", file=sys.stderr)

            if status == 'SUCCESS':
                print("Task completed successfully!", file=sys.stderr)
                return status_response

            elif status == 'FAILURE':
                print("Task failed!", file=sys.stderr)
                return status_response

            elif status in ['PENDING', 'STARTED', 'RETRY']:
                time.sleep(interval)

            else:
                print(f"Unknown status: {status}", file=sys.stderr)
                time.sleep(interval)

    def decode_result(self, result: Dict) -> str:
        """
        Decode the base64-encoded OCR text from a result.

        Args:
            result: Task result containing encoded text

        Returns:
            Decoded text string
        """
        if 'result' in result and result['result']:
            text_base64 = result['result'].get('text', '')
            if text_base64:
                try:
                    return base64.b64decode(text_base64).decode('utf-8')
                except Exception as e:
                    return f"[DECODE ERROR: {e}]"
        return ""

    @staticmethod
    def _get_content_type(path: Path) -> str:
        """Get content type based on file extension."""
        suffix = path.suffix.lower()
        content_types = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.pdf': 'application/pdf',
            '.tif': 'image/tiff',
            '.tiff': 'image/tiff',
        }
        return content_types.get(suffix, 'application/octet-stream')


def cmd_submit(args):
    """Handle the 'submit' command."""
    client = OCRClient(base_url=args.base_url)

    # Prepare metadata if GUID is provided or if webhook is used without --no-metadata
    metadata = None
    if args.guid:
        # Custom GUID provided
        metadata = [{"guid": args.guid, "format": Path(args.files[0]).suffix}]
    elif args.webhook and not args.no_metadata:
        # Webhook provided but no custom GUID - generate metadata automatically
        metadata = [{"guid": f"auto-{uuid.uuid4()}", "format": Path(f).suffix} for f in args.files]

    # Submit files
    print(f"Submitting {len(args.files)} file(s)...", file=sys.stderr)
    if args.webhook:
        print(f"Webhook URL: {args.webhook}", file=sys.stderr)
    if metadata:
        print(f"Metadata: {json.dumps(metadata, indent=2)}", file=sys.stderr)
    else:
        print("No metadata (auto-generated by server)", file=sys.stderr)

    results = client.submit_files(
        file_paths=args.files,
        webhook_url=args.webhook,
        metadata=metadata
    )

    print(json.dumps(results, indent=2))

    # If polling is requested
    if args.poll:
        for item in results:
            task_id = item.get('task_id')
            if not task_id:
                print(f"Skipping item with no task_id: {item}", file=sys.stderr)
                continue

            print(f"\n{'='*60}", file=sys.stderr)
            print(f"Polling task: {task_id}", file=sys.stderr)
            print(f"{'='*60}", file=sys.stderr)

            try:
                final_result = client.poll_until_complete(
                    task_id,
                    interval=args.poll_interval,
                    timeout=args.timeout
                )

                # Decode and display result
                decoded_text = client.decode_result(final_result)

                print("\n" + "="*60, file=sys.stderr)
                print("FINAL RESULT", file=sys.stderr)
                print("="*60, file=sys.stderr)
                print(json.dumps(final_result, indent=2))

                if decoded_text:
                    print("\n" + "="*60, file=sys.stderr)
                    print("DECODED TEXT", file=sys.stderr)
                    print("="*60, file=sys.stderr)
                    print(decoded_text)

                # Save to file if requested
                if args.output:
                    output_path = Path(args.output)
                    if len(results) > 1:
                        # Multiple files: append task_id to filename
                        output_path = output_path.parent / f"{output_path.stem}_{task_id}{output_path.suffix}"

                    output_path.write_text(decoded_text, encoding='utf-8')
                    print(f"\nSaved to: {output_path}", file=sys.stderr)

            except Exception as e:
                print(f"Error polling task {task_id}: {e}", file=sys.stderr)


def cmd_status(args):
    """Handle the 'status' command."""
    client = OCRClient(base_url=args.base_url)

    print(f"Checking status of task: {args.task_id}", file=sys.stderr)
    result = client.get_task_status(args.task_id)

    print(json.dumps(result, indent=2))

    # If completed, show decoded text
    if result.get('status') == 'SUCCESS':
        decoded_text = client.decode_result(result)
        if decoded_text:
            print("\n" + "="*60, file=sys.stderr)
            print("DECODED TEXT", file=sys.stderr)
            print("="*60, file=sys.stderr)
            print(decoded_text)


def main():
    parser = argparse.ArgumentParser(
        description='OCR Client - Submit files and retrieve results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Submit with webhook
  %(prog)s submit image.png --webhook http://localhost:8080/webhook/ocr

  # Submit and poll for results
  %(prog)s submit image.png --poll

  # Submit multiple files
  %(prog)s submit file1.png file2.pdf --poll

  # Check task status
  %(prog)s status <task-id>

  # Submit with custom GUID and save output
  %(prog)s submit image.png --guid my-guid --poll --output result.txt
        """
    )

    parser.add_argument(
        '--base-url',
        default='http://localhost:8000',
        help='Base URL of the OCR API (default: http://localhost:8000)'
    )

    subparsers = parser.add_subparsers(dest='command', required=True)

    # Submit command
    submit_parser = subparsers.add_parser('submit', help='Submit files for OCR')
    submit_parser.add_argument('files', nargs='+', help='Files to submit')
    submit_parser.add_argument('--webhook', help='Webhook URL for async result delivery')
    submit_parser.add_argument('--guid', help='Custom GUID for the task')
    submit_parser.add_argument('--no-metadata', action='store_true', help='Do not send metadata (test server auto-generation)')
    submit_parser.add_argument('--poll', action='store_true', help='Poll for results until complete')
    submit_parser.add_argument('--poll-interval', type=float, default=2.0, help='Polling interval in seconds')
    submit_parser.add_argument('--timeout', type=float, default=300.0, help='Polling timeout in seconds')
    submit_parser.add_argument('--output', '-o', help='Save decoded text to file')

    # Status command
    status_parser = subparsers.add_parser('status', help='Check task status')
    status_parser.add_argument('task_id', help='Task ID to check')

    args = parser.parse_args()

    if args.command == 'submit':
        cmd_submit(args)
    elif args.command == 'status':
        cmd_status(args)


if __name__ == '__main__':
    main()
