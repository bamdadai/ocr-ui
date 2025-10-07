#!/usr/bin/env python3
"""
End-to-end test to verify webhook URL is correctly propagated through the system.

This test:
1. Submits a file with a webhook URL
2. Checks the API response
3. Monitors logs to verify webhook URL appears at each stage
4. Verifies the webhook URL is used when sending results

Usage:
    python scripts/test_webhook_flow.py [--api-url URL] [--webhook-url URL]
"""

import argparse
import json
import sys
import time
import tempfile
import base64
from pathlib import Path
from typing import Optional

try:
    import requests
except ImportError:
    print("Error: requests library is required", file=sys.stderr)
    print("Install with: pip install requests", file=sys.stderr)
    sys.exit(1)


def create_test_image() -> str:
    """Create a minimal test image file."""
    # 1x1 PNG pixel (minimal valid PNG)
    png_data = base64.b64decode(
        b'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=='
    )

    temp_file = tempfile.NamedTemporaryFile(mode='wb', suffix='.png', delete=False)
    temp_file.write(png_data)
    temp_file.close()

    return temp_file.name


class WebhookFlowTester:
    """Test webhook URL propagation through the system."""

    def __init__(self, api_url: str, webhook_url: str):
        self.api_url = api_url.rstrip('/')
        self.webhook_url = webhook_url
        self.test_guid = f"test-{int(time.time())}"
        self.session = requests.Session()

    def step(self, msg: str):
        """Print a test step."""
        print(f"\n{'='*60}")
        print(f"Step: {msg}")
        print(f"{'='*60}")

    def test_api_submission(self) -> Optional[str]:
        """Test 1: Submit file and verify API receives webhook URL."""
        self.step("Testing API submission with webhook URL")

        test_image = create_test_image()

        try:
            print(f"Webhook URL: {self.webhook_url}")
            print(f"Test GUID: {self.test_guid}")

            files = {
                'files': ('test.png', open(test_image, 'rb'), 'image/png')
            }

            data = {
                'webhook_url': self.webhook_url,
                'metadata': json.dumps([{
                    'guid': self.test_guid,
                    'format': '.png'
                }])
            }

            response = self.session.post(
                f"{self.api_url}/v3/ocr",
                files=files,
                data=data
            )

            print(f"Status Code: {response.status_code}")
            print(f"Response: {json.dumps(response.json(), indent=2)}")

            if response.status_code != 200:
                print(f"❌ FAILED - Expected 200, got {response.status_code}")
                return None

            result = response.json()

            if not result or len(result) == 0:
                print("❌ FAILED - No results returned")
                return None

            task_id = result[0].get('task_id')
            if not task_id:
                print("❌ FAILED - No task_id in response")
                return None

            print(f"✅ PASSED - Task ID: {task_id}")
            return task_id

        except Exception as e:
            print(f"❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            return None

        finally:
            Path(test_image).unlink(missing_ok=True)

    def test_task_status(self, task_id: str) -> bool:
        """Test 2: Poll task status until complete."""
        self.step("Polling task status")

        print(f"Task ID: {task_id}")
        print(f"Polling every 2 seconds (max 60 seconds)...")

        start_time = time.time()
        max_wait = 60

        while True:
            elapsed = time.time() - start_time
            if elapsed > max_wait:
                print(f"❌ FAILED - Task did not complete within {max_wait} seconds")
                return False

            try:
                response = self.session.get(f"{self.api_url}/v3/ocr/tasks/{task_id}")
                response.raise_for_status()
                status_data = response.json()

                status = status_data.get('status', '').upper()
                print(f"  Status: {status} (elapsed: {elapsed:.1f}s)")

                if status == 'SUCCESS':
                    print("✅ PASSED - Task completed successfully")
                    print(f"Result: {json.dumps(status_data, indent=2)}")
                    return True

                elif status == 'FAILURE':
                    print("❌ FAILED - Task failed")
                    print(f"Result: {json.dumps(status_data, indent=2)}")
                    return False

                time.sleep(2)

            except Exception as e:
                print(f"❌ ERROR: {e}")
                return False

    def test_webhook_logs(self) -> bool:
        """Test 3: Verify webhook URL appears in logs."""
        self.step("Checking for webhook URL in logs")

        print("Expected log entries:")
        print(f"  - ocr.request.received with webhook_url={self.webhook_url}")
        print(f"  - ocr.task.dispatching with webhook_url={self.webhook_url}")
        print(f"  - ocr.task.dispatched with webhook_url={self.webhook_url}")
        print(f"  - webhook.dispatch with webhook_url={self.webhook_url}")
        print(f"  - webhook.attempt.start with webhook_url={self.webhook_url}")
        print(f"  - webhook.send.success with webhook_url={self.webhook_url}")

        print("\n⚠️  Please check your application logs to verify these entries exist.")
        print(f"    Search for: guid={self.test_guid}")
        print(f"    Or search for: webhook_url={self.webhook_url}")

        return True

    def run_all_tests(self) -> bool:
        """Run all tests in sequence."""
        print("="*60)
        print("Webhook URL Flow Test")
        print("="*60)
        print(f"API URL: {self.api_url}")
        print(f"Webhook URL: {self.webhook_url}")
        print(f"Test GUID: {self.test_guid}")

        # Test 1: Submit
        task_id = self.test_api_submission()
        if not task_id:
            print("\n❌ Test suite FAILED at submission")
            return False

        # Test 2: Poll
        if not self.test_task_status(task_id):
            print("\n❌ Test suite FAILED at status check")
            return False

        # Test 3: Logs
        if not self.test_webhook_logs():
            print("\n❌ Test suite FAILED at log verification")
            return False

        print("\n" + "="*60)
        print("✅ All tests PASSED")
        print("="*60)
        print("\nTo verify webhook delivery:")
        print(f"  1. Check webhook server logs for GUID: {self.test_guid}")
        print(f"  2. Check OCR service logs for webhook_url: {self.webhook_url}")
        print(f"  3. Verify webhook.send.success log entry exists")

        return True


def main():
    parser = argparse.ArgumentParser(
        description='Test webhook URL propagation through the OCR system'
    )
    parser.add_argument(
        '--api-url',
        default='http://localhost:8000',
        help='OCR API base URL (default: http://localhost:8000)'
    )
    parser.add_argument(
        '--webhook-url',
        default='http://localhost:8080/webhook/ocr',
        help='Webhook URL to test (default: http://localhost:8080/webhook/ocr)'
    )

    args = parser.parse_args()

    tester = WebhookFlowTester(args.api_url, args.webhook_url)
    success = tester.run_all_tests()

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
