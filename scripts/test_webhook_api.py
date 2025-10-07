#!/usr/bin/env python3
"""
Test script to verify the API correctly receives webhook URLs.

This tests various webhook URL scenarios:
1. Valid webhook URL
2. Empty webhook URL
3. Webhook URL with whitespace
4. Missing webhook URL parameter
"""

import json
import sys
import requests
from pathlib import Path


def create_test_image():
    """Create a minimal test image file."""
    import tempfile
    import base64

    # 1x1 PNG pixel (minimal valid PNG)
    png_data = base64.b64decode(
        b'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=='
    )

    temp_file = tempfile.NamedTemporaryFile(mode='wb', suffix='.png', delete=False)
    temp_file.write(png_data)
    temp_file.close()

    return temp_file.name


def test_webhook_url(api_url, test_name, webhook_url, expected_status=200):
    """Test sending a request with a specific webhook URL."""
    print(f"\n{'='*60}")
    print(f"Test: {test_name}")
    print(f"{'='*60}")
    print(f"Webhook URL: {repr(webhook_url)}")

    test_image = create_test_image()

    try:
        files = {
            'files': ('test.png', open(test_image, 'rb'), 'image/png')
        }

        data = {}

        # Only add webhook_url if it's not None (to test missing parameter)
        if webhook_url is not None:
            data['webhook_url'] = webhook_url

        print(f"Request data: {data}")

        response = requests.post(
            f"{api_url}/v3/ocr",
            files=files,
            data=data
        )

        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")

        if response.status_code == expected_status:
            print("✅ PASSED")
            return True
        else:
            print(f"❌ FAILED - Expected {expected_status}, got {response.status_code}")
            return False

    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up
        Path(test_image).unlink(missing_ok=True)


def main():
    api_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"

    print(f"Testing API: {api_url}")

    tests = [
        {
            "name": "Valid webhook URL",
            "webhook_url": "http://localhost:8080/webhook/ocr",
            "expected_status": 200
        },
        {
            "name": "Webhook URL with trailing whitespace",
            "webhook_url": "http://localhost:8080/webhook/ocr  ",
            "expected_status": 200
        },
        {
            "name": "Webhook URL with leading whitespace",
            "webhook_url": "  http://localhost:8080/webhook/ocr",
            "expected_status": 200
        },
        {
            "name": "Empty webhook URL",
            "webhook_url": "",
            "expected_status": 200
        },
        {
            "name": "Whitespace-only webhook URL",
            "webhook_url": "   ",
            "expected_status": 200
        },
        {
            "name": "Missing webhook URL parameter",
            "webhook_url": None,
            "expected_status": 200
        },
    ]

    results = []
    for test in tests:
        result = test_webhook_url(
            api_url,
            test["name"],
            test["webhook_url"],
            test["expected_status"]
        )
        results.append((test["name"], result))

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status} - {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print(f"\n❌ {total - passed} test(s) failed")
        sys.exit(1)


if __name__ == '__main__':
    main()
