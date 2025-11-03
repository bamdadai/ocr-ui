#!/usr/bin/env python3
"""
Webhook Server for OCR Results

This server receives OCR results from the OCR service via webhook callbacks.
It logs received results and can be extended to process them further.

Usage:
    python webhook_server.py [--port PORT] [--host HOST]

Example:
    python webhook_server.py --port 8080
    python webhook_server.py --host 0.0.0.0 --port 9000
"""

import argparse
import base64
import json
import logging
from datetime import datetime
from pathlib import Path
from flask import Flask, request, jsonify

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Store received results in memory (for demo purposes)
received_results = []

# Optional: Save results to disk
SAVE_TO_DISK = True
RESULTS_DIR = Path(__file__).parent.parent / "webhook_results"


@app.route('/callbacks/ocr', methods=['POST'])
def receive_ocr_result():
    """
    Endpoint to receive OCR results from the OCR service.

    Expected payload:
    {
        "task_id": "uuid",
        "guid": "client-guid",
        "text": "base64-encoded-utf8-text",
        "confidence": 0.95,
        "status": "completed",
        "error": ""
    }
    """
    try:
        payload = request.get_json()

        if not payload:
            logger.error("Received empty payload")
            return jsonify({"error": "Empty payload"}), 400

        # Extract data
        task_id = payload.get('task_id')
        guid = payload.get('guid')
        text_base64 = payload.get('text', '')
        confidence = payload.get('confidence', 0.0)
        status = payload.get('status', 'unknown')
        error = payload.get('error', '')

        logger.info(f"Received OCR result - Task ID: {task_id}, GUID: {guid}, Status: {status}, Confidence: {confidence:.2f}")

        # Decode the text
        decoded_text = ""
        if text_base64:
            try:
                decoded_text = base64.b64decode(text_base64).decode('utf-8')
                logger.info(f"Decoded text length: {len(decoded_text)} characters")
            except Exception as decode_error:
                logger.error(f"Failed to decode text: {decode_error}")
                decoded_text = f"[DECODE ERROR: {decode_error}]"

        # Store result
        result_record = {
            "received_at": datetime.utcnow().isoformat(),
            "task_id": task_id,
            "guid": guid,
            "status": status,
            "confidence": confidence,
            "error": error,
            "text": decoded_text,
            "text_preview": decoded_text[:200] + "..." if len(decoded_text) > 200 else decoded_text
        }

        received_results.append(result_record)

        # Save to disk if enabled
        if SAVE_TO_DISK:
            save_result_to_disk(result_record)

        # Log preview
        logger.info(f"Text preview: {result_record['text_preview']}")

        if error:
            logger.warning(f"OCR task had error: {error}")

        return jsonify({
            "status": "received",
            "task_id": task_id,
            "guid": guid,
            "message": "OCR result received successfully"
        }), 200

    except Exception as e:
        logger.exception(f"Error processing webhook: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/webhook/results', methods=['GET'])
def list_results():
    """
    List all received results (for debugging/monitoring).
    """
    return jsonify({
        "total_results": len(received_results),
        "results": received_results
    }), 200


@app.route('/webhook/results/<guid>', methods=['GET'])
def get_result_by_guid(guid):
    """
    Get a specific result by GUID.
    """
    for result in received_results:
        if result.get('guid') == guid:
            return jsonify(result), 200

    return jsonify({"error": "Result not found"}), 404


@app.route('/webhook/health', methods=['GET'])
def health_check():
    """
    Health check endpoint.
    """
    return jsonify({
        "status": "healthy",
        "server": "OCR Webhook Server",
        "results_received": len(received_results)
    }), 200


def save_result_to_disk(result_record):
    """
    Save received result to disk for persistence.
    """
    try:
        RESULTS_DIR.mkdir(exist_ok=True)

        guid = result_record.get('guid', 'unknown')
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        filename = f"{guid}_{timestamp}.json"
        filepath = RESULTS_DIR / filename

        # Save full result as JSON
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result_record, f, indent=2, ensure_ascii=False)

        # Save text separately
        text_filename = f"{guid}_{timestamp}.txt"
        text_filepath = RESULTS_DIR / text_filename
        with open(text_filepath, 'w', encoding='utf-8') as f:
            f.write(result_record.get('text', ''))

        logger.info(f"Saved result to {filepath} and {text_filepath}")

    except Exception as e:
        logger.error(f"Failed to save result to disk: {e}")


def main():
    parser = argparse.ArgumentParser(description='Webhook server for OCR results')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=8080, help='Port to listen on (default: 8080)')
    parser.add_argument('--no-save', action='store_true', help='Disable saving results to disk')

    args = parser.parse_args()

    global SAVE_TO_DISK
    if args.no_save:
        SAVE_TO_DISK = False
        logger.info("Disk saving disabled")

    logger.info(f"Starting webhook server on {args.host}:{args.port}")
    logger.info(f"Webhook endpoint: http://{args.host}:{args.port}/webhook/ocr")
    logger.info(f"Health check: http://{args.host}:{args.port}/webhook/health")
    logger.info(f"View results: http://{args.host}:{args.port}/webhook/results")

    if SAVE_TO_DISK:
        logger.info(f"Results will be saved to: {RESULTS_DIR}")

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == '__main__':
    main()
