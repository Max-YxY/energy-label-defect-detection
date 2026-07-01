#!/usr/bin/env python3
"""
RESTful API 服务.
"""
import sys
import base64
import io
import time
from pathlib import Path
import cv2
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image

from inference.detector import EnergyLabelDetector
from utils.logger import setup_logger, get_logger
from utils.database import DetectionDatabase

logger = get_logger(__name__)
app = Flask(__name__)
detector = None
database = None


def init_detector(config_path: str = "config.yaml"):
    global detector, database
    if detector is None:
        detector = EnergyLabelDetector(config_path)
        database = DetectionDatabase()
        logger.info("Detector initialized for API service.")
    return detector, database


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "timestamp": time.time()})


@app.route("/detect", methods=["POST"])
def detect():
    det, db = init_detector()
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"success": False, "error": "Invalid JSON"}), 400

    product_id = data.get("product_id", f"API-{int(time.time())}")
    image_b64 = data.get("image", "")
    if not image_b64:
        return jsonify({"success": False, "error": "Missing 'image' field"}), 400

    try:
        if "," in image_b64:
            image_b64 = image_b64.split(",")[1]
        img_bytes = base64.b64decode(image_b64)
    except Exception:
        return jsonify({"success": False, "error": "Invalid base64 encoding"}), 400

    try:
        pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        image = np.array(pil_image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    except Exception as e:
        return jsonify({"success": False, "error": f"Image decode failed: {e}"}), 400

    try:
        result = det.detect(image)
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

    db.insert(product_id, result)
    return jsonify({
        "success": True,
        "product_id": product_id,
        "energy_level": result["energy_level"],
        "defects": result["defects"],
        "position_deviation": result["position_deviation"],
        "offset_x": result["offset_x"],
        "offset_y": result["offset_y"],
        "inference_time_ms": result["inference_time_ms"],
    })


@app.route("/query", methods=["GET"])
def query():
    _, db = init_detector()
    records = db.query(
        product_id=request.args.get("product_id"),
        start_time=request.args.get("start"),
        end_time=request.args.get("end"),
        limit=int(request.args.get("limit", 100)),
    )
    return jsonify({"success": True, "count": len(records), "records": records})


@app.route("/statistics", methods=["GET"])
def statistics():
    _, db = init_detector()
    return jsonify({"success": True, "statistics": db.get_statistics()})


@app.route("/export", methods=["GET"])
def export():
    _, db = init_detector()
    path = db.export_csv(f"./logs/export_{int(time.time())}.csv")
    return jsonify({"success": True, "file": path})


def main():
    parser = __import__("argparse").ArgumentParser(description="Detection API Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    setup_logger()
    init_detector(args.config)
    logger.info(f"Starting API server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
