#!/usr/bin/env python3
"""
Process single elephant image for identification
"""

import sys
import json
import traceback
from backend_server import backend

def main():
    try:
        if len(sys.argv) < 2:
            print(json.dumps({"error": "Image path required"}))
            sys.exit(1)

        image_path = sys.argv[1]

        if not backend.model_loaded:
            print(json.dumps({"error": "Model not loaded"}))
            sys.exit(1)

        # Process image
        ear_region, ear_tensor, ear_base64 = backend.process_image(image_path)

        # Get predictions
        predictions = backend.predict_elephant(ear_tensor)

        result = {
            "success": True,
            "ear_region_base64": ear_base64,
            "predictions": predictions
        }

        print(json.dumps(result))

    except Exception as e:
        error_result = {
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        print(json.dumps(error_result))
        sys.exit(1)

if __name__ == "__main__":
    main()
