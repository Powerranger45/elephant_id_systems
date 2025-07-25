#!/usr/bin/env python3
"""
Compare elephant image with existing dataset
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

        # Compare with dataset
        matches, feature_analysis = backend.compare_with_dataset(ear_tensor)

        result = {
            "success": True,
            "ear_region_base64": ear_base64,
            "matches": matches,
            "feature_analysis": feature_analysis
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
