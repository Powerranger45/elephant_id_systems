#!/usr/bin/env python3
"""
Get model information and status
"""

import sys
import json
import traceback
from backend_server import backend

def main():
    try:
        if not backend.model_loaded:
            result = {
                "error": "Model not loaded",
                "model_loaded": False,
                "elephants_count": 0
            }
        else:
            result = {
                "success": True,
                "model_loaded": True,
                "elephants_count": len(backend.class_names),
                "device": str(backend.device),
                "class_names": backend.class_names[:10]  # First 10 for preview
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
