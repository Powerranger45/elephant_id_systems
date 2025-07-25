#!/usr/bin/env python3
"""
Process batch of elephant images from folder
"""

import sys
import json
import traceback
from backend_server import backend

def main():
    try:
        if len(sys.argv) < 2:
            print(json.dumps({"error": "Folder path required"}))
            sys.exit(1)

        folder_path = sys.argv[1]
        similarity_threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.85

        if not backend.model_loaded:
            print(json.dumps({"error": "Model not loaded"}))
            sys.exit(1)

        # Process batch from folder
        result = backend.process_batch(folder_path, similarity_threshold)

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
