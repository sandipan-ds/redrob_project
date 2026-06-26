"""Helper for push_to_hf.sh: read HF_TOKEN from .env via python-dotenv."""
import os
import sys

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed; fall back to env var

token = os.environ.get("HF_TOKEN", "")
if not token:
    print("", end="")
    sys.exit(0)
print(token, end="")
