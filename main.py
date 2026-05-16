import os
import sys
from dotenv import load_dotenv
from src import main

# Load a real, gitignored `.env`; `example.env` is only a non-overriding
# placeholder fallback so first-run does not crash. Never put a real key in
# `example.env` (it is committed to git).
load_dotenv(".env")
load_dotenv("example.env")

if __name__ == "__main__":

    if not os.getenv("XAI_API_KEY"):
        print(" XAI_API_KEY not found in environment.", file=sys.stderr)
        print("Please set your API key in a local .env file (gitignored) or export it: export XAI_API_KEY='your_api_key' ", file=sys.stderr)
    else:
        print("XAI_API_KEY found", file=sys.stderr)
        print("Started Grok MCP server", file=sys.stderr)

    main()
