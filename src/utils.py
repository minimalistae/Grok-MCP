import os
import re
import json
import base64
from pathlib import Path
from dotenv import load_dotenv

# Prefer a real, gitignored `.env`. `example.env` (committed, placeholder-only)
# is kept solely as a non-overriding first-run fallback so a missing `.env`
# does not crash startup. python-dotenv default `override=False` means a value
# already present in the environment (e.g. exported by the MCP wrapper) wins.
load_dotenv(".env")
load_dotenv("example.env")

XAI_API_KEY = os.getenv("XAI_API_KEY", "")

# Where client-side chat history is allowed to live. Every session-derived
# path is hard-confined to this directory.
CHATS_DIR = Path("chats")

_SESSION_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def sanitize_session(session: str) -> str:
    """Validate a tool-supplied chat session name.

    Rejects anything that is not a short, flat, alnum/underscore/hyphen token.
    This blocks path separators, `..`, absolute paths, NUL bytes, and any
    other character that could escape the `chats/` directory.
    """
    if not isinstance(session, str) or not _SESSION_RE.fullmatch(session):
        raise ValueError(
            "Invalid session name: must match ^[A-Za-z0-9_-]{1,64}$ "
            "(no path separators, '..', absolute paths, or NUL bytes)."
        )
    return session


def session_path(session: str) -> Path:
    """Resolve the JSON history path for a session, confined to CHATS_DIR.

    Validates the session name, then asserts via realpath containment that
    the resolved target is strictly inside the resolved CHATS_DIR. Defends
    against symlink-based escapes in addition to lexical traversal.
    """
    sanitize_session(session)
    base = CHATS_DIR.resolve()
    target = (base / f"{session}.json").resolve()
    if target != base and not str(target).startswith(str(base) + os.sep):
        raise ValueError("Resolved session path escapes the chats directory.")
    return target


def confine_user_path(user_path: str, *, must_exist: bool = True) -> Path:
    """Resolve a tool-supplied filesystem path and confine it.

    Rejects NUL bytes outright. Resolves the path (following symlinks) and
    requires it to stay within the current working directory tree, the
    directory the resolution itself started in being the process CWD. This
    keeps `upload_file` / image / video reads from being pointed at
    `~/.env.master`, `~/.<svc>_creds`, rotation stash files, or any other
    location outside the project working tree via a poisoned argument.
    """
    if not isinstance(user_path, str) or not user_path or "\x00" in user_path:
        raise ValueError("Invalid file path: empty or contains a NUL byte.")
    base = Path.cwd().resolve()
    target = Path(user_path)
    if not target.is_absolute():
        target = base / target
    target = target.resolve()
    if target != base and not str(target).startswith(str(base) + os.sep):
        raise ValueError(
            "File path escapes the working directory; refusing to access "
            f"{user_path!r}."
        )
    if must_exist and not target.exists():
        raise FileNotFoundError(f"File not found: {user_path}")
    return target


def encode_image_to_base64(image_path: str):
    path = confine_user_path(image_path)
    with open(path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def encode_video_to_base64(video_path: str):
    path = confine_user_path(video_path)
    with open(path, "rb") as video_file:
        return base64.b64encode(video_file.read()).decode("utf-8")


def extract_usage(response):
    
    if not response.usage:
        return {}
    return {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "reasoning_tokens": response.usage.reasoning_tokens,
        "total_tokens": response.usage.total_tokens,
    }

def load_history(session: str):
    path = session_path(session)
    if path.exists():
        return json.loads(path.read_text())
    return []


def save_history(session: str, history: list):
    path = session_path(session)
    CHATS_DIR.mkdir(exist_ok=True)
    path.write_text(json.dumps(history, indent=2, ensure_ascii=False))


def build_params(**kwargs):
    result = {}
    for key, value in kwargs.items():
        if value:
            result[key] = value
    return result