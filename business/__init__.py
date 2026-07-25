"""business package for Dota Analyst MVP.

Loads environment variables from a `.env` file (if python-dotenv is
installed). The import is wrapped so the package remains importable in
environments that don't have python-dotenv — e.g. minimal CI / test
setups that only exercise pure-function modules like `analysis`.
"""

# Load environment variables from .env file. Make this tolerant:
# `from business import app` should not require dotenv to be installed,
# and missing dotenv should not crash unrelated tests.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv is optional at import time. The app, scripts and
    # any code that actually needs the env values will fail loudly
    # downstream (os.environ.get() returns None), which is the right
    # behaviour — no silent fallbacks.
    pass
