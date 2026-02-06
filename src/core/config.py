import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()

# Database paths
DB_DIR = PROJECT_ROOT / "db"
WEALTHFOLIO_DB = DB_DIR / "weatlhfolio.db"
ENRICHMENT_DB = DB_DIR / "enrichment.db"

# Output directories
REPORTS_DIR = PROJECT_ROOT / "reports"
LOGS_DIR = PROJECT_ROOT / "logs"

# Ensure directories exist
DB_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Default settings
DEFAULT_LLM_MODEL = "gemini-1.5-flash"
DEFAULT_REPORT_FORMAT = "markdown"

# Logging configuration
LOG_FILE = LOGS_DIR / "wealthfolio.log"
