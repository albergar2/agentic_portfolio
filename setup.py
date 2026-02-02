#!/usr/bin/env python3
"""
Setup script for Wealthfolio Portfolio Strategy System

This script sets up the project structure and dependencies.
"""

import os
import sys
import subprocess
from pathlib import Path


def create_directories():
    """Create necessary directories for the project."""
    directories = [
        "src",
        "src/core",
        "src/ai", 
        "src/services",
        "reports",
        "logs",
        "tests",
        "docs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")


def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    
    print(f"✅ Python version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")


def install_dependencies():
    """Install required dependencies."""
    print("📦 Installing dependencies...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False
    
    return True


def create_env_file():
    """Create a sample .env file if it doesn't exist."""
    env_file = Path(".env")
    
    if not env_file.exists():
        sample_env = """# Wealthfolio Portfolio Strategy System Configuration

# Gemini API Key (required for AI features)
GEMINI_API_KEY=your_gemini_api_key_here

# Database Configuration
DB_PATH=db/weatlhfolio.db

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=logs/wealthfolio.log

# Report Configuration
REPORT_FORMAT=markdown
OUTPUT_DIR=reports

# AI Model Configuration
LLM_MODEL=gemini-1.5-flash
"""
        
        with open(env_file, 'w') as f:
            f.write(sample_env)
        
        print("✅ Created sample .env file")
        print("📝 Please update .env with your Gemini API key")
    else:
        print("✅ .env file already exists")


def create_gitignore():
    """Create a .gitignore file."""
    gitignore_file = Path(".gitignore")
    
    if not gitignore_file.exists():
        gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so

# Dependencies
pip-log.txt
pip-delete-this-directory.txt

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
.hypothesis/
.pytest_cache/

# Translations
*.mo
*.pot

# Django stuff:
*.log
local_settings.py
db.sqlite3

# Flask stuff:
instance/
.webassets-cache

# Scrapy stuff:
.scrapy

# Sphinx documentation
docs/_build/

# PyBuilder
target/

# Jupyter Notebook
.ipynb_checkpoints

# pyenv
.python-version

# celery beat schedule file
celerybeat-schedule

# SageMath parsed files
*.sage.py

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# IDE files
.vscode/
.idea/
*.swp
*.swo
*~

# Database files
*.db
*.sqlite
*.sqlite3

# Logs
logs/
*.log

# Reports
reports/
"""
        
        with open(gitignore_file, 'w') as f:
            f.write(gitignore_content)
        
        print("✅ Created .gitignore file")
    else:
        print("✅ .gitignore file already exists")


def validate_setup():
    """Validate the setup by running a basic test."""
    print("🔍 Validating setup...")
    
    try:
        # Add src to Python path for validation
        import sys
        import os
        src_path = os.path.join(os.path.dirname(__file__), "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        
        # Test imports
        from core.database import DatabaseManager
        from core.portfolio import PortfolioAnalyzer
        from ai.market_oracle import MarketOracle
        from services.report_orchestrator import ReportOrchestrator
        
        print("✅ All imports successful")
        
        # Test database connection (read-only)
        db_manager = DatabaseManager()
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            if result:
                print("✅ Database connection successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Setup validation failed: {e}")
        return False


def main():
    """Main setup function."""
    print("🚀 Setting up Wealthfolio Portfolio Strategy System...")
    print("=" * 60)
    
    # Check Python version
    check_python_version()
    
    # Create directories
    create_directories()
    
    # Create configuration files
    create_env_file()
    create_gitignore()
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Setup failed during dependency installation")
        sys.exit(1)
    
    # Validate setup
    if validate_setup():
        print("=" * 60)
        print("✅ Setup completed successfully!")
        print("")
        print("📋 Next steps:")
        print("1. Update .env file with your Gemini API key")
        print("2. Ensure your database is accessible at db/weatlhfolio.db")
        print("3. Run 'python src/main.py help' to see available commands")
        print("4. Try 'python src/main.py portfolio' to test portfolio analysis")
    else:
        print("=" * 60)
        print("❌ Setup validation failed")
        print("Please check the error messages above and try again")
        sys.exit(1)


if __name__ == "__main__":
    main()