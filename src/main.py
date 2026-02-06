#!/usr/bin/env python3
"""
Wealthfolio Portfolio Strategy System - Main Orchestrator

Unified interface for generating strategy reports, analyzing portfolio performance,
and enriching asset metadata using AI and market research.
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.logging import RichHandler
from rich.markdown import Markdown

from src.core.config import WEALTHFOLIO_DB, REPORTS_DIR, LOG_FILE

# Configure logging with Rich
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[
        RichHandler(rich_tracebacks=True),
        logging.FileHandler(str(LOG_FILE))
    ]
)

logger = logging.getLogger("wealthfolio")
console = Console()

class WealthfolioCLI:
    """CLI for the Wealthfolio Portfolio Strategy System."""
    
    def __init__(self):
        self.db_path = str(WEALTHFOLIO_DB)

    def show_banner(self):
        """Display a stylish banner."""
        banner_text = """
 [bold cyan]Wealthfolio[/bold cyan] [bold white]Portfolio Strategy System[/bold white]
 [italic blue]AI-Driven Analysis & Market Orchestration[/italic blue]
        """
        console.print(Panel(banner_text, border_style="cyan"))

    def show_modules(self):
        """Display project modules and their responsibilities."""
        table = Table(title="Wealthfolio Core Modules", border_style="blue")
        table.add_column("Module", style="cyan", no_wrap=True)
        table.add_column("Description", style="white")

        modules = [
            ("core.config", "Centralized configuration and path management."),
            ("core.database", "SQLite database management and connection handling."),
            ("core.portfolio", "Core logic for parsing and representing portfolio data."),
            ("services.enricher", "Fetches market data (yfinance) and AI insights for assets."),
            ("services.analysis", "Generates detailed performance, risk, and exposure reports."),
            ("services.orchestrator", "Coordinates AI agents to produce strategic investment reports."),
            ("services.masker", "Ensures data privacy by masking sensitive information."),
            ("ai.market_oracle", "Provides 2026 macro-economic context and market predictions."),
        ]

        for mod, desc in modules:
            table.add_row(mod, desc)

        console.print(table)

    def generate_strategy_report(self, output: str = None):
        """Generate AI-driven strategy report."""
        # Lazy import to avoid slow startup
        from src.services.report_orchestrator import ReportOrchestrator
        
        with console.status("[bold green]Generating strategy report...") as status:
            orchestrator = ReportOrchestrator(self.db_path)
            report = orchestrator.generate_report()
        
        if output:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(report)
            console.print(f"[bold green]✅ Strategy report saved to:[/bold green] [blue]{output}[/blue]")
        else:
            console.print(Markdown(report))

    def analyze_portfolio(self):
        """Generate detailed performance and exposure report."""
        # Lazy import to avoid slow startup
        from src.services.analysis_service import AnalysisService
        
        with console.status("[bold green]Analyzing portfolio...") as status:
            service = AnalysisService()
            report = service.generate_report()
        
        console.print(Panel(report, title="Portfolio Analysis", border_style="green"))

    def enrich_portfolio(self):
        """Enrich portfolio data with yfinance and AI."""
        # Lazy import to avoid slow startup
        from src.services.portfolio_enricher import PortfolioEnricher
        
        console.print("[bold blue]Starting portfolio enrichment...[/bold blue]")
        enricher = PortfolioEnricher()
        enricher.run()
        console.print("[bold green]✅ Portfolio enrichment completed successfully![/bold green]")

    def show_market_context(self):
        """Display 2026 macro context."""
        # Lazy import to avoid slow startup
        from src.ai.market_oracle import get_macro_context
        
        with console.status("[bold blue]Retrieving market context...") as status:
            context = get_macro_context()
        
        console.print(Panel(context, title="2026 MARKET MACRO CONTEXT", border_style="magenta"))

    def run_comprehensive(self, output_dir: str):
        """Run all analysis steps and save reports."""
        # Lazy import to avoid slow startup
        from src.services.analysis_service import AnalysisService
        
        console.print(f"[bold bold cyan]🚀 Running comprehensive analysis suite...[/bold bold cyan]")
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 1. Enrichment
        self.enrich_portfolio()
        
        # 2. Detailed Analysis
        analysis_service = AnalysisService()
        analysis_report = analysis_service.generate_report()
        analysis_file = output_path / f"portfolio_analysis_{now}.txt"
        with open(analysis_file, 'w') as f:
            f.write(analysis_report)
        console.print(f"[bold green]✅ Detailed analysis saved to:[/bold green] [blue]{analysis_file}[/blue]")
        
        # 3. Strategy Report
        strategy_file = output_path / f"strategy_report_{now}.md"
        self.generate_strategy_report(str(strategy_file))
        
        console.print("[bold cyan]✨ Comprehensive analysis suite finished.[/bold cyan]")

def main():
    parser = argparse.ArgumentParser(
        description="Wealthfolio Portfolio Strategy System",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # info (new)
    subparsers.add_parser('info', help='Display project module descriptions and architecture')

    # report
    report_parser = subparsers.add_parser('report', help='Generate AI-driven strategy report')
    report_parser.add_argument('--output', '-o', type=str, help='Output markdown file path')

    # analyze
    subparsers.add_parser('analyze', help='Detailed performance and exposure analysis')

    # enrich
    subparsers.add_parser('enrich', help='Enrich portfolio assets with metadata and AI insights')

    # market
    subparsers.add_parser('market', help='Show 2026 macro-economic context')

    # comprehensive
    comp_parser = subparsers.add_parser('comprehensive', help='Run all analysis and enrichment steps')
    comp_parser.add_argument('--output-dir', '-d', type=str, default='reports', help='Directory to save reports')

    args = parser.parse_args()
    cli = WealthfolioCLI()

    if not args.command:
        cli.show_banner()
        parser.print_help()
        sys.exit(0)

    # Show banner for all commands except help (if needed)
    cli.show_banner()

    if args.command == 'info':
        cli.show_modules()
    elif args.command == 'report':
        cli.generate_strategy_report(args.output)
    elif args.command == 'analyze':
        cli.analyze_portfolio()
    elif args.command == 'enrich':
        cli.enrich_portfolio()
    elif args.command == 'market':
        cli.show_market_context()
    elif args.command == 'comprehensive':
        cli.run_comprehensive(args.output_dir)

if __name__ == "__main__":
    main()