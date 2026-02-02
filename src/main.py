#!/usr/bin/env python3
"""
Wealthfolio Portfolio Strategy System - Main Orchestrator

This is the main entry point for the Wealthfolio Portfolio Strategy System.
It provides a unified interface for generating comprehensive portfolio strategy reports
using AI-driven analysis and market research.

The system combines:
- Portfolio data analysis and privacy-preserving masking
- Market research and institutional outlook analysis
- AI-driven strategy recommendations
- Comprehensive report generation

Usage:
    python src/main.py [command] [options]

Commands:
    report      Generate a complete strategy report
    portfolio   Analyze portfolio health
    market      Get market research and macro context
    help        Show help information
"""

import sys
import argparse
import logging
from datetime import datetime
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('wealthfolio.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class WealthfolioOrchestrator:
    """Main orchestrator for the Wealthfolio Portfolio Strategy System."""
    
    def __init__(self, db_path: str = "db/weatlhfolio.db"):
        """
        Initialize the orchestrator.
        
        Args:
            db_path (str): Path to the wealthfolio database
        """
        self.db_path = db_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration settings."""
        return {
            "db_path": self.db_path,
            "report_format": "markdown",
            "output_dir": "reports",
            "llm_model": "gemini-1.5-flash",
            "analysis_depth": "comprehensive"
        }
    
    def generate_strategy_report(self, output_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a complete strategy report.
        
        Args:
            output_file (Optional[str]): Output file path for the report
            
        Returns:
            Dict[str, Any]: Report generation results
        """
        logger.info("Starting strategy report generation...")
        
        try:
            # Import modules dynamically to avoid circular imports
            import sys
            import os
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            
            from services.report_orchestrator import ReportOrchestrator
            
            # Initialize the report orchestrator
            orchestrator = ReportOrchestrator(self.db_path)
            
            # Generate the report
            report = orchestrator.generate_report()
            
            # Save to file if specified
            if output_file:
                import os
                # Ensure the directory exists
                output_dir = os.path.dirname(output_file)
                if output_dir:  # Only create directory if there is one
                    os.makedirs(output_dir, exist_ok=True)
                with open(output_file, 'w') as f:
                    f.write(report)
                logger.info(f"Report saved to: {output_file}")
            
            return {
                "success": True,
                "report": report,
                "output_file": output_file,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating strategy report: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def analyze_portfolio_health(self) -> Dict[str, Any]:
        """
        Analyze portfolio health and composition.
        
        Returns:
            Dict[str, Any]: Portfolio health analysis results
        """
        logger.info("Starting portfolio health analysis...")
        
        try:
            import sys
            import os
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            
            from core.portfolio import PortfolioAnalyzer
            from core.database import DatabaseManager
            
            # Initialize components
            db_manager = DatabaseManager(self.db_path)
            analyzer = PortfolioAnalyzer(db_manager)
            
            # Analyze portfolio health
            health_analysis = analyzer.analyze_portfolio_health()
            
            return {
                "success": True,
                "analysis": health_analysis,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing portfolio health: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def get_market_context(self) -> Dict[str, Any]:
        """
        Get current market research and macro context.
        
        Returns:
            Dict[str, Any]: Market context and research results
        """
        logger.info("Starting market context analysis...")
        
        try:
            from ai.market_oracle import get_macro_context
            
            # Get macro context
            macro_context = get_macro_context()
            
            return {
                "success": True,
                "macro_context": macro_context,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting market context: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def run_comprehensive_analysis(self, output_dir: str = "reports") -> Dict[str, Any]:
        """
        Run a comprehensive analysis including all components.
        
        Args:
            output_dir (str): Directory to save output files
            
        Returns:
            Dict[str, Any]: Comprehensive analysis results
        """
        logger.info("Starting comprehensive analysis...")
        
        results = {
            "portfolio_health": None,
            "market_context": None,
            "strategy_report": None,
            "success": False,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            # 1. Analyze portfolio health
            portfolio_result = self.analyze_portfolio_health()
            results["portfolio_health"] = portfolio_result
            
            if not portfolio_result["success"]:
                logger.warning("Portfolio health analysis failed, continuing with other analyses")
            
            # 2. Get market context
            market_result = self.get_market_context()
            results["market_context"] = market_result
            
            if not market_result["success"]:
                logger.warning("Market context analysis failed, continuing with report generation")
            
            # 3. Generate strategy report
            report_file = f"{output_dir}/strategy_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            report_result = self.generate_strategy_report(report_file)
            results["strategy_report"] = report_result
            
            # Determine overall success
            results["success"] = report_result["success"]
            
            logger.info("Comprehensive analysis completed")
            return results
            
        except Exception as e:
            logger.error(f"Error in comprehensive analysis: {str(e)}")
            results["error"] = str(e)
            return results


def main():
    """Main entry point for the Wealthfolio Portfolio Strategy System."""
    
    parser = argparse.ArgumentParser(
        description="Wealthfolio Portfolio Strategy System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/main.py report --output reports/strategy.md
  python src/main.py portfolio
  python src/main.py market
  python src/main.py comprehensive --output-dir reports/
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Report command
    report_parser = subparsers.add_parser('report', help='Generate a complete strategy report')
    report_parser.add_argument('--output', '-o', type=str, help='Output file path for the report')
    
    # Portfolio command
    portfolio_parser = subparsers.add_parser('portfolio', help='Analyze portfolio health')
    
    # Market command
    market_parser = subparsers.add_parser('market', help='Get market research and macro context')
    
    # Comprehensive command
    comprehensive_parser = subparsers.add_parser('comprehensive', help='Run comprehensive analysis')
    comprehensive_parser.add_argument('--output-dir', '-d', type=str, default='reports', help='Output directory for reports')
    
    # Help command
    subparsers.add_parser('help', help='Show help information')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Handle help command
    if args.command == 'help' or not args.command:
        parser.print_help()
        return
    
    try:
        # Initialize orchestrator
        orchestrator = WealthfolioOrchestrator()
        
        # Execute command
        if args.command == 'report':
            result = orchestrator.generate_strategy_report(args.output)
            if result["success"]:
                print("✅ Strategy report generated successfully!")
                if args.output:
                    print(f"📄 Report saved to: {args.output}")
                else:
                    print("📄 Report generated in memory")
            else:
                print(f"❌ Error generating report: {result['error']}")
        
        elif args.command == 'portfolio':
            result = orchestrator.analyze_portfolio_health()
            if result["success"]:
                print("✅ Portfolio health analysis completed!")
                analysis = result["analysis"]
                print(f"📊 Current Equity Weight: {analysis['current_equity_weight']}%")
                print(f"🎯 Target Range: {analysis['target_range']}")
                print(f"⚠️  Off Benchmark: {analysis['off_benchmark']}")
                print(f"📈 Total Holdings: {analysis['geographic_analysis']['total_holdings']}")
            else:
                print(f"❌ Error analyzing portfolio: {result['error']}")
        
        elif args.command == 'market':
            result = orchestrator.get_market_context()
            if result["success"]:
                print("✅ Market context analysis completed!")
                print("📋 Macro Context Summary:")
                print(result["macro_context"][:500] + "..." if len(result["macro_context"]) > 500 else result["macro_context"])
            else:
                print(f"❌ Error getting market context: {result['error']}")
        
        elif args.command == 'comprehensive':
            result = orchestrator.run_comprehensive_analysis(args.output_dir)
            if result["success"]:
                print("✅ Comprehensive analysis completed successfully!")
                print(f"📁 Output directory: {args.output_dir}")
            else:
                print(f"❌ Error in comprehensive analysis: {result.get('error', 'Unknown error')}")
        
        else:
            print(f"❌ Unknown command: {args.command}")
            parser.print_help()
    
    except KeyboardInterrupt:
        print("\n⚠️  Operation cancelled by user")
        sys.exit(1)
    
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        print(f"❌ Unexpected error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()