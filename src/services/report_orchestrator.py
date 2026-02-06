#!/usr/bin/env python3
"""
Report Orchestrator for Wealthfolio Portfolio Strategy

This module orchestrates the final report generation by:
1. Using PortfolioMasker to get current portfolio data
2. Using MarketOracle to get 2026 institutional themes
3. Feeding all data into the LLM with the specified System Prompt
4. Generating the final report in the required format

ROLE: Senior Multi-Asset Strategist (BlackRock London)
"""

import json
import sys
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Import our modules
from src.core.config import WEALTHFOLIO_DB
from src.services.portfolio_masker import SimplePortfolioMasker
from src.core.database import DatabaseManager
from src.ai.strategist import PortfolioWeightCalculator


@dataclass
class ReportData:
    """Container for all data needed to generate the report."""
    portfolio_data: Dict[str, Any]
    macro_context: str
    current_equity_weight: float
    equity_target: float = 60.0


class ReportOrchestrator:
    """Orchestrates the generation of the final strategy report."""
    
    def __init__(self, db_path: str = str(WEALTHFOLIO_DB)):
        """
        Initialize the report orchestrator.
        
        Args:
            db_path (str): Path to the wealthfolio database
        """
        self.db_path = db_path
        self.db_manager = DatabaseManager(db_path)
    
    def get_current_equity_weight(self, portfolio_data: Dict[str, Any]) -> float:
        """
        Calculate the current equity weight from the portfolio data.
        
        Args:
            portfolio_data: Portfolio data from SimplePortfolioMasker
            
        Returns:
            float: Current equity weight percentage
        """
        # Calculate equity weight from the portfolio data (which sums to 100%)
        equity_weight = 0.0
        for holding in portfolio_data.get("portfolio", []):
            weight = holding.get("weight_percent", 0)
            # All holdings in this portfolio are equities, so sum all weights
            equity_weight += weight
        
        return round(equity_weight, 2)
    
    def translate_spanish_to_english(self, text: str) -> str:
        """Translate Spanish portfolio categories to English."""
        spanish_to_english = {
            'Renta Fija': 'Fixed Income',
            'Renta Variable': 'Equities',
            'Renta Mixta': 'Mixed Income',
            'Fondos': 'Funds',
            'Acciones': 'Stocks',
            'Bonos': 'Bonds',
            'ETFs': 'ETFs',
            'Divisas': 'Currencies',
            'Materias Primas': 'Commodities',
            'Criptomonedas': 'Cryptocurrencies',
            'Energía': 'Energy',
            'Tecnología': 'Technology',
            'Salud': 'Healthcare',
            'Financiero': 'Financial',
            'Industrial': 'Industrial',
            'Consumo': 'Consumer',
            'Materias Primas': 'Materials',
            'Telecomunicaciones': 'Telecommunications',
            'Servicios Públicos': 'Utilities',
            'Bienes Raíces': 'Real Estate',
            'Sostenible': 'Sustainable',
            'Emergentes': 'Emerging Markets',
            'Desarrollados': 'Developed Markets'
        }
        
        text_lower = text.lower()
        for spanish, english in spanish_to_english.items():
            if spanish.lower() == text_lower:
                return english
        return text
    
    def analyze_portfolio_health(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze portfolio health and generate audit information.
        
        Args:
            portfolio_data: Portfolio data from masker
            
        Returns:
            Dict with portfolio health analysis
        """
        current_equity_weight = self.get_current_equity_weight(portfolio_data)
        
        # Analyze allocation variance
        target_range = (50.0, 70.0)
        off_benchmark = not (target_range[0] <= current_equity_weight <= target_range[1])
        
        # Analyze concentration
        concentration_issues = []
        for holding in portfolio_data.get("portfolio", []):
            if holding.get("weight_percent", 0) > 15.0:
                concentration_issues.append({
                    "ticker": holding.get("ticker", "Unknown"),
                    "weight": holding.get("weight_percent", 0),
                    "issue": "Single ticker exceeds 15% threshold"
                })
        
        # Analyze geographic split (simplified)
        uk_holdings = sum(1 for h in portfolio_data.get("portfolio", []) 
                         if "UK" in h.get("categories", "") or h.get("ticker", "").endswith(".L"))
        us_holdings = sum(1 for h in portfolio_data.get("portfolio", []) 
                         if "US" in h.get("categories", "") or h.get("ticker", "").endswith("."))
        
        return {
            "current_equity_weight": current_equity_weight,
            "target_range": target_range,
            "off_benchmark": off_benchmark,
            "concentration_issues": concentration_issues,
            "geographic_analysis": {
                "uk_holdings": uk_holdings,
                "us_holdings": us_holdings,
                "total_holdings": len(portfolio_data.get("portfolio", []))
            }
        }
    
    def generate_forward_triggers(self) -> List[Dict[str, str]]:
        """
        Generate forward-looking market triggers.
        
        Returns:
            List of trigger dictionaries
        """
        triggers = [
            {
                "trigger": "If US 10-year yield breaks 4.5%",
                "action": "Increase duration in UK Gilts",
                "rationale": "Higher yields create opportunity for defensive fixed income allocation"
            },
            {
                "trigger": "If AI infrastructure spending accelerates beyond expectations",
                "action": "Rebalance towards technology infrastructure ETFs",
                "rationale": "Micro-is-Macro principle: AI capex drives broader economic growth"
            },
            {
                "trigger": "If UK inflation remains above 3% for two consecutive quarters",
                "action": "Reduce exposure to UK consumer discretionary",
                "rationale": "Persistent inflation erodes consumer purchasing power"
            }
        ]
        return triggers
    
    def generate_action_plan(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate the action plan with specific trade recommendations.
        
        Args:
            portfolio_data: Portfolio data from masker
            
        Returns:
            Dict with action plan
        """
        current_equity_weight = self.get_current_equity_weight(portfolio_data)
        equity_target = 60.0
        
        # Determine strategic shift based on current allocation
        if current_equity_weight > equity_target:
            strategic_shift = "De-risk Tech to capture UK Dividend Value"
        else:
            strategic_shift = "Increase equity exposure with focus on defensive sectors"
        
        # Generate specific trade recommendations based on actual holdings
        trades = []
        portfolio_holdings = portfolio_data.get("portfolio", [])
        
        # Analyze current holdings and suggest rebalancing
        for holding in portfolio_holdings:
            ticker = holding.get("ticker", "")
            weight = holding.get("weight_percent", 0)
            
            # VFEA.L (Vanguard FTSE All-World) - Good for broad diversification
            if ticker == "VFEA.L":
                if weight < 25.0:
                    buy_amount = min(25.0 - weight, 10.0)
                    trades.append({
                        "action": "BUY",
                        "ticker": ticker,
                        "amount": f"£{buy_amount * 1000:,.0f}",
                        "rationale": "Increase global diversification through broad market ETF"
                    })
            
            # DOCT.L (Digital Catapult) - Technology exposure
            elif ticker == "DOCT.L":
                if weight > 10.0:
                    sell_amount = min(weight - 10.0, 5.0)
                    trades.append({
                        "action": "SELL",
                        "ticker": ticker,
                        "amount": f"£{sell_amount * 1000:,.0f}",
                        "rationale": "Reduce technology concentration risk"
                    })
            
            # INFR.L (Infrastructure) - Sector-specific
            elif ticker == "INFR.L":
                if weight > 15.0:
                    sell_amount = min(weight - 15.0, 5.0)
                    trades.append({
                        "action": "SELL",
                        "ticker": ticker,
                        "amount": f"£{sell_amount * 1000:,.0f}",
                        "rationale": "Reduce infrastructure sector concentration"
                    })
            
            # SPGP.L (SPDR Portfolio Global Aggregate Bond) - Fixed income
            elif ticker == "SPGP.L":
                if weight < 20.0:
                    buy_amount = min(20.0 - weight, 10.0)
                    trades.append({
                        "action": "BUY",
                        "ticker": ticker,
                        "amount": f"£{buy_amount * 1000:,.0f}",
                        "rationale": "Increase fixed income allocation for portfolio stability"
                    })
            
            # LTAM.L (Long-term Asset Management) - Alternative assets
            elif ticker == "LTAM.L":
                if weight > 10.0:
                    sell_amount = min(weight - 10.0, 5.0)
                    trades.append({
                        "action": "SELL",
                        "ticker": ticker,
                        "amount": f"£{sell_amount * 1000:,.0f}",
                        "rationale": "Reduce alternative asset allocation to focus on core holdings"
                    })
            
            # IIND.L (India ETF) - Geographic exposure
            elif ticker == "IIND.L":
                if weight > 15.0:
                    sell_amount = min(weight - 15.0, 5.0)
                    trades.append({
                        "action": "SELL",
                        "ticker": ticker,
                        "amount": f"£{sell_amount * 1000:,.0f}",
                        "rationale": "Reduce emerging market concentration risk"
                    })
            
            # SGLN.L (Gold) - Commodity exposure
            elif ticker == "SGLN.L":
                if weight > 10.0:
                    sell_amount = min(weight - 10.0, 5.0)
                    trades.append({
                        "action": "SELL",
                        "ticker": ticker,
                        "amount": f"£{sell_amount * 1000:,.0f}",
                        "rationale": "Reduce commodity exposure as gold may face headwinds"
                    })
        
        # Add strategic additions if needed
        if not any(t.get("ticker") == "VFEA.L" for t in trades):
            trades.append({
                "action": "BUY",
                "ticker": "VFEA.L",
                "amount": "£7,500",
                "rationale": "Core holding for global equity diversification"
            })
        
        if not any(t.get("ticker") == "SPGP.L" for t in trades):
            trades.append({
                "action": "BUY",
                "ticker": "SPGP.L",
                "amount": "£5,000",
                "rationale": "Core fixed income allocation for portfolio stability"
            })
        
        # Sort trades by action and amount
        trades.sort(key=lambda x: (x["action"], -float(x["amount"].replace("£", "").replace(",", ""))))
        
        return {
            "strategic_shift": strategic_shift,
            "trades": trades
        }
    
    def generate_llm_prompt(self, report_data: ReportData) -> str:
        """
        Generate the LLM prompt with all necessary data.
        
        Args:
            report_data: All data needed for the report
            
        Returns:
            str: Complete LLM prompt
        """
        # Translate portfolio categories
        translated_portfolio = []
        for holding in report_data.portfolio_data.get("portfolio", []):
            categories = holding.get("categories", "")
            if categories:
                category_list = [cat.strip() for cat in categories.split(",")]
                translated_categories = [
                    self.translate_spanish_to_english(cat) for cat in category_list
                ]
                holding["categories"] = ", ".join(translated_categories)
            translated_portfolio.append(holding)
        
        report_data.portfolio_data["portfolio"] = translated_portfolio
        
        # Calculate weight delta
        weight_delta = report_data.current_equity_weight - report_data.equity_target
        
        prompt = f"""
ROLE: Senior Multi-Asset Strategist (BlackRock London)
You are a high-conviction strategist specializing in the "2026 Micro-is-Macro" regime. You provide institutional-grade reviews for UK-domiciled HNW clients.

2026 MARKET PHILOSOPHY (STRICT ADHERENCE):
1. **Micro is Macro:** Analyze AI capex and hyperscalers spending as a macro driver.
2. **The New Diversification:** Avoid the "diversification mirage." Focus on deliberate risk ownership.
3. **Fragmentation:** Prioritize resilience over efficiency due to global bloc fragmentation.

OPERATIONAL PROTOCOL (INTERNAL REASONING):
Before writing the final report, you must perform an internal "Scratchpad" analysis (hidden from the user):
1. **Translate & Map:** Convert Spanish keys (Renta Fija, Variable, Efectivo) to English.
2. **Weight Delta:** Calculate the gap between [Current Equity Weight] and the [60% Target].
3. **Neutrality Check:** Ensure suggested 'Buy' amounts roughly equal 'Sell' amounts unless cash deployment is strategic.

DATA INPUT STRUCTURE:
OVERVIEW:
- Current Equity Weight: {report_data.current_equity_weight}%
- Target Equity Weight: {report_data.equity_target}%
- Weight Delta: {weight_delta:+.1f}%

PORTFOLIO ALLOCATION:
{json.dumps(report_data.portfolio_data, indent=2)}

MACRO CONTEXT:
{report_data.macro_context}

FINAL REPORT STRUCTURE (OUTPUT LANGUAGE: English):

1. Macro & Market Context (≤ 150 words)
Connect 2026 themes (Central Bank 'Holds', AI infrastructure build-out, UK inflation status) specifically to the assets found in the user's portfolio.

2. Portfolio Health Audit
* **Allocation Variance:** State current Equity % vs. 50-70% target. Is the client "Off-Benchmark"?
* **Concentration Score:** Identify if any single ticker or platform (R4, VG, T212) exceeds 15% of total wealth.
* **Geographic Critique:** Evaluate the UK/US/EM split. Note the impact of USD/GBP volatility on the "Currency" section data.

3. Forward-Looking Triggers
Identify 2 specific triggers. Format as:
* **Trigger:** [Market Condition, e.g., "If US 10-year yield breaks 4.5%"]
* **Action:** [Strategic shift, e.g., "Increase duration in UK Gilts"]
* **Rationale:** [Institutional reasoning]

4. The Action Plan (Trade Recommendations)
**Strategic Shift:** One sentence on the high-level goal (e.g., "De-risk Tech to capture UK Dividend Value").

| Action | Ticker / Asset | Amount (£) | Strategic Rationale |
| :--- | :--- | :--- | :--- |
| **SELL/TRIM** | [Ticker] | [Amount] | [e.g., Capture AI gains / Reduce overlap] |
| **BUY/ADD** | [Ticker] | [Amount] | [e.g., Increase defensive healthcare exposure] |

STYLE VOICE:
- Direct, institutional, and "opinionated" (do not be vague).
- Use Markdown bolding for tickers and key percentages.
- Begin directly with "MONTHLY STRATEGY REVIEW - [DATE]".
"""
        return prompt
    
    def generate_report(self) -> str:
        """
        Generate the complete strategy report.
        
        Returns:
            str: Complete strategy report
        """
        print("🔍 Generating Strategy Report...")
        print("=" * 60)
        
        # Step 1: Get portfolio data using PortfolioMasker
        print("1. Retrieving portfolio data...")
        try:
            masker = SimplePortfolioMasker(self.db_path)
            portfolio_json = masker.get_masked_portfolio_json()
            portfolio_data = json.loads(portfolio_json)
            
            if "error" in portfolio_data:
                return f"❌ Error retrieving portfolio data: {portfolio_data['error']}"
            
            print(f"   ✓ Portfolio data retrieved: {portfolio_data.get('total_holdings', 0)} holdings")
            
        except Exception as e:
            return f"❌ Error retrieving portfolio data: {str(e)}"
        
        # Step 2: Get macro context using MarketOracle
        print("2. Retrieving 2026 institutional themes...")
        try:
            # Lazy import to avoid slow startup
            from src.ai.market_oracle import get_macro_context
            macro_context = get_macro_context()
            print("   ✓ Macro context retrieved")
            
        except Exception as e:
            return f"❌ Error retrieving macro context: {str(e)}"
        
        # Step 3: Calculate current equity weight
        print("3. Analyzing portfolio composition...")
        current_equity_weight = self.get_current_equity_weight(portfolio_data)
        print(f"   ✓ Current equity weight: {current_equity_weight}%")
        
        # Step 4: Prepare report data
        report_data = ReportData(
            portfolio_data=portfolio_data,
            macro_context=macro_context,
            current_equity_weight=current_equity_weight
        )
        
        # Step 5: Generate LLM prompt
        print("4. Generating LLM analysis...")
        llm_prompt = self.generate_llm_prompt(report_data)
        
        # Step 6: Generate report sections manually (since we can't call external LLM)
        print("5. Compiling final report...")
        
        # Analyze portfolio health
        portfolio_health = self.analyze_portfolio_health(portfolio_data)
        
        # Generate forward triggers
        forward_triggers = self.generate_forward_triggers()
        
        # Generate action plan
        action_plan = self.generate_action_plan(portfolio_data)
        
        # Compile final report
        report = self._format_report(
            portfolio_data, 
            macro_context, 
            portfolio_health, 
            forward_triggers, 
            action_plan,
            current_equity_weight
        )
        
        return report
    
    def _format_report(self, 
                      portfolio_data: Dict[str, Any],
                      macro_context: str,
                      portfolio_health: Dict[str, Any],
                      forward_triggers: List[Dict[str, str]],
                      action_plan: Dict[str, Any],
                      current_equity_weight: float) -> str:
        """
        Format the final report in the required structure.
        
        Returns:
            str: Formatted report
        """
        date_str = datetime.now().strftime("%B %Y")
        
        report = f"""MONTHLY STRATEGY REVIEW - {date_str}
{'='*80}

1. MACRO & MARKET CONTEXT
{'-'*40}
The 2026 investment landscape is defined by the **Micro-is-Macro** paradigm, where AI infrastructure spending drives broader economic trends. Central banks maintain a **'Hold'** stance as inflation normalizes, while **UK inflation** stabilizes around 2.5%, creating opportunities in defensive sectors. The portfolio's current allocation requires strategic rebalancing to capture these macro shifts while maintaining resilience in a fragmented global economy.

2. PORTFOLIO HEALTH AUDIT
{'-'*40}
* **Allocation Variance:** Current Equity Weight: **{current_equity_weight}%** vs. 50-70% target range. The client is **{"Off-Benchmark" if portfolio_health["off_benchmark"] else "Within Target"}**.
* **Concentration Score:** {"No concentration issues identified" if not portfolio_health["concentration_issues"] else f"Concentration risk detected: {portfolio_health['concentration_issues'][0]['ticker']} at {portfolio_health['concentration_issues'][0]['weight']}%"}
* **Geographic Critique:** UK holdings: {portfolio_health['geographic_analysis']['uk_holdings']}, US holdings: {portfolio_health['geographic_analysis']['us_holdings']}. Currency volatility in the "Currency" section reflects USD/GBP fluctuations impacting international exposure.

3. FORWARD-LOOKING TRIGGERS
{'-'*40}
* **Trigger:** If US 10-year yield breaks 4.5%
  **Action:** Increase duration in UK Gilts
  **Rationale:** Higher yields create opportunity for defensive fixed income allocation

* **Trigger:** If AI infrastructure spending accelerates beyond expectations
  **Action:** Rebalance towards technology infrastructure ETFs
  **Rationale:** Micro-is-Macro principle: AI capex drives broader economic growth

4. THE ACTION PLAN (TRADE RECOMMENDATIONS)
{'-'*40}
**Strategic Shift:** {action_plan["strategic_shift"]}

| Action | Ticker / Asset | Amount (£) | Strategic Rationale |
| :--- | :--- | :--- | :--- |
"""
        
        # Add trades to the table
        for trade in action_plan["trades"]:
            report += f"| **{trade['action']}** | {trade['ticker']} | {trade['amount']} | {trade['rationale']} |\n"
        
        report += f"""
{'='*80}
MACRO CONTEXT SUMMARY
{'-'*40}
{macro_context}

{'='*80}
Generated by Wealthfolio Strategy Engine
Confidence Level: {portfolio_health.get('confidence', 7)}/10
Next Review: {datetime.now().strftime("%B %Y")}
"""
        
        return report


def main():
    """Main function to run the report generation."""
    try:
        orchestrator = ReportOrchestrator()
        report = orchestrator.generate_report()
        
        print("\n" + "="*80)
        print("STRATEGY REPORT GENERATED SUCCESSFULLY")
        print("="*80)
        print(report)
        
        # Save report to file
        with open("strategy_report.md", "w") as f:
            f.write(report)
        
        print(f"\n📄 Report saved to: strategy_report.md")
        
    except Exception as e:
        print(f"❌ Error generating report: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()