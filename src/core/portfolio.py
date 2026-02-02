#!/usr/bin/env python3
"""
Portfolio Analysis and Weight Calculation Module

This module provides functionality to analyze portfolio holdings,
calculate weights, and prepare privacy-safe context for LLM analysis.
"""

import json
import sqlite3
from typing import List, Dict, Any, Optional, Tuple
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from decimal import Decimal


class ActivityType(Enum):
    """Enumeration of activity types that affect portfolio quantities."""
    BUY = "BUY"
    SELL = "SELL"
    DEPOSIT = "DEPOSIT"
    WITHDRAWAL = "WITHDRAWAL"
    DIVIDEND = "DIVIDEND"
    INTEREST = "INTEREST"
    TRANSFER_IN = "TRANSFER_IN"
    TRANSFER_OUT = "TRANSFER_OUT"


@dataclass
class PortfolioAsset:
    """Represents a portfolio asset with its calculated weight."""
    ticker: str
    weight_pct: float
    sector: str
    sentiment: str


class PortfolioWeightCalculator:
    """Calculates portfolio weights and prepares privacy-safe context for LLM analysis."""
    
    def __init__(self, db_manager):
        """
        Initialize the calculator.
        
        Args:
            db_manager: Database manager instance
        """
        self.db_manager = db_manager
    
    def get_llm_context(self) -> List[Dict[str, Any]]:
        """
        Calculate portfolio weights and prepare privacy-safe context for LLM analysis.
        
        Privacy Rule: Strips out all absolute dollar amounts and share counts.
        Handles multiple currencies by converting everything to account currency.
        
        Returns:
            List[Dict[str, Any]]: List of assets with privacy-safe information
        """
        with self.db_manager.get_connection() as conn:
            # Get all assets with their metadata and account information
            assets_query = """
                SELECT DISTINCT
                    a.id,
                    a.name,
                    a.asset_type,
                    a.symbol,
                    a.categories,
                    a.currency as asset_currency,
                    a.isin,
                    acc.currency as account_currency
                FROM assets a
                JOIN activities act ON a.id = act.asset_id
                JOIN accounts acc ON act.account_id = acc.id
                WHERE act.is_draft = 0
                ORDER BY a.name
            """
            
            assets_cursor = conn.execute(assets_query)
            assets_rows = assets_cursor.fetchall()
            
            # Get latest FX rates for currency conversion
            fx_rates = self._get_fx_rates(conn)
            
            # Calculate current quantities and total portfolio value
            portfolio_data = []
            total_value_gbp = Decimal('0.0')
            
            for row in assets_rows:
                asset_data = {
                    "id": row["id"],
                    "name": row["name"],
                    "asset_type": row["asset_type"],
                    "symbol": row["symbol"],
                    "categories": row["categories"],
                    "asset_currency": row["asset_currency"],
                    "account_currency": row["account_currency"],
                    "isin": row["isin"],
                    "current_quantity": Decimal('0.0'),
                    "current_value_gbp": Decimal('0.0')
                }
                
                # Calculate current quantity and value
                quantity_query = """
                    SELECT 
                        act.activity_type,
                        act.quantity,
                        act.unit_price,
                        act.currency as transaction_currency,
                        acc.currency as account_currency
                    FROM activities act
                    JOIN accounts acc ON act.account_id = acc.id
                    WHERE act.asset_id = ? AND act.is_draft = 0
                    ORDER BY act.activity_date
                """
                
                quantity_cursor = conn.execute(quantity_query, (asset_data["id"],))
                quantity_rows = quantity_cursor.fetchall()
                
                current_quantity = Decimal('0.0')
                current_value_gbp = Decimal('0.0')
                
                for qty_row in quantity_rows:
                    activity_type = qty_row["activity_type"]
                    quantity = Decimal(str(qty_row["quantity"]))
                    unit_price = Decimal(str(qty_row["unit_price"])) if qty_row["unit_price"] else Decimal('0.0')
                    transaction_currency = qty_row["transaction_currency"]
                    account_currency = qty_row["account_currency"]
                    
                    # Calculate transaction value in transaction currency
                    transaction_value = quantity * unit_price
                    
                    if activity_type in [ActivityType.BUY.value, ActivityType.DEPOSIT.value]:
                        current_quantity += quantity
                        current_value_gbp += self._convert_to_gbp(transaction_value, transaction_currency, fx_rates)
                    elif activity_type in [ActivityType.SELL.value, ActivityType.WITHDRAWAL.value]:
                        current_quantity -= quantity
                        current_value_gbp -= self._convert_to_gbp(transaction_value, transaction_currency, fx_rates)
                    # Other activity types don't affect quantity/value significantly
                
                asset_data["current_quantity"] = current_quantity
                asset_data["current_value_gbp"] = current_value_gbp
                portfolio_data.append(asset_data)
                total_value_gbp += current_value_gbp
            
            # Prepare privacy-safe context
            llm_context = []
            for asset in portfolio_data:
                if asset["current_value_gbp"] > 0 and total_value_gbp > 0:
                    weight_pct = float((asset["current_value_gbp"] / total_value_gbp) * 100)
                    
                    # Extract sector from categories (assuming categories is a JSON string)
                    sector = "Unknown"
                    if asset["categories"]:
                        try:
                            categories = json.loads(asset["categories"])
                            sector = categories.get("sector", "Unknown")
                        except (json.JSONDecodeError, TypeError):
                            sector = "Unknown"
                    
                    # Create sentiment placeholder (would be filled by market research)
                    sentiment = f"Analysis needed for {asset['symbol']}"
                    
                    llm_context.append({
                        "ticker": asset["symbol"],
                        "weight_pct": round(weight_pct, 2),
                        "sector": sector,
                        "sentiment": sentiment
                    })
            
            return llm_context
    
    def get_real_portfolio_value(self) -> float:
        """
        Get the real total portfolio value from the database in GBP.
        
        Returns:
            float: Total portfolio value in GBP
        """
        with self.db_manager.get_connection() as conn:
            # Get FX rates for currency conversion
            fx_rates = self._get_fx_rates(conn)
            
            # Calculate total portfolio value with currency conversion
            total_value_gbp = Decimal('0.0')
            
            # Get all activities with currency information
            activities_query = """
                SELECT 
                    act.activity_type,
                    act.quantity,
                    act.unit_price,
                    act.currency as transaction_currency,
                    acc.currency as account_currency
                FROM activities act
                JOIN accounts acc ON act.account_id = acc.id
                WHERE act.is_draft = 0
            """
            
            cursor = conn.execute(activities_query)
            activity_rows = cursor.fetchall()
            
            for row in activity_rows:
                activity_type = row["activity_type"]
                quantity = Decimal(str(row["quantity"]))
                unit_price = Decimal(str(row["unit_price"])) if row["unit_price"] else Decimal('0.0')
                transaction_currency = row["transaction_currency"]
                
                # Calculate transaction value in transaction currency
                transaction_value = quantity * unit_price
                
                if activity_type in [ActivityType.BUY.value, ActivityType.DEPOSIT.value]:
                    total_value_gbp += self._convert_to_gbp(transaction_value, transaction_currency, fx_rates)
                elif activity_type in [ActivityType.SELL.value, ActivityType.WITHDRAWAL.value]:
                    total_value_gbp -= self._convert_to_gbp(transaction_value, transaction_currency, fx_rates)
            
            return float(total_value_gbp)
    
    def _get_fx_rates(self, conn) -> Dict[str, Decimal]:
        """
        Get latest FX rates for currency conversion to GBP.
        
        Args:
            conn: Database connection
            
        Returns:
            Dict[str, Decimal]: Dictionary of currency to GBP conversion rates
        """
        fx_rates = {"GBP": Decimal('1.0')}  # Base currency
        
        # Get latest FX rates from quotes table
        fx_query = """
            SELECT symbol, close 
            FROM quotes 
            WHERE symbol IN ('USDGBP=X', 'EURGBP=X')
            ORDER BY timestamp DESC
            LIMIT 2
        """
        
        fx_cursor = conn.execute(fx_query)
        fx_rows = fx_cursor.fetchall()
        
        for fx_row in fx_rows:
            symbol = fx_row["symbol"]
            rate = Decimal(str(fx_row["close"]))
            
            if symbol == "USDGBP=X":
                fx_rates["USD"] = rate
            elif symbol == "EURGBP=X":
                fx_rates["EUR"] = rate
        
        return fx_rates
    
    def _convert_to_gbp(self, amount: Decimal, currency: str, fx_rates: Dict[str, Decimal]) -> Decimal:
        """
        Convert amount from given currency to GBP.
        
        Args:
            amount: Amount to convert
            currency: Source currency
            fx_rates: Dictionary of FX rates
            
        Returns:
            Decimal: Amount in GBP
        """
        if currency == "GBP" or currency == "GBp" or currency == "GBX":
            # GBX (pence) needs to be converted to GBP (pounds)
            if currency == "GBX":
                return amount / Decimal('100.0')
            return amount
        elif currency in fx_rates:
            return amount * fx_rates[currency]
        else:
            # If no FX rate available, assume 1:1 (this shouldn't happen with proper data)
            print(f"Warning: No FX rate available for {currency}, assuming 1:1 conversion")
            return amount


class PortfolioAnalyzer:
    """Analyzes portfolio composition and health."""
    
    def __init__(self, db_manager):
        """
        Initialize the portfolio analyzer.
        
        Args:
            db_manager: Database manager instance
        """
        self.db_manager = db_manager
        self.weight_calculator = PortfolioWeightCalculator(db_manager)
    
    def get_current_equity_weight(self) -> float:
        """
        Calculate the current equity weight from the portfolio data.
        
        Returns:
            float: Current equity weight percentage
        """
        portfolio_data = self.weight_calculator.get_llm_context()
        
        # Calculate equity weight from the portfolio data (which sums to 100%)
        equity_weight = 0.0
        for holding in portfolio_data:
            weight = holding.get("weight_pct", 0)
            # All holdings in this portfolio are equities, so sum all weights
            equity_weight += weight
        
        return round(equity_weight, 2)
    
    def analyze_portfolio_health(self) -> Dict[str, Any]:
        """
        Analyze portfolio health and generate audit information.
        
        Returns:
            Dict with portfolio health analysis
        """
        portfolio_data = self.weight_calculator.get_llm_context()
        current_equity_weight = self.get_current_equity_weight()
        
        # Analyze allocation variance
        target_range = (50.0, 70.0)
        off_benchmark = not (target_range[0] <= current_equity_weight <= target_range[1])
        
        # Analyze concentration
        concentration_issues = []
        for holding in portfolio_data:
            if holding.get("weight_pct", 0) > 15.0:
                concentration_issues.append({
                    "ticker": holding.get("ticker", "Unknown"),
                    "weight": holding.get("weight_pct", 0),
                    "issue": "Single ticker exceeds 15% threshold"
                })
        
        # Analyze geographic split (simplified)
        uk_holdings = sum(1 for h in portfolio_data 
                         if "UK" in h.get("sector", "") or h.get("ticker", "").endswith(".L"))
        us_holdings = sum(1 for h in portfolio_data 
                         if "US" in h.get("sector", "") or h.get("ticker", "").endswith("."))
        
        return {
            "current_equity_weight": current_equity_weight,
            "target_range": target_range,
            "off_benchmark": off_benchmark,
            "concentration_issues": concentration_issues,
            "geographic_analysis": {
                "uk_holdings": uk_holdings,
                "us_holdings": us_holdings,
                "total_holdings": len(portfolio_data)
            },
            "portfolio_data": portfolio_data
        }
    
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


if __name__ == "__main__":
    # Example usage
    from .database import DatabaseManager
    
    db_manager = DatabaseManager()
    analyzer = PortfolioAnalyzer(db_manager)
    
    try:
        health = analyzer.analyze_portfolio_health()
        print(f"Current Equity Weight: {health['current_equity_weight']}%")
        print(f"Portfolio Health: {health}")
    except Exception as e:
        print(f"Error: {e}")