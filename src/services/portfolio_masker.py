#!/usr/bin/env python3
"""
Simple PortfolioMasker for testing
"""

import sqlite3
import json
import random
from typing import Dict, List, Any, Optional
from contextlib import contextmanager
from datetime import datetime


class SimplePortfolioMasker:
    """Simple privacy-preserving portfolio data processor."""
    
    def __init__(self, db_path: str = "db/weatlhfolio.db"):
        self.db_path = db_path
    
    @contextmanager
    def get_connection(self):
        conn = None
        try:
            conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
            conn.row_factory = sqlite3.Row
            yield conn
        except sqlite3.Error as e:
            print(f"Database error: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def _translate_spanish_to_english(self, text: str) -> str:
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
    
    def get_masked_portfolio_json(self) -> str:
        """Generate a simple masked portfolio JSON."""
        try:
            # Get holdings with values
            holdings = self._get_holdings_with_values()
            
            if not holdings:
                return json.dumps({
                    "portfolio": [],
                    "total_holdings": 0,
                    "timestamp": datetime.now().isoformat(),
                    "note": "No holdings found or database empty"
                }, indent=2)
            
            # Calculate total portfolio value
            total_value = sum(holding["current_value"] for holding in holdings)
            
            if total_value <= 0:
                return json.dumps({
                    "portfolio": [],
                    "total_holdings": 0,
                    "timestamp": datetime.now().isoformat(),
                    "note": "Portfolio has no value"
                }, indent=2)
            
            # Calculate weights based on actual values
            portfolio_data = []
            for holding in holdings:
                weight_percent = (holding["current_value"] / total_value) * 100
                
                categories = holding.get("categories", "")
                if categories:
                    # Translate categories
                    category_list = [cat.strip() for cat in categories.split(",")]
                    translated_categories = [
                        self._translate_spanish_to_english(cat) for cat in category_list
                    ]
                    categories = ", ".join(translated_categories)
                
                portfolio_data.append({
                    "ticker": holding["symbol"],
                    "weight_percent": round(weight_percent, 2),
                    "asset_type": holding["asset_type"],
                    "currency": holding["currency"],
                    "categories": categories if categories else "N/A"
                })
            
            # Sort by weight descending
            portfolio_data.sort(key=lambda x: x["weight_percent"], reverse=True)
            
            # Verify weights sum to approximately 100%
            total_weight = sum(holding["weight_percent"] for holding in portfolio_data)
            
            result = {
                "portfolio": portfolio_data,
                "total_holdings": len(portfolio_data),
                "total_weight_percent": round(total_weight, 2),
                "timestamp": datetime.now().isoformat(),
                "note": "Portfolio weights calculated from actual holdings values",
                "validation": {
                    "weights_sum_to_100": abs(total_weight - 100.0) < 0.1,
                    "no_true_nw_disclosed": True
                }
            }
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            return json.dumps({
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }, indent=2)
    
    def _get_holdings(self) -> List[Dict[str, Any]]:
        """Get holdings from database."""
        holdings = []
        
        with self.get_connection() as conn:
            # Get assets with positive quantities
            query = """
                SELECT 
                    a.id,
                    a.name,
                    a.asset_type,
                    a.symbol,
                    a.categories,
                    a.currency,
                    a.isin
                FROM assets a
                ORDER BY a.name
            """
            
            cursor = conn.execute(query)
            rows = cursor.fetchall()
            
            for row in rows:
                # Calculate current quantity
                quantity_query = """
                    SELECT 
                        activity_type,
                        quantity
                    FROM activities 
                    WHERE asset_id = ? AND is_draft = 0
                    ORDER BY activity_date
                """
                
                quantity_cursor = conn.execute(quantity_query, (row["id"],))
                quantity_rows = quantity_cursor.fetchall()
                
                current_quantity = 0.0
                for qty_row in quantity_rows:
                    activity_type = qty_row["activity_type"]
                    quantity = float(qty_row["quantity"])
                    
                    if activity_type in ["BUY", "DEPOSIT"]:
                        current_quantity += quantity
                    elif activity_type in ["SELL", "WITHDRAWAL"]:
                        current_quantity -= quantity
                
                if current_quantity > 0:
                    holdings.append({
                        "id": row["id"],
                        "name": row["name"],
                        "asset_type": row["asset_type"],
                        "symbol": row["symbol"],
                        "categories": row["categories"],
                        "currency": row["currency"],
                        "isin": row["isin"],
                        "current_quantity": current_quantity
                    })
        
        return holdings
    
    def _get_holdings_with_values(self) -> List[Dict[str, Any]]:
        """Get holdings from database with calculated values."""
        holdings = []
        
        with self.get_connection() as conn:
            # Get assets with positive quantities
            query = """
                SELECT 
                    a.id,
                    a.name,
                    a.asset_type,
                    a.symbol,
                    a.categories,
                    a.currency,
                    a.isin
                FROM assets a
                ORDER BY a.name
            """
            
            cursor = conn.execute(query)
            rows = cursor.fetchall()
            
            for row in rows:
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
                
                quantity_cursor = conn.execute(quantity_query, (row["id"],))
                quantity_rows = quantity_cursor.fetchall()
                
                current_quantity = 0.0
                current_value_gbp = 0.0
                
                for qty_row in quantity_rows:
                    activity_type = qty_row["activity_type"]
                    quantity = float(qty_row["quantity"])
                    unit_price = float(qty_row["unit_price"]) if qty_row["unit_price"] else 0.0
                    transaction_currency = qty_row["transaction_currency"]
                    account_currency = qty_row["account_currency"]
                    
                    # Calculate transaction value in transaction currency
                    transaction_value = quantity * unit_price
                    
                    if activity_type in ["BUY", "DEPOSIT"]:
                        current_quantity += quantity
                        current_value_gbp += self._convert_to_gbp(transaction_value, transaction_currency, conn)
                    elif activity_type in ["SELL", "WITHDRAWAL"]:
                        current_quantity -= quantity
                        current_value_gbp -= self._convert_to_gbp(transaction_value, transaction_currency, conn)
                
                if current_quantity > 0:
                    holdings.append({
                        "id": row["id"],
                        "name": row["name"],
                        "asset_type": row["asset_type"],
                        "symbol": row["symbol"],
                        "categories": row["categories"],
                        "currency": row["currency"],
                        "isin": row["isin"],
                        "current_quantity": current_quantity,
                        "current_value": current_value_gbp
                    })
        
        return holdings
    
    def _convert_to_gbp(self, amount: float, currency: str, conn) -> float:
        """
        Convert amount from given currency to GBP.
        
        Args:
            amount: Amount to convert
            currency: Source currency
            conn: Database connection
            
        Returns:
            float: Amount in GBP
        """
        if currency == "GBP" or currency == "GBp" or currency == "GBX":
            # GBX (pence) needs to be converted to GBP (pounds)
            if currency == "GBX":
                return amount / 100.0
            return amount
        
        # Get latest FX rate from quotes table
        fx_query = """
            SELECT close 
            FROM quotes 
            WHERE symbol = ? || 'GBP=X'
            ORDER BY timestamp DESC
            LIMIT 1
        """
        
        fx_cursor = conn.execute(fx_query, (currency,))
        fx_row = fx_cursor.fetchone()
        
        if fx_row:
            rate = float(fx_row["close"])
            return amount * rate
        else:
            # If no FX rate available, assume 1:1 (this shouldn't happen with proper data)
            print(f"Warning: No FX rate available for {currency}, assuming 1:1 conversion")
            return amount


def main():
    """Test the simple portfolio masker."""
    try:
        masker = SimplePortfolioMasker()
        portfolio_json = masker.get_masked_portfolio_json()
        
        print("Masked Portfolio for BlackRock Prompt:")
        print("=" * 50)
        print(portfolio_json)
        print("=" * 50)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()