#!/usr/bin/env python3
"""
Database Manager for Wealthfolio Portfolio

This module provides read-only access to the wealthfolio database
and functions to retrieve portfolio summary information.
"""

import sqlite3
import json
from typing import List, Dict, Any, Optional
from contextlib import contextmanager
from src.core.config import WEALTHFOLIO_DB


class DatabaseManager:
    """Manages read-only connection to the wealthfolio database."""
    
    def __init__(self, db_path: str = str(WEALTHFOLIO_DB)):
        """
        Initialize the database manager.
        
        Args:
            db_path (str): Path to the SQLite database file
        """
        self.db_path = db_path
    
    @contextmanager
    def get_connection(self):
        """
        Context manager for database connections.
        
        Ensures connections are properly closed and provides read-only access.
        """
        conn = None
        try:
            # Open database in read-only mode
            conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
            conn.row_factory = sqlite3.Row  # Enable dict-like access to rows
            yield conn
        except sqlite3.Error as e:
            print(f"Database error: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def get_portfolio_summary(self) -> List[Dict[str, Any]]:
        """
        Get a summary of all assets in the portfolio with their categories and current quantities.
        
        Returns:
            List[Dict[str, Any]]: List of assets with their details including current quantities
            
        The function calculates current quantities by summing all activities for each asset,
        handling different activity types (BUY, SELL, DIVIDEND, etc.) appropriately.
        """
        with self.get_connection() as conn:
            # First, get all assets with their metadata
            assets_query = """
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
            
            assets_cursor = conn.execute(assets_query)
            assets_rows = assets_cursor.fetchall()
            
            # Convert to list of dictionaries
            portfolio_summary = []
            for row in assets_rows:
                asset_data = {
                    "id": row["id"],
                    "name": row["name"],
                    "asset_type": row["asset_type"],
                    "symbol": row["symbol"],
                    "categories": row["categories"],
                    "currency": row["currency"],
                    "isin": row["isin"],
                    "current_quantity": "0"  # Will be calculated below
                }
                portfolio_summary.append(asset_data)
            
            # Calculate current quantities for each asset
            for asset in portfolio_summary:
                quantity_query = """
                    SELECT 
                        activity_type,
                        quantity
                    FROM activities 
                    WHERE asset_id = ? AND is_draft = 0
                    ORDER BY activity_date
                """
                
                quantity_cursor = conn.execute(quantity_query, (asset["id"],))
                quantity_rows = quantity_cursor.fetchall()
                
                current_quantity = 0.0
                for qty_row in quantity_rows:
                    activity_type = qty_row["activity_type"]
                    quantity = float(qty_row["quantity"])
                    
                    # Handle different activity types
                    if activity_type in ["BUY", "DEPOSIT"]:
                        current_quantity += quantity
                    elif activity_type in ["SELL", "WITHDRAWAL"]:
                        current_quantity -= quantity
                    # For other types like DIVIDEND, INTEREST, etc., they don't affect quantity
                    # so we don't modify current_quantity for those
                
                asset["current_quantity"] = str(current_quantity)
            
            return portfolio_summary


def get_portfolio_summary() -> List[Dict[str, Any]]:
    """
    Convenience function to get portfolio summary.
    
    Returns:
        List[Dict[str, Any]]: Portfolio summary as JSON-serializable list
    """
    db_manager = DatabaseManager()
    return db_manager.get_portfolio_summary()


if __name__ == "__main__":
    # Example usage
    try:
        summary = get_portfolio_summary()
        print(json.dumps(summary, indent=2))
    except Exception as e:
        print(f"Error: {e}")