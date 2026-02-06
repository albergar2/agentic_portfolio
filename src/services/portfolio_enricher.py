import sqlite3
import json
import logging
from typing import List, Dict, Any, Optional
from src.core.config import WEALTHFOLIO_DB, ENRICHMENT_DB
from src.services.ai_enricher import AIEnricher
from src.core.privacy import PrivacyManager

logger = logging.getLogger(__name__)

# Try to import yfinance
try:
    import yfinance as yf
except ImportError:
    logger.error("yfinance not found. Please install it: pip install yfinance")
    yf = None

class PortfolioEnricher:
    """Enriches portfolio assets with data from yfinance and Gemini AI."""
    
    def __init__(self, main_db: str = str(WEALTHFOLIO_DB), enrichment_db: str = str(ENRICHMENT_DB)):
        self.main_db = main_db
        self.enrichment_db = enrichment_db
        self._ensure_enrichment_table()
        self.ai_enricher = AIEnricher()

    def _ensure_enrichment_table(self):
        """Creates the enrichment table if it doesn't exist."""
        conn = sqlite3.connect(self.enrichment_db)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS asset_enrichment (
                symbol TEXT PRIMARY KEY,
                name TEXT,
                isin TEXT,
                sector TEXT,
                industry TEXT,
                country_exposure TEXT,
                currency_exposure TEXT,
                is_hedged BOOLEAN,
                long_business_summary TEXT,
                full_metadata TEXT,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()

    def get_symbols(self) -> List[Dict[str, str]]:
        """Retrieves unique symbols from the main database."""
        conn = sqlite3.connect(f"file:{self.main_db}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        cursor = conn.execute("SELECT DISTINCT symbol, name, isin FROM assets WHERE symbol IS NOT NULL AND symbol != ''")
        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return rows

    def enrich_symbol(self, symbol: str, name: str = "", isin: str = "") -> Optional[Dict[str, Any]]:
        """Fetches data from yfinance and falls back to Gemini AI if needed."""
        # Redact any values from inputs
        safe_symbol = PrivacyManager.redact_text(symbol)
        safe_name = PrivacyManager.redact_text(name)
        safe_isin = PrivacyManager.redact_text(isin)

        enrichment = None
        
        # 1. Try yfinance first for basic data
        if yf:
            logger.info(f"Fetching yfinance data for {safe_symbol}...")
            try:
                ticker = yf.Ticker(safe_symbol)
                info = ticker.info
                
                if info:
                    long_name = info.get("longName") or info.get("shortName") or name or symbol
                    is_hedged = any(word in long_name.lower() for word in ["hedged", "hedg", " h-", "-h "])
                    
                    enrichment = {
                        "symbol": symbol,
                        "name": long_name,
                        "isin": info.get("isin") or isin,
                        "sector": info.get("sector") or info.get("category"),
                        "industry": info.get("industry"),
                        "long_business_summary": info.get("longBusinessSummary"),
                        "full_metadata": json.dumps(info),
                        "is_hedged": is_hedged,
                        "country_exposure": json.dumps({info.get("country"): 100}) if info.get("country") else None,
                        "currency_exposure": json.dumps({info.get("currency"): 100}) if info.get("currency") else None,
                    }
            except Exception as e:
                logger.warning(f"yfinance failed for {symbol}: {e}")

        # 2. Check if enrichment is sufficient. If not, fallback to AI.
        needs_ai = (
            not enrichment or 
            not enrichment.get("sector") or 
            not enrichment.get("country_exposure") or
            symbol.startswith("0P")  # Funds usually need better data
        )
        
        if needs_ai:
            logger.info(f"Using AI fallback for {symbol} ({name})...")
            ai_data = self.ai_enricher.enrich_asset(symbol, name, isin)
            if ai_data:
                if not enrichment:
                    enrichment = {"symbol": symbol, "full_metadata": "{}"}
                
                # Merge AI data, giving it priority for missing/better fields
                enrichment["name"] = ai_data.get("name") or enrichment.get("name") or name or symbol
                enrichment["sector"] = ai_data.get("sector") or enrichment.get("sector")
                enrichment["industry"] = ai_data.get("industry") or enrichment.get("industry")
                enrichment["country_exposure"] = ai_data.get("country_exposure") or enrichment.get("country_exposure")
                enrichment["currency_exposure"] = ai_data.get("currency_exposure") or enrichment.get("currency_exposure")
                enrichment["is_hedged"] = ai_data.get("is_hedged") if ai_data.get("is_hedged") is not None else enrichment.get("is_hedged")
                enrichment["long_business_summary"] = ai_data.get("long_business_summary") or enrichment.get("long_business_summary")
                
                # Update full metadata with AI contribution
                current_meta = json.loads(enrichment["full_metadata"])
                current_meta["ai_enriched"] = True
                current_meta["ai_data"] = ai_data
                enrichment["full_metadata"] = json.dumps(current_meta)
                
                logger.info(f"AI enrichment successful for {symbol}")

        return enrichment

    def save_enrichment(self, data: Dict[str, Any]):
        """Saves enrichment data to the enrichment database."""
        conn = sqlite3.connect(self.enrichment_db)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO asset_enrichment (
                symbol, name, isin, sector, industry, 
                country_exposure, currency_exposure, is_hedged, 
                long_business_summary, full_metadata, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(symbol) DO UPDATE SET
                name=excluded.name,
                isin=excluded.isin,
                sector=excluded.sector,
                industry=excluded.industry,
                country_exposure=excluded.country_exposure,
                currency_exposure=excluded.currency_exposure,
                is_hedged=excluded.is_hedged,
                long_business_summary=excluded.long_business_summary,
                full_metadata=excluded.full_metadata,
                updated_at=CURRENT_TIMESTAMP
        """, (
            data["symbol"], data.get("name"), data.get("isin"), data.get("sector"), data.get("industry"),
            data.get("country_exposure"), data.get("currency_exposure"), data.get("is_hedged"),
            data.get("long_business_summary"), data.get("full_metadata")
        ))
        
        conn.commit()
        conn.close()

    def run(self):
        """Main execution loop."""
        symbols = self.get_symbols()
        logger.info(f"Found {len(symbols)} unique symbols to enrich.")
        
        for item in symbols:
            symbol = item["symbol"]
            name = item.get("name", "")
            isin = item.get("isin", "")
            
            enrichment_data = self.enrich_symbol(symbol, name, isin)
            if enrichment_data:
                self.save_enrichment(enrichment_data)
                logger.info(f"Successfully enriched {symbol}")
            else:
                logger.warning(f"Failed to enrich {symbol}")
