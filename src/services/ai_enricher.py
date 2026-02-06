import os
import json
import logging
from typing import Dict, Any, Optional
from google import genai
from google.genai import types
from dotenv import load_dotenv
from src.core.privacy import PrivacyManager

logger = logging.getLogger(__name__)

class AIEnricher:
    """Service to enrich asset metadata using Gemini AI."""
    
    def __init__(self, api_key: Optional[str] = None):
        if not api_key:
            load_dotenv()
            api_key = os.getenv('GEMINI_API_KEY')
        
        if not api_key:
            logger.error("GEMINI_API_KEY not found in environment.")
            self.client = None
        else:
            self.client = genai.Client(api_key=api_key)
            self.model_name = 'gemini-1.5-flash'

    def enrich_asset(self, symbol: str, name: str, isin: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Queries Gemini for deep asset metadata."""
        if not hasattr(self, 'client') or not self.client:
            return None
            
        # Redact any values from inputs
        safe_symbol = PrivacyManager.redact_text(symbol)
        safe_name = PrivacyManager.redact_text(name)
        safe_isin = PrivacyManager.redact_text(isin) if isin else None

        prompt = f"""
        Research the financial asset with symbol '{safe_symbol}' and name '{safe_name}' (ISIN: {safe_isin or 'Unknown'}).
        Provide detailed metadata for portfolio analysis.
        
        Fields required:
        - sector: The broad economic sector (e.g., Technology, Financial Services, Healthcare) or Fund Category (e.g., Equity Europe, Fixed Income).
        - industry: The specific industry or sub-category.
        - country_exposure: A JSON-like dictionary of country percentages (e.g., {{"USA": 60, "UK": 20, "Other": 20}}).
        - currency_exposure: A JSON-like dictionary of currency percentages (e.g., {{"USD": 70, "EUR": 30}}).
        - is_hedged: Boolean (true if the asset is currency-hedged, false otherwise).
        - long_business_summary: A brief description of the asset's objective or business.
        
        Format your response as a valid JSON object only. No markdown formatting, just the raw JSON.
        """
        
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    response_mime_type="application/json"
                )
            )
            
            data = json.loads(response.text)
            
            # Ensure exposures are strings for DB storage if they are dicts
            if isinstance(data.get("country_exposure"), dict):
                data["country_exposure"] = json.dumps(data["country_exposure"])
            if isinstance(data.get("currency_exposure"), dict):
                data["currency_exposure"] = json.dumps(data["currency_exposure"])
                
            return data
            
        except Exception as e:
            logger.error(f"AI enrichment failed for {symbol}: {e}")
            return None
