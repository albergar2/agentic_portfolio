import re
import logging
from typing import Any, Dict, List, Union

logger = logging.getLogger(__name__)

class PrivacyManager:
    """Utility to redact confidential information from strings sent to external APIs."""
    
    # Regex to catch currency patterns like $1,000, 500€, 1.2M GBP, etc.
    CURRENCY_REGEX = re.compile(
        r'(?i)(?:[\$\xA3\xA5\u20AC]\s?\d+(?:[.,]\d+)*(?:\s?[MBK])?)|'  # Symbols start: $100, £500
        r'(?:\d+(?:[.,]\d+)*(?:\s?[MBK])?\s?(?:USD|GBP|EUR|Pounds|Dollars|Euros|Pence|GBp|GBX|JPY))', # Codes end: 100 USD
        re.IGNORECASE
    )
    
    # Regex to catch suspicious numbers that might be share quantities in names
    # E.g. "Apple 500 shares" or just large numbers in specific contexts
    QUANTITY_REGEX = re.compile(
        r'(?i)(?:\d+(?:[.,]\d+)*\s?(?:shares|units|qty|quantity|pos|position))',
        re.IGNORECASE
    )

    @classmethod
    def redact_text(cls, text: str) -> str:
        """Redacts currency and quantity patterns from a string."""
        if not text:
            return text
        
        redacted = cls.CURRENCY_REGEX.sub("[REDACTED VALUE]", text)
        redacted = cls.QUANTITY_REGEX.sub("[REDACTED QTY]", redacted)
        
        return redacted

    @classmethod
    def sanitize_asset_data(cls, data: Union[str, Dict[str, Any], List[Any]]) -> Any:
        """Recursively sanitizes asset data (strings, dicts, lists)."""
        if isinstance(data, str):
            return cls.redact_text(data)
        elif isinstance(data, dict):
            return {k: cls.sanitize_asset_data(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [cls.sanitize_asset_data(item) for item in data]
        return data
