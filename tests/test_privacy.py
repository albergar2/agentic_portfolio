from src.core.privacy import PrivacyManager

def test_redaction():
    examples = [
        ("My Apple £5000", "My Apple [REDACTED VALUE]"),
        ("Savings $1,200.50", "Savings [REDACTED VALUE]"),
        ("Position with 500 shares", "Position with [REDACTED QTY]"),
        ("Qty: 1200 units of MSFT", "Qty: [REDACTED QTY] of MSFT"),
        ("100 EUR in wallet", "[REDACTED VALUE] in wallet"),
        ("Account balance: 1.5M GBP", "Account balance: [REDACTED VALUE]"),
        ("Regular name", "Regular name"),
        ("Symbol MSFT", "Symbol MSFT"),
    ]
    
    for input_text, expected_output in examples:
        result = PrivacyManager.redact_text(input_text)
        assert result == expected_output, f"Failed for '{input_text}': expected '{expected_output}', got '{result}'"
        print(f"✓ '{input_text}' -> '{result}'")

if __name__ == "__main__":
    try:
        test_redaction()
        print("\nAll privacy tests passed successfully!")
    except AssertionError as e:
        print(f"\nTest failed: {e}")
        exit(1)
