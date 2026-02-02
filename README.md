# Wealthfolio Portfolio Strategy System

A sophisticated AI-driven portfolio management and strategy recommendation system designed for high-net-worth individuals and institutional investors.

## 🏗️ Architecture

```
src/
├── core/           # Core business logic
│   ├── database.py     # Database management
│   └── portfolio.py    # Portfolio analysis
├── ai/             # AI and machine learning components
│   ├── market_oracle.py    # Market research and analysis
│   └── strategist.py       # Strategy generation
├── services/       # Service layer components
│   ├── portfolio_masker.py     # Privacy-preserving data masking
│   └── report_orchestrator.py  # Report generation orchestration
└── main.py         # Main orchestrator and CLI interface
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- SQLite database
- Gemini API key for AI features

### Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Configure your environment:
   ```bash
   cp .env.example .env
   # Edit .env with your Gemini API key
   ```

### Usage

```bash
# Generate a complete strategy report
python src/main.py report --output reports/strategy.md

# Analyze portfolio health
python src/main.py portfolio

# Get market research
python src/main.py market

# Run comprehensive analysis
python src/main.py comprehensive --output-dir reports/
```

## 📊 Features

### Privacy-First Design
- **Data Masking**: Sensitive financial information is anonymized before AI processing
- **Local Processing**: Critical calculations happen locally, not in the cloud
- **Selective Disclosure**: Only necessary data is shared with external AI services

### AI-Driven Insights
- **Market Research**: Real-time analysis of institutional outlooks and macro trends
- **Strategy Generation**: Hybrid approach combining local analysis with AI recommendations
- **Risk Assessment**: Comprehensive portfolio health analysis with concentration monitoring

### Professional Reporting
- **Institutional-Grade**: Reports formatted for professional wealth management standards
- **Comprehensive Analysis**: Multi-layered insights covering macro context, portfolio health, and actionable recommendations
- **Forward-Looking**: Trigger-based strategies for dynamic market conditions

## 🔧 Components

### Core Layer
- **Database Manager**: Robust SQLite database operations with connection pooling
- **Portfolio Analyzer**: Advanced portfolio health assessment and risk analysis

### AI Layer
- **Market Oracle**: Real-time market research and institutional sentiment analysis
- **Strategist**: Hybrid strategy generation with privacy-preserving data handling

### Service Layer
- **Portfolio Masker**: Privacy-first data anonymization for AI processing
- **Report Orchestrator**: Comprehensive report generation with professional formatting

## 🛡️ Privacy & Security

The system implements a multi-layered privacy approach:

1. **Data Minimization**: Only essential portfolio data is processed
2. **Anonymization**: Sensitive identifiers are removed before AI analysis
3. **Local Processing**: Critical calculations remain on-premises
4. **Selective Sharing**: Minimal data exposure to external services

## 📈 Output Examples

The system generates comprehensive reports including:

- **Macro & Market Context**: Current market conditions and institutional outlook
- **Portfolio Health Audit**: Allocation variance, concentration analysis, and geographic critique
- **Forward-Looking Triggers**: Market condition-based action triggers
- **Action Plan**: Specific trade recommendations with strategic rationale

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Create an issue in this repository
- Check the documentation in the `docs/` directory
- Review the comprehensive logging in `logs/` directory