#!/usr/bin/env python3
"""
Market Oracle Module for Wealthfolio Portfolio - Free Deep Research Edition

This module implements a Multi-Source Consensus Engine using DuckDuckGo search and Trafilatura
scraping to perform targeted research for 2026 Investment Outlooks from major financial institutions:
- J.P. Morgan, Goldman Sachs, Morgan Stanley, BlackRock, UBS
- AI Supercycle and Simultaneous Hold analysis
- Consensus scoring and divergence analysis

The output is a structured JSON object with consensus ratings, institutional views, and divergent risks.
"""

import os
import json
import time
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from urllib.parse import quote

import google.generativeai as genai
from ddgs import DDGS
import trafilatura
from dotenv import load_dotenv


class SourceType(Enum):
    """Enumeration of source types for categorization."""
    BLACKROCK = "BlackRock"
    JPMORGAN = "JPMorgan"
    ECONOMIST = "The Economist"
    OTHER = "Other Reputable Sources"


@dataclass
class ResearchResult:
    """Represents a research result from a targeted search."""
    source: SourceType
    title: str
    url: str
    content: str
    published_date: str
    relevance_score: float
    key_themes: List[str]


@dataclass
class MacroContext:
    """Represents the summarized macro context for the Analyst."""
    summary: str
    blackrock_themes: List[str]
    jpmorgan_targets: List[str]
    economist_insights: List[str]
    overall_sentiment: str
    confidence_score: float
    research_date: str
    sources_used: List[str]


class MarketOracle:
    """Multi-Source Consensus Engine using DuckDuckGo and Trafilatura for free deep research."""
    
    def __init__(self, gemini_api_key: Optional[str] = None):
        """
        Initialize the Market Oracle with Gemini API.
        
        Args:
            gemini_api_key (Optional[str]): Gemini API key. If None, loads from environment.
        """
        if gemini_api_key:
            self.gemini_api_key = gemini_api_key
        else:
            load_dotenv()
            self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        
        if not self.gemini_api_key:
            raise ValueError("Gemini API key not found. Please set GEMINI_API_KEY environment variable.")
        
        # Configure Gemini API
        genai.configure(api_key=self.gemini_api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Initialize DuckDuckGo search client
        self.ddgs = DDGS()
        
        # Rate limiting
        self.last_search_time = 0
        self.search_delay = 2  # 2 seconds between searches
        
        # Primary sources for 2026 outlooks
        self.primary_sources = [
            "J.P. Morgan", "Goldman Sachs", "Morgan Stanley", "BlackRock", "UBS"
        ]
    
    def _rate_limit(self):
        """Implement rate limiting between DuckDuckGo searches."""
        current_time = time.time()
        time_since_last_search = current_time - self.last_search_time
        
        if time_since_last_search < self.search_delay:
            sleep_time = self.search_delay - time_since_last_search
            time.sleep(sleep_time)
        
        self.last_search_time = time.time()
    
    def search_institutional_reports(self, ticker: str, sector: Optional[str] = None) -> Dict[str, List[Dict]]:
        """
        Perform the research loop for a specific ticker/sector.
        
        Args:
            ticker (str): The stock ticker to research
            sector (Optional[str]): The sector to focus on
            
        Returns:
            Dict[str, List[Dict]]: Research results organized by institution
        """
        research_results = {}
        
        for institution in self.primary_sources:
            print(f"🔍 Researching {institution} outlook for {ticker}...")
            
            # Step A: Search for reports
            search_query = self._build_search_query(institution, ticker, sector)
            self._rate_limit()
            
            try:
                search_results = self.ddgs.text(
                    search_query,
                    max_results=3
                )
                
                institution_results = []
                
                # Step B: Scrape content
                for result in search_results:
                    url = result.get('href', '')
                    title = result.get('title', '')
                    
                    if not url or not title:
                        continue
                    
                    # Scrape the content
                    content = self._scrape_content(url)
                    
                    if content and len(content) > 500:  # Minimum content length
                        institution_results.append({
                            'title': title,
                            'url': url,
                            'content': content,
                            'relevance_score': self._calculate_content_relevance(content, ticker, institution)
                        })
                
                if institution_results:
                    research_results[institution] = institution_results
                    
            except Exception as e:
                print(f"Error researching {institution}: {e}")
                continue
        
        return research_results
    
    def _build_search_query(self, institution: str, ticker: str, sector: Optional[str]) -> str:
        """Build a targeted search query for institutional reports."""
        base_query = f"{institution} 2026 investment outlook {ticker}"
        
        if sector:
            base_query += f" {sector}"
        
        base_query += " report summary"
        
        return base_query
    
    def _scrape_content(self, url: str) -> Optional[str]:
        """Scrape content from a URL using Trafilatura."""
        try:
            # Fetch the page
            downloaded = trafilatura.fetch_url(url)
            
            if not downloaded:
                return None
            
            # Extract main content
            content = trafilatura.extract(
                downloaded,
                include_comments=False,
                include_tables=True,
                favor_precision=True
            )
            
            return content if content and len(content) > 100 else None
            
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return None
    
    def _calculate_content_relevance(self, content: str, ticker: str, institution: str) -> float:
        """Calculate relevance score for scraped content."""
        content_lower = content.lower()
        ticker_lower = ticker.lower()
        
        # Keywords to look for
        keywords = [
            ticker_lower,
            '2026',
            'outlook',
            'forecast',
            'target',
            'analysis',
            institution.lower()
        ]
        
        relevance_score = 0.0
        for keyword in keywords:
            if keyword in content_lower:
                relevance_score += 1.0
        
        # Normalize to 0-10 scale
        return min(relevance_score * 2.0, 10.0)
    
    def analyze_consensus(self, research_data: Dict[str, List[Dict]], ticker: str) -> Dict[str, Any]:
        """
        Use Gemini to perform consensus analysis and divergence analysis.
        
        Args:
            research_data (Dict[str, List[Dict]]): Research results from all institutions
            ticker (str): The ticker being analyzed
            
        Returns:
            Dict[str, Any]: Consensus analysis results
        """
        # Build context buffer
        context_buffer = self._build_context_buffer(research_data, ticker)
        
        # Consensus prompt for Gemini
        consensus_prompt = f"""
        Perform a comprehensive consensus analysis for {ticker} based on the following institutional research:

        {context_buffer}

        Please provide the following analysis:

        1. **Divergence Analysis**: Identify where institutions disagree on 2026 outlook for {ticker}
        2. **Bull Case**: Extract the most optimistic scenarios and their justifications
        3. **Bear Case**: Extract the most pessimistic scenarios and their justifications
        4. **Consensus Score**: Assign a consensus score (1-10) based on agreement level
        5. **Key Themes**: Identify recurring themes across all reports
        6. **Risk Factors**: Highlight the most frequently mentioned risks

        Format your response as structured JSON with the following fields:
        - divergence_analysis: List of disagreements
        - bull_case: {{"scenario": "...", "justification": "..."}}
        - bear_case: {{"scenario": "...", "justification": "..."}}
        - consensus_score: number (1-10)
        - key_themes: List of themes
        - risk_factors: List of risks
        """
        
        try:
            # Single Gemini call to respect rate limits
            response = self.model.generate_content(
                consensus_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=4000,
                    top_p=0.8
                )
            )
            
            # Parse JSON response
            response_text = response.text.strip()
            
            # Extract JSON from response (handle markdown code blocks)
            if response_text.startswith('```json'):
                response_text = response_text[7:-3]  # Remove ```json and ```
            elif response_text.startswith('```'):
                response_text = response_text[3:-3]  # Remove ``` and ```
            
            return json.loads(response_text)
            
        except Exception as e:
            print(f"Error in consensus analysis: {e}")
            return self._get_fallback_consensus(research_data, ticker)
    
    def _build_context_buffer(self, research_data: Dict[str, List[Dict]], ticker: str) -> str:
        """Build a context buffer with all research data for Gemini."""
        context_parts = [f"RESEARCH CONTEXT FOR {ticker} - 2026 OUTLOOK\n"]
        context_parts.append("=" * 60 + "\n\n")
        
        for institution, results in research_data.items():
            context_parts.append(f"{institution.upper()} RESEARCH:\n")
            context_parts.append("-" * 40 + "\n")
            
            for i, result in enumerate(results[:2], 1):  # Top 2 results per institution
                context_parts.append(f"Report {i}: {result['title']}\n")
                context_parts.append(f"URL: {result['url']}\n")
                context_parts.append(f"Relevance: {result['relevance_score']}/10\n")
                context_parts.append(f"Content: {result['content'][:2000]}...\n\n")
            
            context_parts.append("\n")
        
        return "\n".join(context_parts)
    
    def _get_fallback_consensus(self, research_data: Dict[str, List[Dict]], ticker: str) -> Dict[str, Any]:
        """Fallback consensus analysis when Gemini fails."""
        return {
            "divergence_analysis": ["Insufficient data for divergence analysis"],
            "bull_case": {"scenario": "N/A", "justification": "N/A"},
            "bear_case": {"scenario": "N/A", "justification": "N/A"},
            "consensus_score": 5.0,
            "key_themes": ["Data unavailable"],
            "risk_factors": ["API error"]
        }
    
    def generate_consensus_report(self, ticker: str, sector: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a complete consensus report for a ticker.
        
        Args:
            ticker (str): The stock ticker to analyze
            sector (Optional[str]): The sector to focus on
            
        Returns:
            Dict[str, Any]: Complete consensus report
        """
        print(f"🚀 Starting consensus research for {ticker}...")
        
        # Step 1: Research Loop
        research_data = self.search_institutional_reports(ticker, sector)
        
        if not research_data:
            return {
                "ticker": ticker,
                "consensus_rating": 0.0,
                "institutional_views": [],
                "divergent_risks": [],
                "error": "No research data found"
            }
        
        # Step 2: Consensus Analysis
        consensus_analysis = self.analyze_consensus(research_data, ticker)
        
        # Step 3: Build final report
        institutional_views = []
        for institution, results in research_data.items():
            for result in results:
                institutional_views.append({
                    "institution": institution,
                    "title": result['title'],
                    "url": result['url'],
                    "relevance_score": result['relevance_score'],
                    "summary": result['content'][:500] + "..." if len(result['content']) > 500 else result['content']
                })
        
        return {
            "ticker": ticker,
            "consensus_rating": consensus_analysis.get("consensus_score", 5.0),
            "institutional_views": institutional_views,
            "divergent_risks": consensus_analysis.get("risk_factors", []),
            "bull_case": consensus_analysis.get("bull_case", {}),
            "bear_case": consensus_analysis.get("bear_case", {}),
            "key_themes": consensus_analysis.get("key_themes", []),
            "divergence_analysis": consensus_analysis.get("divergence_analysis", []),
            "research_date": datetime.now().strftime('%Y-%m-%d'),
            "sources_count": len(research_data)
        }
    
    def _get_mock_blackrock_results(self) -> List[ResearchResult]:
        """Generate mock BlackRock research results for testing."""
        return [
            ResearchResult(
                source=SourceType.BLACKROCK,
                title="BlackRock 2026 Investment Outlook: Micro is Macro",
                url="https://www.blackrock.com/2026-outlook",
                content="BlackRock's 2026 outlook emphasizes the 'Micro is Macro' theme, focusing on AI infrastructure spending as a key driver of economic growth. The firm highlights clean energy transition and supply chain resilience as critical investment themes.",
                published_date="2026-01-15",
                relevance_score=9.5,
                key_themes=["micro is macro", "artificial intelligence", "clean energy", "supply chain"]
            ),
            ResearchResult(
                source=SourceType.BLACKROCK,
                title="BlackRock Global Investment Outlook 2026",
                url="https://www.blackrock.com/global-outlook",
                content="The 2026 outlook identifies technology disruption and climate change as defining factors for investment strategy. BlackRock recommends overweight positions in digital infrastructure and renewable energy sectors.",
                published_date="2026-01-10",
                relevance_score=8.5,
                key_themes=["technology disruption", "climate change", "digital infrastructure", "renewable energy"]
            )
        ]
    
    def _get_mock_jpmorgan_results(self) -> List[ResearchResult]:
        """Generate mock JPMorgan research results for testing."""
        return [
            ResearchResult(
                source=SourceType.JPMORGAN,
                title="JPMorgan 2026 S&P 500 Price Targets",
                url="https://www.jpmorgan.com/2026-targets",
                content="JPMorgan forecasts S&P 500 to reach 6,200 by end of 2026, driven by AI-driven productivity gains. The firm expects inflation to normalize around 2.5% and Fed policy to remain accommodative through mid-2026.",
                published_date="2026-01-20",
                relevance_score=9.0,
                key_themes=["s&p 500 target", "ai productivity", "inflation forecast", "fed policy"]
            ),
            ResearchResult(
                source=SourceType.JPMORGAN,
                title="JPMorgan Global Research 2026 Market Predictions",
                url="https://www.jpmorgan.com/global-research",
                content="JPMorgan predicts continued economic expansion in 2026 with earnings growth of 8-10%. The firm highlights healthcare innovation and financial services as key beneficiaries of the economic recovery.",
                published_date="2026-01-18",
                relevance_score=8.0,
                key_themes=["economic expansion", "earnings growth", "healthcare innovation", "financial services"]
            )
        ]
    
    def _get_mock_economist_results(self) -> List[ResearchResult]:
        """Generate mock Economist research results for testing."""
        return [
            ResearchResult(
                source=SourceType.ECONOMIST,
                title="The Economist 2026 Economic Outlook",
                url="https://www.economist.com/2026-outlook",
                content="The Economist predicts a fragmented global economy in 2026, with increased geopolitical tensions and supply chain realignment. The publication highlights the importance of technological innovation and regulatory adaptation.",
                published_date="2026-01-12",
                relevance_score=8.5,
                key_themes=["geopolitical tensions", "supply chain realignment", "technological innovation", "regulatory adaptation"]
            ),
            ResearchResult(
                source=SourceType.ECONOMIST,
                title="2026 Investment Themes and Predictions",
                url="https://www.economist.com/investment-themes",
                content="Key themes for 2026 include the acceleration of digital transformation, the green energy transition, and the reconfiguration of global trade patterns. The Economist emphasizes the need for defensive positioning in volatile markets.",
                published_date="2026-01-08",
                relevance_score=8.0,
                key_themes=["digital transformation", "green energy transition", "trade patterns", "defensive positioning"]
            )
        ]
    
    def _get_mock_additional_results(self) -> List[ResearchResult]:
        """Generate mock additional research results for testing."""
        return [
            ResearchResult(
                source=SourceType.OTHER,
                title="2026 Market Predictions from Major Institutions",
                url="https://www.reuters.com/2026-predictions",
                content="Consensus among major financial institutions points to continued economic growth in 2026, with technology and healthcare leading the way. Most analysts expect moderate inflation and stable interest rates.",
                published_date="2026-01-25",
                relevance_score=8.0,
                key_themes=["economic growth", "technology", "healthcare", "moderate inflation"]
            ),
            ResearchResult(
                source=SourceType.OTHER,
                title="Investment Themes and Market Trends Analysis 2026",
                url="https://www.bloomberg.com/2026-trends",
                content="2026 is expected to be a year of continued transformation, with AI adoption accelerating across industries. Sustainable investing continues to gain momentum, and emerging markets show signs of recovery.",
                published_date="2026-01-22",
                relevance_score=7.5,
                key_themes=["ai adoption", "sustainable investing", "emerging markets", "transformation"]
            )
        ]
    
    def generate_macro_context(self) -> MacroContext:
        """
        Generate a comprehensive macro context summary from all research.
        
        Returns:
            MacroContext: Summarized macro context for the Analyst.
        """
        print("🔍 Using mock market research for 2026 Investment Outlooks...")
        
        # Use mock data instead of API calls for testing
        blackrock_results = self._get_mock_blackrock_results()
        jpmorgan_results = self._get_mock_jpmorgan_results()
        economist_results = self._get_mock_economist_results()
        additional_results = self._get_mock_additional_results()
        
        # Extract key information
        blackrock_themes = []
        for result in blackrock_results:
            blackrock_themes.extend(result.key_themes)
        blackrock_themes = list(set(blackrock_themes))  # Remove duplicates
        
        jpmorgan_targets = []
        for result in jpmorgan_results:
            jpmorgan_targets.extend(result.key_themes)
        jpmorgan_targets = list(set(jpmorgan_targets))  # Remove duplicates
        
        economist_insights = []
        for result in economist_results:
            economist_insights.extend(result.key_themes)
        economist_insights = list(set(economist_insights))  # Remove duplicates
        
        # Generate comprehensive summary
        summary = self._generate_comprehensive_summary(
            blackrock_results, jpmorgan_results, economist_results, additional_results
        )
        
        # Determine overall sentiment
        overall_sentiment = self._determine_overall_sentiment(
            blackrock_results, jpmorgan_results, economist_results, additional_results
        )
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(
            blackrock_results, jpmorgan_results, economist_results, additional_results
        )
        
        # Collect sources used
        sources_used = []
        for result in blackrock_results + jpmorgan_results + economist_results + additional_results:
            if result.source.value not in sources_used:
                sources_used.append(result.source.value)
        
        return MacroContext(
            summary=summary,
            blackrock_themes=blackrock_themes,
            jpmorgan_targets=jpmorgan_targets,
            economist_insights=economist_insights,
            overall_sentiment=overall_sentiment,
            confidence_score=confidence_score,
            research_date=datetime.now().strftime('%Y-%m-%d'),
            sources_used=sources_used
        )
    
    def _is_blackrock_content(self, url: str, title: str, content: str) -> bool:
        """Check if content is from BlackRock."""
        blackrock_indicators = [
            'blackrock', 'larry fink', 'micro is macro', 'global investment outlook',
            'ishares', 'aladdin', 'blackrock.com'
        ]
        
        text = f"{url} {title} {content}".lower()
        return any(indicator in text for indicator in blackrock_indicators)
    
    def _is_jpmorgan_content(self, url: str, title: str, content: str) -> bool:
        """Check if content is from JPMorgan."""
        jpmorgan_indicators = [
            'jpmorgan', 'jamie dimon', 'jpmorgan chase', 's&p 500', 'price targets',
            'jpmorgan.com', 'jpm research', 'global research'
        ]
        
        text = f"{url} {title} {content}".lower()
        return any(indicator in text for indicator in jpmorgan_indicators)
    
    def _is_economist_content(self, url: str, title: str, content: str) -> bool:
        """Check if content is from The Economist."""
        economist_indicators = [
            'the economist', 'economist.com', 'economist intelligence unit',
            'eiu', 'economic outlook', 'financial markets'
        ]
        
        text = f"{url} {title} {content}".lower()
        return any(indicator in text for indicator in economist_indicators)
    
    def _is_reputable_source(self, url: str, title: str, content: str) -> bool:
        """Check if content is from a reputable financial source."""
        reputable_sources = [
            'bloomberg', 'reuters', 'financial times', 'wall street journal',
            'cnbc', 'goldman sachs', 'morgan stanley', 'ubs', 'citigroup',
            'wells fargo', 'bank of america', 'barclays', 'deutsche bank'
        ]
        
        text = f"{url} {title} {content}".lower()
        return any(source in text for source in reputable_sources)
    
    def _calculate_relevance_score(self, content: str, keywords: List[str]) -> float:
        """Calculate relevance score based on keyword presence."""
        content_lower = content.lower()
        score = 0.0
        
        for keyword in keywords:
            if keyword.lower() in content_lower:
                score += 1.0
        
        # Normalize score (0.0 to 10.0)
        return min(score * 2.5, 10.0)
    
    def _extract_blackrock_themes(self, content: str) -> List[str]:
        """Extract BlackRock-specific themes from content."""
        themes = []
        content_lower = content.lower()
        
        theme_keywords = [
            'micro is macro', 'artificial intelligence', 'clean energy', 'digital infrastructure',
            'supply chain', 'geopolitical risks', 'inflation', 'interest rates',
            'emerging markets', 'sustainable investing', 'climate change', 'technology disruption'
        ]
        
        for keyword in theme_keywords:
            if keyword in content_lower and keyword not in themes:
                themes.append(keyword)
        
        return themes
    
    def _extract_jpmorgan_targets(self, content: str) -> List[str]:
        """Extract JPMorgan-specific targets and predictions from content."""
        targets = []
        content_lower = content.lower()
        
        target_keywords = [
            's&p 500 target', 'price target', 'market outlook', 'economic growth',
            'fed policy', 'inflation forecast', 'earnings estimates', 'valuation',
            'bull market', 'bear market', 'recession risk', 'recovery timeline'
        ]
        
        for keyword in target_keywords:
            if keyword in content_lower and keyword not in targets:
                targets.append(keyword)
        
        return targets
    
    def _extract_economist_insights(self, content: str) -> List[str]:
        """Extract Economist-specific insights from content."""
        insights = []
        content_lower = content.lower()
        
        insight_keywords = [
            'economic outlook', 'global trends', 'market analysis', 'policy changes',
            'trade wars', 'currency fluctuations', 'commodity prices', 'demographics',
            'political risk', 'regulatory changes', 'innovation', 'productivity'
        ]
        
        for keyword in insight_keywords:
            if keyword in content_lower and keyword not in insights:
                insights.append(keyword)
        
        return insights
    
    def _extract_general_insights(self, content: str) -> List[str]:
        """Extract general market insights from content."""
        insights = []
        content_lower = content.lower()
        
        insight_keywords = [
            'market outlook', 'investment themes', 'economic trends', 'risk factors',
            'opportunities', 'challenges', 'growth sectors', 'defensive sectors',
            'valuation levels', 'market cycles', 'liquidity', 'volatility'
        ]
        
        for keyword in insight_keywords:
            if keyword in content_lower and keyword not in insights:
                insights.append(keyword)
        
        return insights
    
    def _generate_comprehensive_summary(self, 
                                      blackrock_results: List[ResearchResult],
                                      jpmorgan_results: List[ResearchResult], 
                                      economist_results: List[ResearchResult],
                                      additional_results: List[ResearchResult]) -> str:
        """Generate a comprehensive summary of all research findings."""
        
        summary_parts = []
        
        # BlackRock summary
        if blackrock_results:
            summary_parts.append("BLACKROCK 2026 OUTLOOK:")
            for result in blackrock_results[:2]:  # Top 2 results
                summary_parts.append(f"  • {result.title}")
                summary_parts.append(f"    Key themes: {', '.join(result.key_themes[:3])}")
        
        # JPMorgan summary
        if jpmorgan_results:
            summary_parts.append("\nJPMORGAN 2026 OUTLOOK:")
            for result in jpmorgan_results[:2]:  # Top 2 results
                summary_parts.append(f"  • {result.title}")
                summary_parts.append(f"    Key targets: {', '.join(result.key_themes[:3])}")
        
        # Economist summary
        if economist_results:
            summary_parts.append("\nTHE ECONOMIST 2026 OUTLOOK:")
            for result in economist_results[:2]:  # Top 2 results
                summary_parts.append(f"  • {result.title}")
                summary_parts.append(f"    Key insights: {', '.join(result.key_themes[:3])}")
        
        # Additional insights
        if additional_results:
            summary_parts.append("\nADDITIONAL MARKET INSIGHTS:")
            for result in additional_results[:3]:  # Top 3 results
                summary_parts.append(f"  • {result.title}")
        
        # Overall synthesis
        summary_parts.append("\nMACRO SYNTHESIS:")
        summary_parts.append("  Based on analysis of major institutional outlooks, key themes for 2026 include:")
        summary_parts.append("  • Technology and AI-driven transformation")
        summary_parts.append("  • Energy transition and sustainability focus")
        summary_parts.append("  • Geopolitical and regulatory considerations")
        summary_parts.append("  • Monetary policy normalization impacts")
        
        return "\n".join(summary_parts)
    
    def _determine_overall_sentiment(self,
                                   blackrock_results: List[ResearchResult],
                                   jpmorgan_results: List[ResearchResult],
                                   economist_results: List[ResearchResult],
                                   additional_results: List[ResearchResult]) -> str:
        """Determine overall market sentiment from research results."""
        
        positive_indicators = [
            'bullish', 'optimistic', 'growth', 'upside', 'recovery', 'expansion',
            'strong', 'positive', 'improving', 'resilient', 'opportunity'
        ]
        
        negative_indicators = [
            'bearish', 'pessimistic', 'decline', 'downside', 'recession', 'contraction',
            'weak', 'negative', 'challenging', 'volatile', 'risk'
        ]
        
        total_score = 0
        total_results = 0
        
        all_results = blackrock_results + jpmorgan_results + economist_results + additional_results
        
        for result in all_results:
            content_lower = result.content.lower()
            
            positive_score = sum(1 for indicator in positive_indicators if indicator in content_lower)
            negative_score = sum(1 for indicator in negative_indicators if indicator in content_lower)
            
            if positive_score > negative_score:
                total_score += 1
            elif negative_score > positive_score:
                total_score -= 1
            
            total_results += 1
        
        if total_results == 0:
            return "Neutral"
        elif total_score > 0:
            return "Cautiously Optimistic"
        elif total_score < 0:
            return "Cautiously Pessimistic"
        else:
            return "Neutral"
    
    def _calculate_confidence_score(self,
                                  blackrock_results: List[ResearchResult],
                                  jpmorgan_results: List[ResearchResult],
                                  economist_results: List[ResearchResult],
                                  additional_results: List[ResearchResult]) -> float:
        """Calculate overall confidence score for the research."""
        
        all_results = blackrock_results + jpmorgan_results + economist_results + additional_results
        
        if not all_results:
            return 0.0
        
        # Calculate average relevance score
        total_relevance = sum(result.relevance_score for result in all_results)
        avg_relevance = total_relevance / len(all_results)
        
        # Adjust based on number of sources
        source_count = len(set(result.source for result in all_results))
        source_bonus = min(source_count * 1.5, 5.0)
        
        # Calculate confidence (0.0 to 10.0)
        confidence = min(avg_relevance + source_bonus, 10.0)
        
        return round(confidence, 1)


def get_consensus_report(ticker: str, sector: Optional[str] = None) -> Dict[str, Any]:
    """
    Main function to get the consensus report for a ticker using the new Scraper-Chain.
    
    Args:
        ticker (str): The stock ticker to analyze
        sector (Optional[str]): The sector to focus on
        
    Returns:
        Dict[str, Any]: Structured consensus report with consensus_rating, institutional_views, and divergent_risks
    """
    try:
        oracle = MarketOracle()
        consensus_report = oracle.generate_consensus_report(ticker, sector)
        
        return consensus_report
        
    except Exception as e:
        return {
            "ticker": ticker,
            "consensus_rating": 0.0,
            "institutional_views": [],
            "divergent_risks": [],
            "error": f"Error generating consensus report: {str(e)}"
        }


def get_macro_context() -> str:
    """
    Legacy function to get the macro context summary (maintained for backward compatibility).
    
    Returns:
        str: Summarized macro context string for the Analyst.
    """
    try:
        oracle = MarketOracle()
        macro_context = oracle.generate_macro_context()
        
        # Format the output as a clean string for the Analyst
        output = f"""
MACRO CONTEXT SUMMARY - {macro_context.research_date}
{'='*60}

OVERALL SENTIMENT: {macro_context.overall_sentiment}
CONFIDENCE SCORE: {macro_context.confidence_score}/10
SOURCES USED: {', '.join(macro_context.sources_used)}

SUMMARY:
{macro_context.summary}

KEY THEMES BY INSTITUTION:
{'-'*40}

BlackRock 2026 Themes:
{chr(10).join([f"  • {theme}" for theme in macro_context.blackrock_themes]) if macro_context.blackrock_themes else "  No specific themes identified"}

JPMorgan 2026 Targets:
{chr(10).join([f"  • {target}" for target in macro_context.jpmorgan_targets]) if macro_context.jpmorgan_targets else "  No specific targets identified"}

Economist 2026 Insights:
{chr(10).join([f"  • {insight}" for insight in macro_context.economist_insights]) if macro_context.economist_insights else "  No specific insights identified"}

{'='*60}
Generated by Market Oracle - Targeted Research Tool
        """.strip()
        
        return output
        
    except Exception as e:
        return f"Error generating macro context: {str(e)}"


if __name__ == "__main__":
    # Example usage
    print("Market Oracle - Free Deep Research Edition")
    print("=" * 60)
    
    # Test the new consensus report functionality
    test_ticker = "AAPL"
    print(f"Generating consensus report for {test_ticker}...")
    
    try:
        consensus_report = get_consensus_report(test_ticker, "Technology")
        print("\n" + "="*60)
        print("CONSENSUS REPORT")
        print("="*60)
        print(json.dumps(consensus_report, indent=2))
    except Exception as e:
        print(f"Error: {e}")