import sqlite3
import pandas as pd
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from src.core.config import WEALTHFOLIO_DB, ENRICHMENT_DB

logger = logging.getLogger(__name__)

class AnalysisService:
    """Service to generate detailed portfolio performance and exposure reports."""
    
    def __init__(self, main_db: str = str(WEALTHFOLIO_DB), enrichment_db: str = str(ENRICHMENT_DB)):
        self.main_db = main_db
        self.enrichment_db = enrichment_db

    def _load_json(self, x):
        try:
            return json.loads(x) if x and x != '{}' and x is not None else {}
        except:
            return {}

    def _get_closest_price(self, df_quotes, symbol, target_dt, fx_rates):
        """Finds price for symbol at target_dt, converted to GBP."""
        symbol_quotes = df_quotes[df_quotes['symbol'] == symbol]
        if symbol_quotes.empty:
            return None
        
        past_quotes = symbol_quotes[symbol_quotes['dt'] <= target_dt]
        if past_quotes.empty:
            return None
        
        latest = past_quotes.sort_values('dt').iloc[-1]
        price = latest['close']
        currency = latest['currency']
        
        # 1. Normalize Pence to Pounds
        if currency == 'GBp':
            price = price / 100.0
            currency = 'GBP'
        
        # 2. FX Conversion to GBP
        if currency != 'GBP':
            rate = fx_rates.get(f"{currency}GBP=X", 1.0)
            if rate == 1.0 and currency == 'USD':
                rate = fx_rates.get('USDGBP=X', 0.82)
            price = price * rate
            
        return price

    def generate_report(self) -> str:
        """Generates a comprehensive text-based analysis report."""
        if not WEALTHFOLIO_DB.exists():
            return f"Error: Database missing at {self.main_db}"

        conn = sqlite3.connect(self.main_db)
        
        # 1. Load Data
        df_assets = pd.read_sql_query("SELECT symbol, name, currency FROM assets", conn)
        
        latest_date = pd.read_sql_query("SELECT MAX(snapshot_date) FROM holdings_snapshots WHERE positions != '{}'", conn).iloc[0, 0]
        if not latest_date: 
            conn.close()
            return "No data found in database."

        # Use 'TOTAL' aggregate if available, or individual accounts.
        df_snaps = pd.read_sql_query("SELECT * FROM holdings_snapshots WHERE snapshot_date = ?", conn, params=(latest_date,))
        if 'TOTAL' in df_snaps['account_id'].values:
            df_snaps = df_snaps[df_snaps['account_id'] == 'TOTAL']
        else:
            df_snaps = df_snaps[df_snaps['account_id'] != 'TOTAL']

        # 3. Load Quotes & FX
        df_q = pd.read_sql_query("SELECT symbol, timestamp, close, currency FROM quotes", conn)
        df_q['dt'] = pd.to_datetime(df_q['timestamp'], utc=True, format='ISO8601')
        df_q['close'] = pd.to_numeric(df_q['close'], errors='coerce')

        # Latest FX rates
        fx_latest = df_q[df_q['symbol'].str.endswith('=X')].sort_values('dt').groupby('symbol').last()['close'].to_dict()
        fx_latest['GBPGBP=X'] = 1.0

        # 2. Process Positions
        positions = []
        for _, row in df_snaps.iterrows():
            pos_dict = self._load_json(row['positions'])
            for sym, data in pos_dict.items():
                cost = float(data.get('totalCostBasis', 0))
                asset_currency = df_assets[df_assets['symbol'] == sym]['currency'].iloc[0] if sym in df_assets['symbol'].values else 'GBP'
                
                if asset_currency == 'GBp':
                    cost = cost / 100.0
                    asset_currency = 'GBP'
                
                if asset_currency != 'GBP':
                    rate = fx_latest.get(f"{asset_currency}GBP=X", 1.0)
                    if rate == 1.0 and asset_currency == 'USD':
                        rate = fx_latest.get('USDGBP=X', 0.82)
                    cost = cost * rate
                
                positions.append({
                    'symbol': sym,
                    'quantity': float(data.get('quantity', 0)),
                    'totalCostBasis': cost
                })
        
        df_p = pd.DataFrame(positions).groupby('symbol').sum().reset_index()
        df_p = df_p.merge(df_assets, on='symbol', how='left')

        current_dt = df_q['dt'].max()
        
        report_lines = []
        report_lines.append("="*80)
        report_lines.append(f"PORTFOLIO PERFORMANCE REPORT (Currency: GBP) | Ref Date: {latest_date}")
        report_lines.append("="*80)

        # 4. Performance Lags
        lags = {'1w': 7, '1m': 30, '3m': 90, '6m': 180, '1y': 365}
        
        df_p['price_curr'] = df_p['symbol'].apply(lambda s: self._get_closest_price(df_q, s, current_dt, fx_latest))
        df_p['current_val'] = df_p['quantity'] * df_p['price_curr']
        total_val = df_p['current_val'].sum()

        for label, days in lags.items():
            bench_dt = current_dt - timedelta(days=days)
            df_p[f'price_{label}'] = df_p['symbol'].apply(lambda s: self._get_closest_price(df_q, s, bench_dt, fx_latest))
            df_p[f'perf_{label}'] = ((df_p['price_curr'] - df_p[f'price_{label}']) / df_p[f'price_{label}'] * 100).round(2)

        df_p['Weight %'] = (df_p['current_val'] / total_val * 100).round(2)
        df_p['Total Gain %'] = ((df_p['current_val'] - df_p['totalCostBasis']) / df_p['totalCostBasis'] * 100).round(2)

        report_lines.append(f"\nTotal Portfolio Value: {total_val:,.2f} GBP")
        
        report_lines.append("\n[1] PERFORMANCE METRICS ATTRIBUTION (%)")
        cols = ['symbol', 'Weight %', 'Total Gain %'] + [f'perf_{l}' for l in lags.keys()]
        display_df = df_p[cols].sort_values('Weight %', ascending=False)
        display_df.columns = ['Symbol', 'Weight %', 'Total Gain %', '1w', '1m', '3m', '6m', '1y']
        report_lines.append(display_df.to_string(index=False))

        # Enrichment Data Join
        if ENRICHMENT_DB.exists():
            conn_e = sqlite3.connect(self.enrichment_db)
            df_e = pd.read_sql_query("SELECT symbol, sector, industry, country_exposure, currency_exposure, is_hedged, full_metadata FROM asset_enrichment", conn_e)
            df_p = df_p.merge(df_e, on='symbol', how='left')
            conn_e.close()

        df_p['sector'] = df_p['sector'].fillna('Other/ETF')
        df_p['industry'] = df_p['industry'].fillna('Other/ETF')
        df_p['is_hedged'] = df_p['is_hedged'].fillna(0).astype(bool)

        def extract_meta(x, key):
            data = self._load_json(x)
            return data.get(key)

        meta_keys = ['morningStarOverallRating', 'morningStarRiskRating', 'annualReportExpenseRatio', 'beta', 'beta3Year', 'trailingPE', 'payoutRatio']
        for k in meta_keys:
            df_p[k] = df_p['full_metadata'].apply(lambda x: extract_meta(x, k))

        report_lines.append("\n[2] SECTOR EXPOSURE (%)")
        sec_dist = (df_p.groupby('sector')['current_val'].sum() / total_val * 100).round(2).sort_values(ascending=False)
        report_lines.append(pd.DataFrame({'Weight %': sec_dist}).to_string())

        report_lines.append("\n[3] INDUSTRY BREAKDOWN (Top 10 %)")
        ind_dist = (df_p.groupby('industry')['current_val'].sum() / total_val * 100).round(2).sort_values(ascending=False).head(10)
        report_lines.append(pd.DataFrame({'Weight %': ind_dist}).to_string())

        report_lines.append("\n[4] GEOGRAPHIC EXPOSURE (%)")
        geo = {}
        for _, row in df_p.iterrows():
            if pd.notna(row['country_exposure']):
                try:
                    exposure = self._load_json(row['country_exposure'])
                    for c, p in exposure.items():
                        v = (float(p)/100.0) * row['current_val']
                        geo[c] = geo.get(c, 0) + v
                except: pass
        if geo:
            geo_df = (pd.Series(geo) / sum(geo.values()) * 100).round(2).sort_values(ascending=False)
            report_lines.append(pd.DataFrame({'Exposure %': geo_df}).head(10).to_string())

        report_lines.append("\n[5] CURRENCY EXPOSURE (%)")
        cur = {}
        for _, row in df_p.iterrows():
            if row['is_hedged']:
                cur['GBP'] = cur.get('GBP', 0) + row['current_val']
                continue
            if pd.notna(row['currency_exposure']) and row['currency_exposure'] != '{}':
                try:
                    exposure = self._load_json(row['currency_exposure'])
                    for c, p in exposure.items():
                        c_norm = 'GBP' if c in ['GBp', 'GBP'] else c
                        v = (float(p)/100.0) * row['current_val']
                        cur[c_norm] = cur.get(c_norm, 0) + v
                except: pass
            else:
                c = row['currency']
                if c in ['GBp', 'GBP']: c = 'GBP'
                cur[c] = cur.get(c, 0) + row['current_val']
        
        if cur:
            cur_df = (pd.Series(cur) / sum(cur.values()) * 100).round(2).sort_values(ascending=False)
            report_lines.append(pd.DataFrame({'Exposure %': cur_df}).to_string())

        report_lines.append("\n[6] HEDGING STATUS (%)")
        hedge_dist = (df_p.groupby('is_hedged')['current_val'].sum() / total_val * 100).round(2)
        hedge_df = pd.DataFrame({'Weight %': hedge_dist})
        hedge_df.index = hedge_df.index.map({True: 'Hedged', False: 'Unhedged'})
        report_lines.append(hedge_df.to_string())

        report_lines.append("\n[7] FUND QUALITY & COSTS (Morningstar)")
        fund_mask = df_p['morningStarOverallRating'].notna() | df_p['annualReportExpenseRatio'].notna()
        if fund_mask.any():
            fund_cols = ['symbol', 'Weight %', 'morningStarOverallRating', 'morningStarRiskRating', 'annualReportExpenseRatio']
            fund_df = df_p[fund_mask][fund_cols].sort_values('Weight %', ascending=False)
            fund_df.columns = ['Symbol', 'Weight %', 'Stars', 'Risk', 'Exp Ratio (%)']
            fund_df['Exp Ratio (%)'] = fund_df['Exp Ratio (%)'].apply(lambda x: round(x * 100, 3) if pd.notna(x) else None)
            report_lines.append(fund_df.to_string(index=False))

        report_lines.append("\n[8] VOLATILITY & RISK (Beta)")
        df_p['beta3Year'] = pd.to_numeric(df_p['beta3Year'], errors='coerce')
        df_p['beta'] = pd.to_numeric(df_p['beta'], errors='coerce')
        df_p['effective_beta'] = df_p['beta3Year'].fillna(df_p['beta'])
        beta_mask = df_p['effective_beta'].notna()
        if beta_mask.any():
            beta_df = df_p[beta_mask][['symbol', 'Weight %', 'effective_beta']].copy()
            beta_df['Weighted Beta'] = (beta_df['Weight %'] / 100 * beta_df['effective_beta']).round(4)
            report_lines.append(beta_df.sort_values('Weight %', ascending=False).to_string(index=False))
            portfolio_beta = beta_df['Weighted Beta'].sum()
            report_lines.append(f"\nEstimated Portfolio Beta: {portfolio_beta:.2f}")

        report_lines.append("\n[9] VALUATION & DIVIDENDS")
        val_cols = ['symbol', 'Weight %', 'trailingPE', 'payoutRatio']
        val_mask = df_p['trailingPE'].notna() | df_p['payoutRatio'].notna()
        if val_mask.any():
            val_df = df_p[val_mask][val_cols].sort_values('Weight %', ascending=False)
            val_df.columns = ['Symbol', 'Weight %', 'P/E', 'Payout %']
            val_df['Payout %'] = val_df['Payout %'].apply(lambda x: round(x * 100, 1) if pd.notna(x) else None)
            report_lines.append(val_df.to_string(index=False))

        total_cost = df_p['totalCostBasis'].sum()
        total_gain = total_val - total_cost
        total_gain_pct = (total_gain / total_cost * 100) if total_cost else 0
        
        report_lines.append("\n" + "="*80)
        report_lines.append(f"PORTFOLIO SUMMARY")
        report_lines.append(f"Total Cost Basis:  {total_cost:,.2f} GBP")
        report_lines.append(f"Current Value:     {total_val:,.2f} GBP")
        report_lines.append(f"Total Gain/Loss:   {total_gain:,.2f} GBP ({total_gain_pct:+.2f}%)")
        report_lines.append("="*80)

        report_lines.append(f"\nReport Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        report_lines.append("="*80)

        conn.close()
        return "\n".join(report_lines)
