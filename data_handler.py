import pandas as pd
import os
import numpy as np
import getfactormodels as gfm
import yfinance as yf
import sys
import requests
from datetime import datetime

from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler

# --- Helper and Evaluation Functions ---
def import_and_resample_files(file_paths, target_frequency, start_date_filter=None):
    """
    Loads data from CSV files, filters by date, and resamples.
    """
    dataframes = {}
    for file_path in file_paths:
        try:
            df_name = os.path.splitext(os.path.basename(file_path))[0]
            df = pd.read_csv(file_path, index_col=0, parse_dates=True, sep=',', encoding='utf-8', skipinitialspace=True)
            df.columns = df.columns.str.strip()
            if start_date_filter:
                df = df[df.index >= start_date_filter]
            if not df.empty:
                numeric_df = df.select_dtypes(include=np.number)
                dataframes[df_name] = numeric_df.resample(target_frequency).mean()
        except FileNotFoundError:
            print(f"⚠️ Warning: File not found at {file_path}. Skipping.")
        except Exception as e:
            print(f"Could not process {file_path}: {e}")
    return dataframes

### --- NEW IMF API HELPER FUNCTION --- ###
def fetch_imf_data(period, indicator_code, countries, column_suffix):
    """
    Fetches and processes quarterly data from the IMF API for multiple countries.
    """
    print(f"\n--- Fetching IMF Data: {column_suffix} ---")
    try:
        years = int(period.replace('y', ''))
        end_year = datetime.now().year
        start_year = end_year - years

        countries_str = "+".join(countries)

        # Construct the URL for the IMF SDMX JSON API
        url = f"http://dataservices.imf.org/REST/SDMX_JSON.svc/CompactData/IFS/Q.{countries_str}.{indicator_code}?startPeriod={start_year}&endPeriod={end_year}"

        response = requests.get(url)
        response.raise_for_status() # Raises an error for bad status codes

        data = response.json()

        series = data['CompactData']['DataSet']['Series']
        if not series:
            print(f"⚠️ No data returned from IMF API for {column_suffix}.")
            return pd.DataFrame()

        all_records = []
        for s in series:
            country_code = s['@REF_AREA']

            if 'Obs' not in s or not s['Obs']:
                continue # Skip if no observations for this country

            for obs in s['Obs']:
                all_records.append({
                    'Country': country_code,
                    'Date': obs['@TIME_PERIOD'],
                    'Value': float(obs['@OBS_VALUE'])
                })

        if not all_records:
            print(f"⚠️ No observations found in the IMF data for {column_suffix}.")
            return pd.DataFrame()

        df = pd.DataFrame(all_records)
        df['Date'] = pd.to_datetime(df['Date'])

        # Pivot the table to get countries as columns
        pivot_df = df.pivot(index='Date', columns='Country', values='Value')
        pivot_df.columns = [f"{col}_{column_suffix}" for col in pivot_df.columns]

        print(f"✅ Successfully fetched quarterly data for {len(pivot_df.columns)} countries.")
        return pivot_df

    except requests.exceptions.HTTPError as http_err:
        print(f"❌ HTTP error fetching IMF data for {column_suffix}: {http_err} - Check indicator/country codes.")
        return pd.DataFrame()
    except Exception as e:
        print(f"❌ An error occurred while fetching IMF data for {column_suffix}: {e}")
        return pd.DataFrame()

# --- Main Script Execution ---
# --- Step 1: Load and Prepare Data ---

period = "10y"
interval = "1d"

print("\n--- Downloading Price Data via yfinance ---")
tickers = [
    "SPY", "MDY", "IJR", "XLU", "XLK", "XLV", "XLF", "XLY",
    "XLP", "XLE", "XLI", "XLC", "EWJ", "EWG", "EWU", "EWC",
    "EWA", "EWW", "EWS", "EWH", "XLB"
]

etf_expense_mapping = {
    "SPY": 0.000945, "MDY": 0.0023, "IJR": 0.0006, "XLU": 0.0008,
    "XLK": 0.0008, "XLV": 0.0008, "XLF": 0.0008, "XLY": 0.0008,
    "XLP": 0.0008, "XLE": 0.0008, "XLI": 0.0008, "XLC": 0.0008,
    "EWJ": 0.0050, "EWG": 0.0050, "EWU": 0.0050, "EWC": 0.0050,
    "EWA": 0.0050, "EWW": 0.0050, "EWS": 0.0050, "EWH": 0.0050,
    "XLB": 0.0008
}

fx = ["EUR=X", "GBP=X", "JPY=X", "CNY=X", "AUD=X", "TWD=X", "NZD=X", "INR=X", "BRL=X", "NOK=X", "SEK=X", "CHF=X", "PLN=X", "ZAR=X", "CAD=X", "MXN=X","HUF=X"]

price_data_raw = yf.download(tickers, period=period, interval=interval, progress=False)
fx_data_raw = yf.download(fx, period=period, interval=interval, progress=False)

# --- Process FX Data ---
print("\n--- Processing FX Data ---")
if fx_data_raw.empty:
    print("❌ Critical Error: Failed to download ANY FX data from yfinance.")
    fx_daily = pd.DataFrame()
else:
    fx_daily = fx_data_raw.get('Close', pd.DataFrame())
    failed_fx = fx_daily.columns[fx_daily.isna().all()].tolist()
    if failed_fx:
        print(f"⚠️ Failed to download data for {len(failed_fx)} FX pair(s): {', '.join(sorted(failed_fx))}")
        fx_daily.drop(columns=failed_fx, inplace=True)
    fx_daily.columns = fx_daily.columns.str.replace('=X', '')
    if not fx_daily.empty:
        print(f"✅ FX data processed successfully. Shape: {fx_daily.shape}")
    else:
        print("⚠️ All FX tickers failed to download.")

# --- Process Price Data ---
if price_data_raw.empty:
    print("❌ Critical Error: Failed to download ANY stock price data from yfinance.")
    prices_daily, prices_weekly, prices_dividends = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
else:
    close_prices = price_data_raw.get('Close', pd.DataFrame())
    failed_tickers = close_prices.columns[close_prices.isna().all()].tolist()
    if failed_tickers:
        print(f"⚠️ Failed to download data for {len(failed_tickers)} securities: {', '.join(sorted(failed_tickers))}")
        price_data_raw.drop(columns=failed_tickers, level=1, inplace=True)

    print("✅ Price data downloaded successfully.")
    ohlc_data = price_data_raw[['Open', 'High', 'Low', 'Close']]
    prices_daily = ohlc_data.groupby(level=1, axis=1).mean()
    prices_weekly = prices_daily.resample('W-FRI').mean()
    print(f"Daily average prices created. Shape: {prices_daily.shape}")
    print(f"Weekly average prices created. Shape: {prices_weekly.shape}")

    # --- Fetch Dividend Data ---
    print("\n--- Fetching Dividend Data ---")
    all_dividends = []
    successful_tickers = prices_daily.columns.unique()
    for ticker in successful_tickers:
        try:
            stock = yf.Ticker(ticker)
            divs = stock.dividends
            if divs.index.tz is not None:
                divs.index = divs.index.tz_localize(None)
            if not divs.empty:
                divs = divs[divs.index >= prices_daily.index.min()]
                all_dividends.append(divs.to_frame(name=ticker))
        except Exception:
            pass # Suppress dividend fetch errors

    if all_dividends:
        prices_dividends = pd.concat(all_dividends, axis=1)
        print(f"✅ Dividends data created. Shape: {prices_dividends.shape}")
    else:
        print("No dividend data found.")
        prices_dividends = pd.DataFrame()

cutoff_date = prices_daily.index.min().to_period('Q').start_time if not prices_daily.empty else None

# --- Define local CSV file paths ---
csv_files_to_import = [
    r"/home/felhasznalo/Scrapers/Data/CSV/shortsales_tw.csv", r"/home/felhasznalo/Scrapers/Data/CSV/spread_bank_lend_depo_in.csv", r"/home/felhasznalo/Scrapers/Data/CSV/spread_futures_tw.csv", r"/home/felhasznalo/Scrapers/Data/CSV/spread_vix_us.csv", r"/home/felhasznalo/Scrapers/Data/CSV/tankan_employment_jp.csv", r"/home/felhasznalo/Scrapers/Data/CSV/tankan_manu_jp.csv", r"/home/felhasznalo/Scrapers/Data/CSV/tankan_nonmanu_jp.csv", r"/home/felhasznalo/Scrapers/Data/CSV/tourism_hk.csv", r"/home/felhasznalo/Scrapers/Data/CSV/transfdepo_outstanding_tw.csv", r"/home/felhasznalo/Scrapers/Data/CSV/vix_hsi_hk.csv", r"/home/felhasznalo/Scrapers/Data/CSV/vix_tw.csv", r"/home/felhasznalo/Scrapers/Data/CSV/weighting_sectors_sp500_us.csv", r"/home/felhasznalo/Scrapers/Data/CSV/accounts_trading_tw.csv", r"/home/felhasznalo/Scrapers/Data/CSV/approvals_housing_au.csv", r"/home/felhasznalo/Scrapers/Data/CSV/assetalloc_aaii_us.csv", r"/home/felhasznalo/Scrapers/Data/CSV/backlog_orders_tw.csv", r"/home/felhasznalo/Scrapers/Data/CSV/breadth_eps_growth_us.csv", r"/home/felhasznalo/Scrapers/Data/CSV/breadth_indices_eu.csv", r"/home/felhasznalo/Scrapers/Data/CSV/breadth_indices_global.csv", r"/home/felhasznalo/Scrapers/Data/CSV/breadth_nikkei225_jp.csv", r"/home/felhasznalo/Scrapers/Data/CSV/breadth_rev_tw.csv", r"/home/felhasznalo/Scrapers/Data/CSV/breadth_sectors_sp500_us.csv", r"/home/felhasznalo/Scrapers/Data/CSV/buys_net_jp.csv", r"/home/felhasznalo/Scrapers/Data/CSV/conf_cons_eu.csv", r"/home/felhasznalo/Scrapers/Data/CSV/conf_cons_in.csv", r"/home/felhasznalo/Scrapers/Data/CSV/conf_cons_tw.csv", r"/home/felhasznalo/Scrapers/Data/CSV/conf_nab_au.csv", r"/home/felhasznalo/Scrapers/Data/CSV/conf_westpac_au.csv", r"/home/felhasznalo/Scrapers/Data/CSV/corridor_intrate_cn.csv", r"/home/felhasznalo/Scrapers/Data/CSV/corridor_intrate_in.csv", r"/home/felhasznalo/Scrapers/Data/CSV/cot_eur.csv", r"/home/felhasznalo/Scrapers/Data/CSV/cot_jpy_jp.csv", r"/home/felhasznalo/Scrapers/Data/CSV/cycle_manufacturing_mm_global.csv", r"/home/felhasznalo/Scrapers/Data/CSV/diffusion_cli_global.csv", r"/home/felhasznalo/Scrapers/Data/CSV/diffusion_inflation_global.csv", r"/home/felhasznalo/Scrapers/Data/CSV/diffusion_pmi_global.csv", r"/home/felhasznalo/Scrapers/Data/CSV/diffusion_ratemoves_cb_global.csv", r"/home/felhasznalo/Scrapers/Data/CSV/diffusion_unrate_global.csv", r"/home/felhasznalo/Scrapers/Data/CSV/drawdown_eps_indices_global.csv", r"/home/felhasznalo/Scrapers/Data/CSV/drawdown_eps_sectors_sp500_us.csv", r"/home/felhasznalo/Scrapers/Data/CSV/eps_est_cn.csv", r"/home/felhasznalo/Scrapers/Data/CSV/eps_fwd_au.csv", r"/home/felhasznalo/Scrapers/Data/CSV/eps_fwd_eu.csv", r"/home/felhasznalo/Scrapers/Data/CSV/eps_fwd_nikkei_jp.csv", r"/home/felhasznalo/Scrapers/Data/CSV/expect_econ_mm_global.csv", r"/home/felhasznalo/Scrapers/Data/CSV/export_electronic_in.csv", r"/home/felhasznalo/Scrapers/Data/CSV/export_electronic_tw.csv", r"/home/felhasznalo/Scrapers/Data/CSV/fdi_in.csv", r"/home/felhasznalo/Scrapers/Data/CSV/feargreed_mm_au.csv", r"/home/felhasznalo/Scrapers/Data/CSV/feargreed_mm_br.csv", r"/home/felhasznalo/Scrapers/Data/CSV/feargreed_mm_cn.csv", r"/home/felhasznalo/Scrapers/Data/CSV/feargreed_mm_eu.csv", r"/home/felhasznalo/Scrapers/Data/CSV/feargreed_mm_in.csv", r"/home/felhasznalo/Scrapers/Data/CSV/feargreed_mm_jp.csv", r"/home/felhasznalo/Scrapers/Data/CSV/feargreed_mm_tw.csv", r"/home/felhasznalo/Scrapers/Data/CSV/feargreed_mm_us.csv", r"/home/felhasznalo/Scrapers/Data/CSV/fwdspot_hkd_hk.csv", r"/home/felhasznalo/Scrapers/Data/CSV/growth_eps_countries_global.csv", r"/home/felhasznalo/Scrapers/Data/CSV/growth_eps_industries_global.csv", r"/home/felhasznalo/Scrapers/Data/CSV/holdings_stocks_boj_jp.csv", r"/home/felhasznalo/Scrapers/Data/CSV/index_ccl_hk.csv", r"/home/felhasznalo/Scrapers/Data/CSV/index_cli_oecd_global.csv", r"/home/felhasznalo/Scrapers/Data/CSV/index_hsahp_hk.csv", r"/home/felhasznalo/Scrapers/Data/CSV/index_ici_in.csv", r"/home/felhasznalo/Scrapers/Data/CSV/index_ifo_de.csv", r"/home/felhasznalo/Scrapers/Data/CSV/index_li_oecd_br.csv", r"/home/felhasznalo/Scrapers/Data/CSV/index_likeqiang_cn.csv", r"/home/felhasznalo/Scrapers/Data/CSV/index_surprise_earnings_citi_us.csv", r"/home/felhasznalo/Scrapers/Data/CSV/indicator_buffett_global.csv", r"/home/felhasznalo/Scrapers/Data/CSV/indices_ccfi_scfi_cn.csv", r"/home/felhasznalo/Scrapers/Data/CSV/indices_leading_lagging_tw.csv", r"/home/felhasznalo/Scrapers/Data/CSV/indices_monitoring_tw.csv", r"/home/felhasznalo/Scrapers/Data/CSV/inflows_northbound_cn.csv", r"/home/felhasznalo/Scrapers/Data/CSV/inflows_southbound_cn.csv", r"/home/felhasznalo/Scrapers/Data/CSV/loan_industrials_in.csv", r"/home/felhasznalo/Scrapers/Data/CSV/loan_nonfood_in.csv", r"/home/felhasznalo/Scrapers/Data/CSV/loan_personal_in.csv", r"/home/felhasznalo/Scrapers/Data/CSV/loan_services_in.csv", r"/home/felhasznalo/Scrapers/Data/CSV/loans_hk.csv", r"/home/felhasznalo/Scrapers/Data/CSV/loans_housing_au.csv", r"/home/felhasznalo/Scrapers/Data/CSV/margin_maint_tw.csv", r"/home/felhasznalo/Scrapers/Data/CSV/margin_outstanding_cn.csv", r"/home/felhasznalo/Scrapers/Data/CSV/mktcap_indices_global.csv", r"/home/felhasznalo/Scrapers/Data/CSV/mm_funda_au.csv", r"/home/felhasznalo/Scrapers/Data/CSV/mm_funda_br.csv", r"/home/felhasznalo/Scrapers/Data/CSV/mm_funda_cn.csv", r"/home/felhasznalo/Scrapers/Data/CSV/mm_funda_eu.csv", r"/home/felhasznalo/Scrapers/Data/CSV/mm_funda_in.csv", r"/home/felhasznalo/Scrapers/Data/CSV/mm_funda_jp.csv", r"/home/felhasznalo/Scrapers/Data/CSV/mm_funda_tw.csv", r"/home/felhasznalo/Scrapers/Data/CSV/mm_funda_us.csv", r"/home/felhasznalo/Scrapers/Data/CSV/momentum_econ_jp.csv", r"/home/felhasznalo/Scrapers/Data/CSV/openint_foreign_tw.csv", r"/home/felhasznalo/Scrapers/Data/CSV/openint_institutional_tw.csv", r"/home/felhasznalo/Scrapers/Data/CSV/orders_dg_us.csv", r"/home/felhasznalo/Scrapers/Data/CSV/orders_industrial_de.csv", r"/home/felhasznalo/Scrapers/Data/CSV/orders_machine_jp.csv", r"/home/felhasznalo/Scrapers/Data/CSV/outlook_industrial_in.csv", r"/home/felhasznalo/Scrapers/Data/CSV/output_industrial_valueadded_cn.csv", r"/home/felhasznalo/Scrapers/Data/CSV/output_production_de.csv", r"/home/felhasznalo/Scrapers/Data/CSV/pe_countries_global.csv", r"/home/felhasznalo/Scrapers/Data/CSV/pe_fwd_jp.csv", r"/home/felhasznalo/Scrapers/Data/CSV/pmi_au.csv", r"/home/felhasznalo/Scrapers/Data/CSV/pmi_hk.csv", r"/home/felhasznalo/Scrapers/Data/CSV/pmi_jp.csv", r"/home/felhasznalo/Scrapers/Data/CSV/pmi_nmi_br.csv", r"/home/felhasznalo/Scrapers/Data/CSV/pmi_nmi_cn.csv", r"/home/felhasznalo/Scrapers/Data/CSV/pmi_nmi_eu.csv", r"/home/felhasznalo/Scrapers/Data/CSV/pmi_nmi_in.csv", r"/home/felhasznalo/Scrapers/Data/CSV/pmi_tw.csv", r"/home/felhasznalo/Scrapers/Data/CSV/portfolio_fpi_in.csv", r"/home/felhasznalo/Scrapers/Data/CSV/pos_net_foreign_jp.csv", r"/home/felhasznalo/Scrapers/Data/CSV/pos_net_institutional_tw.csv", r"/home/felhasznalo/Scrapers/Data/CSV/price_residential_au.csv", r"/home/felhasznalo/Scrapers/Data/CSV/prob_correction_market_global.csv", r"/home/felhasznalo/Scrapers/Data/CSV/prob_recession_eu.csv", r"/home/felhasznalo/Scrapers/Data/CSV/prob_recession_global.csv", r"/home/felhasznalo/Scrapers/Data/CSV/prod_industrial_in.csv", r"/home/felhasznalo/Scrapers/Data/CSV/profits_corporate_us.csv", r"/home/felhasznalo/Scrapers/Data/CSV/profits_industrial_cn.csv", r"/home/felhasznalo/Scrapers/Data/CSV/putcall_cboe_us.csv", r"/home/felhasznalo/Scrapers/Data/CSV/putcall_hk.csv", r"/home/felhasznalo/Scrapers/Data/CSV/putcall_tw.csv", r"/home/felhasznalo/Scrapers/Data/CSV/rates_hbor_hk.csv", r"/home/felhasznalo/Scrapers/Data/CSV/rates_mclr_baserate_in.csv", r"/home/felhasznalo/Scrapers/Data/CSV/ratio_futures_longshort_individual_tw.csv", r"/home/felhasznalo/Scrapers/Data/CSV/ratio_rxi_kxi_us.csv", r"/home/felhasznalo/Scrapers/Data/CSV/reserves_foreign_cn.csv", r"/home/felhasznalo/Scrapers/Data/CSV/reserves_foreign_hk.csv", r"/home/felhasznalo/Scrapers/Data/CSV/reserves_foreign_in.csv", r"/home/felhasznalo/Scrapers/Data/CSV/reserves_fx_br.csv", r"/home/felhasznalo/Scrapers/Data/CSV/rev_ni_tw.csv", r"/home/felhasznalo/Scrapers/Data/CSV/rev_ops_tw.csv", r"/home/felhasznalo/Scrapers/Data/CSV/sales_residential_hk.csv", r"/home/felhasznalo/Scrapers/Data/CSV/sales_retail_hk.csv", r"/home/felhasznalo/Scrapers/Data/CSV/sales_retail_industries_hk.csv", r"/home/felhasznalo/Scrapers/Data/CSV/sales_retail_jp.csv", r"/home/felhasznalo/Scrapers/Data/CSV/sent_econ_zew_de.csv", r"/home/felhasznalo/Scrapers/Data/CSV/sent_econ_zew_eu.csv", r"/home/felhasznalo/Scrapers/Data/CSV/sentiment_econ_eu.csv", r"/home/felhasznalo/Scrapers/Data/CSV/sentiment_smart_dumb_money_us.csv", r"/home/felhasznalo/Scrapers/Data/CSV/shorts_outstanding_cn.csv"
]


print("\n--- Processing Local CSV Files ---")
quarterly_dfs = import_and_resample_files(csv_files_to_import, 'QS', start_date_filter=cutoff_date)
if quarterly_dfs:
    market_quarterly = pd.concat(quarterly_dfs.values(), axis=1, join='outer')
    market_quarterly.sort_index(inplace=True)
else:
    print("No local CSV data was loaded. Creating an empty DataFrame.")
    market_quarterly = pd.DataFrame()

if not market_quarterly.empty and market_quarterly.columns.has_duplicates:
    market_quarterly = market_quarterly.loc[:, ~market_quarterly.columns.duplicated()]
    print("Handled duplicate column names by keeping the first occurrence.")

# --- Fetch and merge external factors ---
if not market_quarterly.empty:
    start_date_factors = market_quarterly.index.min()
    try:
        print("\n--- Fetching Fama-French & Carhart Factors ---")
        ff_raw = gfm.get_factors(model='ff6', frequency='m', start_date=start_date_factors)
        carhart_raw = gfm.get_factors(model='carhart', frequency='m', start_date=start_date_factors)
        if ff_raw is not None and carhart_raw is not None:
            factors_monthly = ff_raw.merge(carhart_raw[['MOM']], left_index=True, right_index=True, how='left')
            factors_quarterly = factors_monthly.resample('QS').mean()
            factors_quarterly.columns = [f"FF_Factor_{col}" for col in factors_quarterly.columns]
            market_quarterly = market_quarterly.merge(factors_quarterly, left_index=True, right_index=True, how='left')
            print("✅ External factors merged successfully.")
    except Exception as e:
        print(f"An error occurred during factor processing: {e}")

# --- Step 2 (Original): Impute the market_quarterly data ---
if not market_quarterly.empty and market_quarterly.isnull().values.any():
    print("\n--- Imputing Market Data ---")
    print(f"Total NaNs before imputation: {market_quarterly.isnull().sum().sum()}")
    market_quarterly_imputed = market_quarterly.interpolate(method='time', limit_direction='both')
    if market_quarterly_imputed.isnull().sum().sum() > 0:
        scaler = RobustScaler()
        scaled_data = scaler.fit_transform(market_quarterly_imputed)
        imputer = KNNImputer(n_neighbors=5)
        imputed_scaled_data = imputer.fit_transform(scaled_data)
        market_quarterly_imputed = pd.DataFrame(scaler.inverse_transform(imputed_scaled_data),
                                                index=market_quarterly.index,
                                                columns=market_quarterly.columns)
    market_quarterly = market_quarterly_imputed
    print(f"Total NaNs after final imputation: {market_quarterly.isnull().sum().sum()}")
else:
    print("\n--- No NaNs found in market_quarterly or DataFrame is empty. Skipping imputation. ---")


# --- Step 3 (New): Fetch, Assemble, and Process API Data ---
country_mapping = {
    'Austria': 'AT', 'Belgium': 'BE', 'Croatia': 'HR', 'Cyprus': 'CY', 'Estonia': 'EE',
    'Finland': 'FI', 'France': 'FR', 'Germany': 'DE', 'Greece': 'GR', 'Ireland': 'IE',
    'Italy': 'IT', 'Latvia': 'LV', 'Lithuania': 'LT', 'Luxembourg': 'LU', 'Malta': 'MT',
    'Netherlands': 'NL', 'Portugal': 'PT', 'Slovakia': 'SK', 'Slovenia': 'SI', 'Spain': 'ES',
    'Sweden': 'SE', 'Norway': 'NO', 'United Kingdom': 'GB', 'Japan': 'JP', 'China': 'CN',
    'Canada': 'CA', 'Australia': 'AU', 'Mexico': 'MX', 'India': 'IN',
    'New Zealand': 'NZ', 'Brazil': 'BR', 'Switzerland': 'CH', 'Poland': 'PL',
    'South Africa': 'ZA', 'Hungary': 'HU', 'Indonesia': 'ID', 'South Korea': 'KR', 'Turkey': 'TR'
    # Note: Taiwan (TW) is generally not available in the main IMF IFS database
}
country_codes = list(country_mapping.values())

indicators_to_fetch = [
    {"code": "BCA_BP6_USD", "suffix": "CA"},
    {"code": "TXG_FOB_USD", "suffix": "Exports"},
    {"code": "TMG_CIF_USD", "suffix": "Imports"},
    {"code": "PCPI_IX",     "suffix": "CPI"},
    {"code": "PPI_IX",      "suffix": "PPI"},
    {"code": "NGDPD",       "suffix": "GDP"}
]

# Initialize an empty DataFrame to hold all the macro data
macro_data_all = pd.DataFrame()

for indicator in indicators_to_fetch:
    wb_df = fetch_imf_data(
        period=period,
        indicator_code=indicator["code"],
        countries=country_codes,
        column_suffix=indicator["suffix"]
    )

    if not wb_df.empty:
        if macro_data_all.empty:
            macro_data_all = wb_df
        else:
            # Use an outer join to combine all indicators over time
            macro_data_all = macro_data_all.merge(wb_df, left_index=True, right_index=True, how='outer')

# --- Step 4: Export Data to Separate Locations ---
print("\n--- Step 4: Exporting Processed Data ---")
exported_files = []

def export_df(df, name, directory):
    os.makedirs(directory, exist_ok=True)
    if not df.empty:
        path = os.path.join(directory, f"{name}.csv")
        df.to_csv(path)
        print(f"Successfully exported {name} to {path}")
        exported_files.append(path)
    else:
        print(f"⚠️ {name} DataFrame is empty, skipping export.")

try:
    # --- Export financial and local data to 'Data' folder ---
    data_dir = os.path.join(os.getcwd(), 'Data')

    if etf_expense_mapping:
        df_expenses = pd.DataFrame.from_dict(etf_expense_mapping, orient='index', columns=['ExpenseRatio_Percent'])
        df_expenses.index.name = 'Ticker'
        export_df(df_expenses, "etf_expenses", data_dir)

    export_df(prices_daily, "prices_daily", data_dir)
    export_df(prices_weekly, "prices_weekly", data_dir)
    export_df(prices_dividends, "prices_dividends", data_dir)
    export_df(fx_daily, "fx_prices_daily", data_dir)
    export_df(market_quarterly, "market_quarterly", data_dir)

    # --- Export the new, separate macro data to 'Data/FEER' folder ---
    feer_dir = os.path.join(os.getcwd(), 'Data', 'FEER')
    export_df(macro_data_all, "macro_all", feer_dir)

except Exception as e:
    print(f"❌ An error occurred during file export: {e}")

print("\n--- Output CSV File Links ---")
if exported_files:
    for path in sorted(exported_files):
        print(path)
else:
    print("No files were exported.")