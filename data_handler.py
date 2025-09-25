import pandas as pd
import os
import numpy as np
import getfactormodels as gfm
import yfinance as yf
import sys

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
        except Exception as e:
            print(f"Could not process {file_path}: {e}")
    return dataframes

# --- Main Script Execution ---
# --- Step 1: Load and Prepare Data ---

print("\n--- Downloading Price Data via yfinance ---")
tickers = [
    "SPY", "MDY", "IJR", "XLU", "XLK", "XLV", "XLF", "XLY",
    "XLP", "XLE", "XLI", "IYR", "EWJ", "EWG", "EWU", "EWC",
    "EWA", "EWW", "EWS", "EWH", "XLB"
]

# Download the full OHLCV data
price_data_raw = yf.download(tickers, period="10y", interval="1d", progress=False)

if price_data_raw.empty:
    print("❌ Critical Error: Failed to download ANY stock price data from yfinance.")
    prices_daily = pd.DataFrame()
    prices_weekly = pd.DataFrame()
    prices_dividends = pd.DataFrame() # Ensure dataframe exists
else:
    # 1. Identify failed tickers using the 'Close' price column
    close_prices = price_data_raw['Close']
    failed_tickers = close_prices.columns[close_prices.isna().all()].tolist()

    if failed_tickers:
        print(f"⚠️ Failed to download data for {len(failed_tickers)} security/securities:")
        for ticker in sorted(list(set(failed_tickers))):
            print(f"   - {ticker}")

        price_data_raw.drop(columns=failed_tickers, level=1, inplace=True)

    print("✅ Price data downloaded successfully.")

    # --- MODIFIED: Use groupby() instead of the deprecated 'level' argument ---
    # 2. Select OHLC data, then average across the metrics for each ticker
    ohlc_data = price_data_raw[['Open', 'High', 'Low', 'Close']]
    prices_daily = ohlc_data.groupby(level=1, axis=1).mean()

    # 3. Create weekly prices by resampling the new daily average prices
    prices_weekly = prices_daily.resample('W-FRI').mean()

    print(f"Daily average prices created. Shape: {prices_daily.shape}")
    print(f"Weekly average prices created. Shape: {prices_weekly.shape}")
    # --- END MODIFICATION ---


    # --- Fetch Dividend Data ---
    print("\n--- Fetching Dividend Data ---")
    all_dividends = []
    # Get a clean list of tickers from the new simplified columns
    successful_tickers = prices_daily.columns.unique()

    for ticker in successful_tickers:
        try:
            stock = yf.Ticker(ticker)
            divs = stock.dividends

            if divs.index.tz is not None:
                 divs.index = divs.index.tz_localize(None)

            if not divs.empty:
                # Filter dividends to match the price data's time frame
                divs = divs[divs.index >= prices_daily.index.min()]
                all_dividends.append(divs.to_frame(name=ticker))
                print(f"  Successfully fetched dividends for {ticker}")
        except Exception as e:
            print(f"  Could not fetch dividends for {ticker}: {e}")

    if all_dividends:
        prices_dividends = pd.concat(all_dividends, axis=1)
        print(f"✅ Dividends data created. Shape: {prices_dividends.shape}")
    else:
        print("No dividend data found for the given tickers and period.")
        prices_dividends = pd.DataFrame()

# Set the cutoff date for other data based on the earliest stock price
cutoff_date = prices_daily.index.min().to_period('Q').start_time if not prices_daily.empty else None

# Load economic indicator data from local CSVs
csv_files_to_import = [
    r"/home/felhasznalo/Scrapers/Data/CSV/shortsales_tw.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/spread_bank_lend_depo_in.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/spread_futures_tw.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/spread_vix_us.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/tankan_employment_jp.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/tankan_manu_jp.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/tankan_nonmanu_jp.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/tourism_hk.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/transfdepo_outstanding_tw.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/vix_hsi_hk.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/vix_tw.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/weighting_sectors_sp500_us.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/accounts_trading_tw.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/approvals_housing_au.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/assetalloc_aaii_us.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/backlog_orders_tw.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/breadth_eps_growth_us.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/breadth_indices_eu.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/breadth_indices_global.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/breadth_nikkei225_jp.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/breadth_rev_tw.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/breadth_sectors_sp500_us.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/buys_net_jp.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/conf_cons_eu.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/conf_cons_in.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/conf_cons_tw.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/conf_nab_au.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/conf_westpac_au.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/corridor_intrate_cn.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/corridor_intrate_in.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/cot_eur.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/cot_jpy_jp.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/cycle_manufacturing_mm_global.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/diffusion_cli_global.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/diffusion_inflation_global.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/diffusion_pmi_global.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/diffusion_ratemoves_cb_global.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/diffusion_unrate_global.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/drawdown_eps_indices_global.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/drawdown_eps_sectors_sp500_us.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/eps_est_cn.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/eps_fwd_au.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/eps_fwd_eu.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/eps_fwd_nikkei_jp.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/expect_econ_mm_global.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/export_electronic_in.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/export_electronic_tw.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/fdi_in.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/feargreed_mm_au.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/feargreed_mm_br.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/feargreed_mm_cn.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/feargreed_mm_eu.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/feargreed_mm_in.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/feargreed_mm_jp.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/feargreed_mm_tw.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/feargreed_mm_us.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/fwdspot_hkd_hk.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/growth_eps_countries_global.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/growth_eps_industries_global.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/holdings_stocks_boj_jp.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/index_ccl_hk.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/index_cli_oecd_global.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/index_hsahp_hk.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/index_ici_in.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/index_ifo_de.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/index_li_oecd_br.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/index_likeqiang_cn.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/index_surprise_earnings_citi_us.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/indicator_buffett_global.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/indices_ccfi_scfi_cn.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/indices_leading_lagging_tw.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/indices_monitoring_tw.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/inflows_northbound_cn.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/inflows_southbound_cn.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/loan_industrials_in.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/loan_nonfood_in.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/loan_personal_in.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/loan_services_in.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/loans_hk.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/loans_housing_au.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/margin_maint_tw.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/margin_outstanding_cn.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/mktcap_indices_global.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/mm_funda_au.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/mm_funda_br.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/mm_funda_cn.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/mm_funda_eu.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/mm_funda_in.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/mm_funda_jp.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/mm_funda_tw.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/mm_funda_us.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/momentum_econ_jp.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/openint_foreign_tw.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/openint_institutional_tw.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/orders_dg_us.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/orders_industrial_de.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/orders_machine_jp.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/outlook_industrial_in.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/output_industrial_valueadded_cn.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/output_production_de.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/pe_countries_global.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/pe_fwd_jp.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/pmi_au.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/pmi_hk.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/pmi_jp.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/pmi_nmi_br.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/pmi_nmi_cn.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/pmi_nmi_eu.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/pmi_nmi_in.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/pmi_tw.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/portfolio_fpi_in.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/pos_net_foreign_jp.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/pos_net_institutional_tw.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/price_residential_au.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/prob_correction_market_global.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/prob_recession_eu.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/prob_recession_global.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/prod_industrial_in.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/profits_corporate_us.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/profits_industrial_cn.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/putcall_cboe_us.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/putcall_hk.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/putcall_tw.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/rates_hbor_hk.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/rates_mclr_baserate_in.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/ratio_futures_longshort_individual_tw.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/ratio_rxi_kxi_us.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/reserves_foreign_cn.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/reserves_foreign_hk.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/reserves_foreign_in.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/reserves_fx_br.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/rev_ni_tw.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/rev_ops_tw.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/sales_residential_hk.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/sales_retail_hk.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/sales_retail_industries_hk.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/sales_retail_jp.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/sent_econ_zew_de.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/sent_econ_zew_eu.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/sentiment_econ_eu.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/sentiment_smart_dumb_money_us.csv",
    r"/home/felhasznalo/Scrapers/Data/CSV/shorts_outstanding_cn.csv"
]

if csv_files_to_import:
    print("\n--- Input CSV Files to be Processed ---")
    for path in csv_files_to_import:
        print(path)

    quarterly_dfs = import_and_resample_files(csv_files_to_import, 'QS', start_date_filter=cutoff_date)
    market_quarterly = pd.concat(quarterly_dfs.values(), axis=1, join='outer')
    market_quarterly.sort_index(inplace=True)
else:
    print("\n--- No local CSVs to process, continuing with price data only ---")
    market_quarterly = pd.DataFrame()

# Fix for Duplicate Column Names
if not market_quarterly.empty and market_quarterly.columns.has_duplicates:
    print("Found and handling duplicate column names...")
    cols = pd.Series(market_quarterly.columns)
    for dup in cols[cols.duplicated()].unique():
        dup_locs = cols[cols == dup].index
        new_names = [f"{dup}_{i}" for i, _ in enumerate(dup_locs, 1)]
        rename_dict = dict(zip(dup_locs, new_names))
        cols.update(pd.Series(rename_dict))
    market_quarterly.columns = cols


# Fetch and merge external factors
if not market_quarterly.empty:
    try:
        ff_raw = gfm.get_factors(model='ff6', frequency='m', start_date=market_quarterly.index.min())
        carhart_raw = gfm.get_factors(model='carhart', frequency='m', start_date=market_quarterly.index.min())
        if ff_raw is not None and carhart_raw is not None:
            factors_monthly = ff_raw.merge(carhart_raw[['MOM']], left_index=True, right_index=True, how='left')
            factors_quarterly = factors_monthly.resample('QS').mean()
            factors_quarterly.columns = [f"FF_Factor_{col}" for col in factors_quarterly.columns]
            market_quarterly = market_quarterly.merge(factors_quarterly, left_index=True, right_index=True, how='left')
            print("External factors merged successfully.")
    except Exception as e:
        print(f"An error occurred during factor processing: {e}")

# --- Step 2: Imputation using a Simplified, Robust Strategy ---
if market_quarterly is not None and not market_quarterly.empty and market_quarterly.isnull().values.any():
    print("\n--- Step 2: Imputing Missing Data ---")
    nans_before = market_quarterly.isnull().sum().sum()
    print(f"Total NaNs before imputation: {nans_before}")

    print("Applying time-series interpolation...")
    market_quarterly_imputed = market_quarterly.interpolate(method='time', limit_direction='both')
    nans_after_interp = market_quarterly_imputed.isnull().sum().sum()
    print(f"NaNs remaining after interpolation: {nans_after_interp}")

    if nans_after_interp > 0:
        print("Applying KNN Imputer for remaining gaps...")
        scaler = RobustScaler()
        scaled_data = scaler.fit_transform(market_quarterly_imputed)
        imputer = KNNImputer(n_neighbors=5)
        imputed_scaled_data = imputer.fit_transform(scaled_data)
        market_quarterly_imputed = pd.DataFrame(scaler.inverse_transform(imputed_scaled_data),
                                                index=market_quarterly.index,
                                                columns=market_quarterly.columns)

    print(f"Total NaNs after final imputation: {market_quarterly_imputed.isnull().sum().sum()}")
    market_quarterly = market_quarterly_imputed
else:
    print("\n--- No NaNs found or DataFrame is empty. Skipping imputation. ---")

# --- Step 3: Export the Processed DataFrames ---
print("\n--- Step 3: Exporting Processed Data ---")

output_dir = os.path.join(os.getcwd(), 'Data')
os.makedirs(output_dir, exist_ok=True)
exported_files = []

try:
    if not market_quarterly.empty:
        market_quarterly_path = os.path.join(output_dir, "market_quarterly.csv")
        market_quarterly.to_csv(market_quarterly_path)
        print(f"Successfully exported market_quarterly to {market_quarterly_path}")
        exported_files.append(market_quarterly_path)
    else:
        print("market_quarterly DataFrame is empty, skipping export.")

    if not prices_daily.empty:
        prices_daily_path = os.path.join(output_dir, "prices_daily.csv")
        prices_daily.to_csv(prices_daily_path)
        print(f"Successfully exported prices_daily to {prices_daily_path}")
        exported_files.append(prices_daily_path)
    else:
        print("prices_daily DataFrame is empty, skipping export.")

    if not prices_weekly.empty:
        prices_weekly_path = os.path.join(output_dir, "prices_weekly.csv")
        prices_weekly.to_csv(prices_weekly_path)
        print(f"Successfully exported prices_weekly to {prices_weekly_path}")
        exported_files.append(prices_weekly_path)
    else:
        print("prices_weekly DataFrame is empty, skipping export.")

    if not prices_dividends.empty:
        prices_dividends_path = os.path.join(output_dir, "prices_dividends.csv")
        prices_dividends.to_csv(prices_dividends_path)
        print(f"Successfully exported prices_dividends to {prices_dividends_path}")
        exported_files.append(prices_dividends_path)
    else:
        print("prices_dividends DataFrame is empty, skipping export.")

except Exception as e:
    print(f"An error occurred during file export: {e}")

# --- Final output of all CSV file paths ---
print("\n--- Output CSV File Links ---")
if exported_files:
    for path in exported_files:
        print(path)
else:
    print("No files were exported.")