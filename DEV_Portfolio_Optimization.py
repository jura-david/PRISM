#%% Data Preparation and Integration Script
import pandas as pd
import notebook as nb
import os
import numpy as np
import getfactormodels as gfm

# --- NEW: Import RobustScaler ---
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
csv_files_to_import = [
    # Full list of your 150+ CSV files
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
stock_price_files = [
    r"/mnt/c/Users/Felhasznalo/Desktop/Finance_v2/Portfolio/Data/Stock_Prices.csv",
    r"/mnt/c/Users/Felhasznalo/Desktop/Finance_v2/Portfolio/Data/URTH_Prices.csv"
]

# Load data
price_dfs = import_and_resample_files(stock_price_files, target_frequency='W-FRI')
prices_weekly = pd.concat(price_dfs.values(), axis=1) if price_dfs else pd.DataFrame()
cutoff_date = prices_weekly.index.min().to_period('Q').start_time if not prices_weekly.empty else None
prices_weekly = prices_weekly.T.drop_duplicates().T  # Ensure no duplicate columns
quarterly_dfs = import_and_resample_files(csv_files_to_import, 'QS', start_date_filter=cutoff_date)
market_quarterly = pd.concat(quarterly_dfs.values(), axis=1, join='outer')
market_quarterly.sort_index(inplace=True)

#  Fix for Duplicate Column Names
if market_quarterly.columns.has_duplicates:
    print("Found and handling duplicate column names...")
    cols = pd.Series(market_quarterly.columns)
    for dup in cols[cols.duplicated()].unique():
        new_names = [f"{dup}_{i}" if i != 0 else dup for i in range(sum(cols == dup))]
        dup_indices = cols[cols == dup].index
        cols.update(pd.Series(dict(zip(dup_indices, new_names))))
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

    # Strategy 1: Time-series interpolation
    # This is ideal for filling gaps within a series.
    print("Applying time-series interpolation...")
    market_quarterly_imputed = market_quarterly.interpolate(method='time', limit_direction='both')

    nans_after_interp = market_quarterly_imputed.isnull().sum().sum()
    print(f"NaNs remaining after interpolation: {nans_after_interp}")

    # Strategy 2: KNN Imputation for remaining gaps
    # This handles NaNs at the start/end of the series by using other columns.
    if nans_after_interp > 0:
        print("Applying KNN Imputer for remaining gaps...")
        # KNNImputer needs data to be scaled for distance calculations.
        # RobustScaler is used here as it's less sensitive to outliers.
        scaler = RobustScaler() # <-- MODIFIED
        scaled_data = scaler.fit_transform(market_quarterly_imputed)

        # n_neighbors is a key parameter; typically 3-7 is a good range
        imputer = KNNImputer(n_neighbors=5)
        imputed_scaled_data = imputer.fit_transform(scaled_data)

        # Inverse scale the data to its original representation
        market_quarterly_imputed = pd.DataFrame(scaler.inverse_transform(imputed_scaled_data),
                                                index=market_quarterly.index,
                                                columns=market_quarterly.columns)

    print(f"Total NaNs after final imputation: {market_quarterly_imputed.isnull().sum().sum()}")
    market_quarterly = market_quarterly_imputed

else:
    print("\n--- No NaNs found or DataFrame is empty. Skipping imputation. ---")

# Create new DF for current (last) row
current_market = market_quarterly.iloc[[-1]]

### Final Output ###
print("\n--- Final Results ---")
if market_quarterly is not None and not market_quarterly.empty:
    print("\n--- Final Imputed Quarterly Data (market_quarterly) ---")
    print(market_quarterly.head())
    print(f"\nShape: {market_quarterly.shape}")
    market_quarterly.info()
else:
    print("\nMarket quarterly data is empty.")

if not prices_weekly.empty:
    print("\n--- Resampled Weekly Price Data (prices_weekly) ---")
    print(prices_weekly.head())
    print(f"\nShape: {prices_weekly.shape}")

# --- Step 3: Export the Processed DataFrames ---
print("\n--- Step 3: Exporting Processed Data ---")

# Define the output directory relative to the current working directory
output_dir = os.path.join(os.getcwd(), 'Data')
os.makedirs(output_dir, exist_ok=True)

try:
    # Export the final, imputed market quarterly data
    market_quarterly_path = os.path.join(output_dir, "market_quarterly.csv")
    market_quarterly.to_csv(market_quarterly_path)
    print(f"Successfully exported market_quarterly to {market_quarterly_path}")

    # Export the weekly price data
    if not prices_weekly.empty:
        prices_weekly_path = os.path.join(output_dir, "prices_weekly.csv")
        prices_weekly.to_csv(prices_weekly_path)
        print(f"Successfully exported prices_weekly to {prices_weekly_path}")
    else:
        print("prices_weekly DataFrame is empty, skipping export.")

except Exception as e:
    print(f"An error occurred during file export: {e}")
#%% APSO Optimization Script
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import pandas as pd
from pyswarms.single import GlobalBestPSO
import numpy as np

# Import the required libraries
from sklearn.covariance import LedoitWolf
from scipy.optimize import minimize
from scipy.stats import kurtosis, skew
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap

# Set pandas display options
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 500)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --- Helper Functions ---

def calculate_quarterly_return(log_returns, weights):
    adjusted_weights = weights / 100.0 if np.sum(np.abs(weights)) > 0 else np.zeros_like(weights)
    portfolio_weekly_returns = np.dot(log_returns, adjusted_weights)
    cumulative_quarterly_log_return = np.sum(portfolio_weekly_returns)
    return np.exp(cumulative_quarterly_log_return) - 1

def calculate_shrunk_tracking_error(weights, shrunk_excess_return_cov_matrix):
    if np.sum(np.abs(weights)) == 0: return 1e-8
    w = weights / 100.0
    portfolio_variance = w.T @ shrunk_excess_return_cov_matrix @ w
    return np.sqrt(np.abs(portfolio_variance)) + 1e-8

def calculate_transaction_cost(current_weights, previous_weights, buy_cost=0.0005, sell_cost=0.0005):
    w_curr = current_weights / 100.0
    w_prev = previous_weights / 100.0
    delta_w = w_curr - w_prev
    buys = np.sum(delta_w[delta_w > 0])
    sells = np.sum(np.abs(delta_w[delta_w < 0]))
    return (buys * buy_cost) + (sells * sell_cost)

def scipy_objective_function(weights, log_returns, benchmark_log_returns, shrunk_cov_matrix, mu, diversification_penalty, previous_weights, num_stocks):
    gross_portfolio_return = calculate_quarterly_return(log_returns, weights)
    turnover_cost = calculate_transaction_cost(weights, previous_weights)
    net_portfolio_return = gross_portfolio_return - turnover_cost

    benchmark_return = np.exp(np.sum(benchmark_log_returns)) - 1
    excess_return = net_portfolio_return - benchmark_return

    tracking_error = calculate_shrunk_tracking_error(weights, shrunk_cov_matrix)

    hhi = np.sum((weights / 100.0)**2)
    if num_stocks > 1:
        hhi_normalized = (hhi - (1 / num_stocks)) / (1 - (1 / num_stocks))
    else:
        hhi_normalized = 1.0

    hhi_penalty = hhi_normalized

    cost = -excess_return + mu * tracking_error + diversification_penalty * hhi_penalty
    return cost

def adaptive_inertia_weight(iteration, max_iter, w_max=0.9, w_min=0.4, decay_rate=3.0):
    if max_iter <= 1: return w_min
    t_norm = iteration / (max_iter - 1)
    return w_min + (w_max - w_min) * np.exp(-decay_rate * t_norm)

def adaptive_cognitive_weight(iteration, max_iter, c1_initial=2.5, c1_final=0.5, decay_rate=3.0):
    if max_iter <= 1: return c1_final
    t_norm = iteration / (max_iter - 1)
    return c1_final + (c1_initial - c1_final) * np.exp(-decay_rate * t_norm)

# --- NEW: Hybrid "Delayed" Adaptive Social Weight Function ---
def delayed_adaptive_social_weight(iteration, max_iter, c2_initial=0.5, c2_final=2.5, delay_fraction=0.7):
    """
    Keeps c2 constant for a delay phase, then linearly adapts to the final value.
    This forces a long exploration period before exploitation begins.
    """
    if max_iter <= 1:
        return c2_initial

    # Check if we are in the delay (exploration) phase
    if iteration < max_iter * delay_fraction:
        return c2_initial
    else:
        # Start the adaptation (exploitation) phase
        # Calculate how many iterations are in the adaptation phase
        adaptation_total_iters = max_iter * (1.0 - delay_fraction)
        if adaptation_total_iters <= 0: return c2_final

        # Calculate how far into the adaptation phase we are
        adaptation_current_iter = iteration - (max_iter * delay_fraction)

        # Perform a linear ramp-up during the adaptation phase
        t_norm_adaptation = adaptation_current_iter / adaptation_total_iters
        return c2_initial + (c2_final - c2_initial) * t_norm_adaptation
# --- END NEW FUNCTION ---

def winsorize_series(series, lower_percentile=1, upper_percentile=99):
    if series.empty: return series
    lower_bound, upper_bound = np.percentile(series.dropna(), [lower_percentile, upper_percentile])
    return series.clip(lower=lower_bound, upper=upper_bound)

def create_pso_objective_function(log_returns_quarterly_slice, benchmark_log_returns_quarterly_slice,
                                  shrunk_cov_matrix, num_stocks, mu=1.0, diversification_penalty=0.1, previous_weights=None):
    def pso_objective(particles):
        costs = np.zeros(particles.shape[0])
        for i, p in enumerate(particles):
            abs_sum = np.sum(np.abs(p))
            weights = (p / abs_sum) * 100.0 if abs_sum > 0 else np.zeros_like(p)
            costs[i] = scipy_objective_function(weights, log_returns_quarterly_slice, benchmark_log_returns_quarterly_slice, shrunk_cov_matrix, mu, diversification_penalty, previous_weights, num_stocks)
        return costs
    return pso_objective

def multi_start_pso(objective_function, bounds, n_particles, dimensions, max_iter, n_starts,
                    w_decay_rate, c1_decay_rate, c2_increase_rate):
    best_costs_all_starts, best_positions_all_starts, all_iteration_costs = [], [], []
    w_initial, w_final = 0.9, 0.4; c1_initial, c1_final = 2.5, 0.5; c2_initial, c2_final = 0.5, 2.5
    for start in range(n_starts):
        print(f"\nStarting Adaptive PSO Run {start + 1}/{n_starts}")
        optimizer = GlobalBestPSO(n_particles, dimensions, options={'c1': c1_initial, 'c2': c2_initial, 'w': w_initial}, bounds=bounds)
        cost_history_this_run = []
        for i in range(max_iter):
            w = adaptive_inertia_weight(i, max_iter, w_initial, w_final, decay_rate=w_decay_rate)
            c1 = adaptive_cognitive_weight(i, max_iter, c1_initial, c1_final, decay_rate=c1_decay_rate)

            # --- UPDATED: Call the new delayed adaptive function ---
            c2 = delayed_adaptive_social_weight(i, max_iter, c2_initial, c2_final)
            # --- END UPDATE ---

            optimizer.swarm.options.update({'w': w, 'c1': c1, 'c2': c2})
            optimizer.optimize(objective_function, iters=1, verbose=False)
            cost_history_this_run.append(optimizer.swarm.best_cost)
            if i % 10 == 0 or i == max_iter - 1:
                print(f"Run {start + 1}, Iteration {i}: Best Cost={optimizer.swarm.best_cost:.4f}, w={w:.4f}, c1={c1:.4f}, c2={c2:.4f}")
        best_costs_all_starts.append(optimizer.swarm.best_cost)
        best_positions_all_starts.append(optimizer.swarm.best_pos)
        all_iteration_costs.append(cost_history_this_run)
    best_idx = np.argmin(best_costs_all_starts)
    return best_costs_all_starts[best_idx], best_positions_all_starts[best_idx], best_costs_all_starts, np.array(all_iteration_costs)

def perform_bootstrap_analysis_on_weights(log_returns, benchmark_returns, fixed_weights, n_bootstrap_samples):
    bootstrap_ir_samples = []
    original_bench_return = np.exp(np.sum(benchmark_returns)) - 1
    for i in range(n_bootstrap_samples):
        indices = np.random.choice(np.arange(len(log_returns)), size=len(log_returns), replace=True)
        bootstrap_log_ret, bootstrap_bench_ret = log_returns[indices], benchmark_returns[indices]
        bootstrap_excess_returns = bootstrap_log_ret - bootstrap_bench_ret

        # Handle cases with no variance
        if np.all(np.var(bootstrap_excess_returns, axis=0) < 1e-10):
            bootstrap_shrunk_cov = np.zeros((log_returns.shape[1], log_returns.shape[1]))
        else:
            bootstrap_shrunk_cov = LedoitWolf().fit(bootstrap_excess_returns).covariance_

        sim_port_ret = calculate_quarterly_return(bootstrap_log_ret, fixed_weights)
        sim_te = calculate_shrunk_tracking_error(fixed_weights, bootstrap_shrunk_cov)
        sim_ir = (sim_port_ret - original_bench_return) / sim_te if sim_te > 1e-8 else -np.inf
        bootstrap_ir_samples.append(sim_ir)
    return bootstrap_ir_samples

def center_plot_around_zero(data_list):
    combined_data = np.array([])
    for data in data_list:
        if data is not None and not data.empty:
            combined_data = np.concatenate((combined_data, data.values))

    finite_data = combined_data[np.isfinite(combined_data)]
    if finite_data.size > 0:
        max_abs = np.max(np.abs(finite_data))
        if max_abs > 0:
            plt.xlim(-max_abs * 1.1, max_abs * 1.1)

def run_portfolio_optimization(market_quarterly, prices_weekly,
                               n_bootstrap_samples,
                               set_particles, set_iters_pso, set_starts,
                               pso_w_decay_rate, pso_c1_decay_rate, pso_c2_increase_rate,
                               diversification_penalty=100.0):
    # --- Data Prep ---
    if prices_weekly is None or prices_weekly.empty: return None, None, 0, 0
    if not isinstance(market_quarterly.index, pd.PeriodIndex): market_quarterly.index = market_quarterly.index.to_period('Q')
    benchmark_col_name = 'SPY'
    if benchmark_col_name not in prices_weekly.columns: return None, None, 0, 0
    stock_cols = [col for col in prices_weekly.columns if col != benchmark_col_name]
    if not stock_cols: return None, None, 0, 0
    stock_prices = prices_weekly[stock_cols].copy().fillna(0)
    benchmark_prices = prices_weekly[[benchmark_col_name]].copy()
    stock_weekly_log_returns = np.log(stock_prices / stock_prices.shift(1)); rsp_weekly_log_returns = np.log(benchmark_prices / benchmark_prices.shift(1))
    stock_weekly_log_returns.replace([np.inf, -np.inf], 0, inplace=True); stock_weekly_log_returns.fillna(0, inplace=True)
    rsp_weekly_log_returns.replace([np.inf, -np.inf], 0, inplace=True); rsp_weekly_log_returns.fillna(0, inplace=True)
    for col in stock_weekly_log_returns.columns: stock_weekly_log_returns[col] = winsorize_series(stock_weekly_log_returns[col])
    rsp_weekly_log_returns[benchmark_col_name] = winsorize_series(rsp_weekly_log_returns[benchmark_col_name])
    common_index = stock_weekly_log_returns.index.intersection(rsp_weekly_log_returns.index)
    stock_weekly_log_returns = stock_weekly_log_returns.loc[common_index]; rsp_weekly_log_returns = rsp_weekly_log_returns.loc[common_index]
    stock_names_for_objective = list(stock_weekly_log_returns.columns)

    # --- Main Loop ---
    all_optimal_weights, all_information_ratios, valid_quarters = [], [], []
    bootstrap_results = {}
    all_pso_runs_iteration_costs, all_pso_best_costs_runs, all_bootstrap_irs_for_plot = [], [], []
    available_quarters = stock_weekly_log_returns.index.to_period('Q').unique()
    quarters_to_process = market_quarterly.index.intersection(available_quarters).sort_values()
    MIN_WEEKLY_OBSERVATIONS_PER_QUARTER = 10
    successful_quarters_count = 0; unsuccessful_quarters_count = 0
    previous_quarter_weights = np.zeros(len(stock_names_for_objective))

    for quarter in quarters_to_process:
        quarter_start, quarter_end = quarter.start_time, quarter.end_time
        log_returns_in_quarter = stock_weekly_log_returns.loc[quarter_start:quarter_end].values
        benchmark_log_returns_in_quarter = rsp_weekly_log_returns.loc[quarter_start:quarter_end].values
        num_stocks = log_returns_in_quarter.shape[1]

        print(f"\n{'='*20} Processing Quarter: {quarter} {'='*20}")
        if len(log_returns_in_quarter) < MIN_WEEKLY_OBSERVATIONS_PER_QUARTER:
            print(f"Skipping quarter {quarter} due to insufficient data."); continue

        print("Calculating shrunk covariance matrix for the quarter...")
        excess_returns = log_returns_in_quarter - benchmark_log_returns_in_quarter

        # Handle cases where excess returns have no variance
        if np.all(np.var(excess_returns, axis=0) < 1e-10):
            print("Warning: Zero variance in excess returns. Using zero covariance matrix.")
            quarterly_shrunk_cov_matrix = np.zeros((num_stocks, num_stocks))
        else:
            quarterly_shrunk_cov_matrix = LedoitWolf().fit(excess_returns).covariance_

        print("Shrunk covariance matrix calculated.")

        max_refinement_attempts = 10
        mu_val = 0.2; mu_increment = 0.4
        found_robust_solution = False
        best_fallback_solution = {'weights': np.zeros(num_stocks), 'ir': 0.0, 'ci': (-np.inf, -np.inf), 'bootstrap_samples': [], 'pso_costs_runs': [], 'pso_iter_costs': []}

        for attempt in range(max_refinement_attempts):
            print(f"\n--- Quarter {quarter} | Attempt {attempt + 1}/{max_refinement_attempts} | Risk Aversion (mu) = {mu_val:.2f} ---")

            try:
                print("Phase 1: Starting PSO global search...")
                pso_bounds = (np.full(num_stocks, -1.0), np.full(num_stocks, 1.0))
                pso_objective_func = create_pso_objective_function(log_returns_in_quarter, benchmark_log_returns_in_quarter, quarterly_shrunk_cov_matrix, num_stocks, mu=mu_val, diversification_penalty=diversification_penalty, previous_weights=previous_quarter_weights)
                _, pso_best_solution, pso_costs_runs, pso_iter_costs = multi_start_pso(
                    pso_objective_func, pso_bounds, set_particles, num_stocks, set_iters_pso, set_starts,
                    pso_w_decay_rate, pso_c1_decay_rate, pso_c2_increase_rate
                )
                all_pso_runs_iteration_costs.append(pso_iter_costs)
                all_pso_best_costs_runs.extend(pso_costs_runs)

                print("\nPhase 2: Starting SLSQP local refinement...")
                x0 = (pso_best_solution / np.sum(np.abs(pso_best_solution))) * 100.0 if np.sum(np.abs(pso_best_solution)) > 0 else np.zeros(num_stocks)
                slsqp_bounds = tuple([(0.0, 100.0) for _ in range(num_stocks)])
                constraints = ({'type': 'eq', 'fun': lambda w: np.sum(np.abs(w)) - 100.0})

                slsqp_result = minimize(scipy_objective_function, x0,
                                        args=(log_returns_in_quarter, benchmark_log_returns_in_quarter, quarterly_shrunk_cov_matrix, mu_val, diversification_penalty, previous_quarter_weights, num_stocks),
                                        method='SLSQP', bounds=slsqp_bounds, constraints=constraints,
                                        options={'disp': False, 'ftol': 1e-9, 'maxiter': 200})

                final_optimal_weights_unrounded = slsqp_result.x if slsqp_result.success else x0
                if not slsqp_result.success:
                    print(f"WARNING: SLSQP refinement did not converge. Using PSO result for this attempt.")

                print(f"\nPhase 3: Starting Robustness Check for Attempt {attempt + 1}...")
                bootstrap_ir_samples = perform_bootstrap_analysis_on_weights(log_returns_in_quarter, benchmark_log_returns_in_quarter, final_optimal_weights_unrounded, n_bootstrap_samples)
                ci_lower, ci_upper = np.percentile(bootstrap_ir_samples, 2.5), np.percentile(bootstrap_ir_samples, 97.5)

                gross_port_return = calculate_quarterly_return(log_returns_in_quarter, final_optimal_weights_unrounded)
                bench_return = np.exp(np.sum(benchmark_log_returns_in_quarter)) - 1
                track_error = calculate_shrunk_tracking_error(final_optimal_weights_unrounded, quarterly_shrunk_cov_matrix)
                information_ratio = (gross_port_return - bench_return) / track_error if track_error > 1e-8 else 0

                print(f"Attempt {attempt + 1} Result -> Gross IR: {information_ratio:.4f}, CI Lower: {ci_lower:.4f}, CI Upper: {ci_upper:.4f}")

                current_attempt_solution = {"weights": final_optimal_weights_unrounded, "ir": information_ratio, "ci": (ci_lower, ci_upper),
                                            "bootstrap_samples": bootstrap_ir_samples, "pso_costs_runs": pso_costs_runs, "pso_iter_costs": pso_iter_costs}

                if ci_lower > 0:
                    best_fallback_solution = current_attempt_solution
                    print(f"SUCCESS: Found a robust solution for {quarter}.")
                    found_robust_solution = True
                    break
                else:
                    if ci_lower > best_fallback_solution['ci'][0]:
                        print(f"INFO: Storing this attempt as the best fallback option so far (CI Lower: {ci_lower:.4f}).")
                        best_fallback_solution = current_attempt_solution
                    print(f"WARNING: Solution is not robust. Increasing risk aversion and retrying.")
                    mu_val += mu_increment
            except Exception as e:
                print(f"CRITICAL ERROR during optimization attempt for {quarter}: {repr(e)}"); break

        if not found_robust_solution:
            unsuccessful_quarters_count += 1
            print(f"\n{'!'*10} CRITICAL FAILURE: No robust solution found. Using best fallback solution with CI Lower {best_fallback_solution['ci'][0]:.4f}.")
        else:
            successful_quarters_count += 1
            print(f"\nFinal selected solution for {quarter} is robust.")

        final_weights_unrounded = best_fallback_solution['weights']
        final_ir = best_fallback_solution['ir']
        final_ci = best_fallback_solution['ci']

        print("Applying rounding to 2 decimal places and re-normalizing...")
        final_weights_rounded = np.round(final_weights_unrounded, 2)
        residual = np.sum(np.abs(final_weights_rounded)) - 100.0
        if abs(residual) > 1e-6 and np.sum(np.abs(final_weights_rounded)) > 0:
            largest_pos_idx = np.argmax(np.abs(final_weights_rounded))
            final_weights_rounded[largest_pos_idx] -= residual
        final_weights = final_weights_rounded

        valid_quarters.append(quarter)
        all_optimal_weights.append(final_weights)
        all_information_ratios.append(final_ir)
        bootstrap_results[quarter] = {"weights": final_weights, "IR_samples": None, "IR_CI": final_ci}
        all_bootstrap_irs_for_plot.extend(best_fallback_solution.get('bootstrap_samples', []))
        previous_quarter_weights = final_weights
        print(f"Final results for {quarter}: IR={final_ir:.4f}, 95% CI=({final_ci[0]:.4f}, {final_ci[1]:.4f})")

    # --- Plotting and Final DataFrame Assembly ---
    print("\n--- Generating Final Plots ---")
    if len(all_pso_runs_iteration_costs) > 0:
        flattened_iteration_costs = np.concatenate(all_pso_runs_iteration_costs, axis=0)
        mean_costs = np.mean(flattened_iteration_costs, axis=0)
        std_costs = np.std(flattened_iteration_costs, axis=0)
        plt.figure(figsize=(10, 6)); plt.plot(mean_costs, 'b-', label='Mean Cost'); plt.fill_between(range(len(mean_costs)), mean_costs - std_costs, mean_costs + std_costs, color='blue', alpha=0.2, label='Standard Deviation')
        plt.title('Average PSO Convergence Across All Attempts & Runs'); plt.xlabel('Iteration'); plt.ylabel('Cost'); plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig('Plots/APSO/pso_convergence.png')
        plt.close()


    if len(all_pso_best_costs_runs) > 0:
        plt.figure(figsize=(10, 6));
        data = np.array(all_pso_best_costs_runs)
        plt.hist(data, bins=30, density=True, alpha=0.7); plt.axvline(np.min(data), color='r', linestyle='dashed', label=f'Overall Best Cost: {np.min(data):.4f}')
        plt.title('Distribution of Best PSO Costs Across All Runs'); plt.xlabel('Cost'); plt.ylabel('Density'); plt.legend(); plt.grid(True, alpha=0.3);
        plt.tight_layout()
        plt.savefig('Plots/APSO/pso_costs_distribution.png')
        plt.close()


    if len(all_bootstrap_irs_for_plot) > 0:
        plt.figure(figsize=(10, 6)); irs = np.array(all_bootstrap_irs_for_plot); q1, q3 = np.percentile(irs, [25, 75]); iqr = q3 - q1
        lower_b, upper_b = q1 - 3 * iqr, q3 + 3 * iqr
        filtered_irs = irs[(irs >= lower_b) & (irs <= upper_b) & np.isfinite(irs)]
        if len(filtered_irs) > 0:
            skew_val = skew(filtered_irs); kurt_val = kurtosis(filtered_irs)
            plt.hist(filtered_irs, bins=50, density=True, alpha=0.7); plt.axvline(np.mean(filtered_irs), color='r', linestyle='dashed', label=f'Mean IR: {np.mean(filtered_irs):.4f}')
            plt.axvline(np.percentile(filtered_irs, 2.5), color='g', linestyle='dashed', label=f'2.5% CI'); plt.axvline(np.percentile(filtered_irs, 97.5), color='g', linestyle='dashed', label=f'97.5% CI')
            plt.title(f'Bootstrap IR Distribution (Skew: {skew_val:.2f}, Kurtosis: {kurt_val:.2f})'); plt.xlabel('Information Ratio'); plt.ylabel('Density'); plt.legend(); plt.grid(True, alpha=0.3);
            center_plot_around_zero([pd.Series(filtered_irs)])
            plt.tight_layout()
            plt.savefig('Plots/APSO/bootstrap_ir_distribution.png')
            plt.close()


    if valid_quarters:
        optimal_weights_df = pd.DataFrame(all_optimal_weights, index=valid_quarters, columns=stock_names_for_objective).rename(columns={c: f"{c}_Weight" for c in stock_names_for_objective})

        ir_df = pd.DataFrame(all_information_ratios, index=valid_quarters, columns=['Information_Ratio'])
        ci_df = pd.DataFrame([bootstrap_results[q]["IR_CI"] for q in valid_quarters], index=valid_quarters, columns=['IR_CI_Lower', 'IR_CI_Upper'])
        final_data_local = pd.concat([market_quarterly.loc[valid_quarters], optimal_weights_df, ir_df], axis=1)
        final_data_with_ci = pd.concat([final_data_local, ci_df], axis=1)

        final_data_local.index.name = 'Date'
        final_data_with_ci.index.name = 'Date'

        return final_data_with_ci, final_data_local, successful_quarters_count, unsuccessful_quarters_count

    return None, None, 0, 0

def create_summary_distribution_plots(final_data_with_ci_result):
    """Creates and saves a 2x2 grid of summary distribution plots."""
    successful_results = final_data_with_ci_result[final_data_with_ci_result['IR_CI_Lower'] > 0]
    unsuccessful_results = final_data_with_ci_result[final_data_with_ci_result['IR_CI_Lower'] <= 0]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Summary Distributions', fontsize=16)

    # Plot 1: Information Ratio (Successful)
    ax = axes[0, 0]
    if not successful_results.empty:
        s_data = successful_results['Information_Ratio']
        s_mean, s_median = s_data.mean(), s_data.median()
        ax.hist(s_data, bins=20, density=True, alpha=0.7, color='green', label='Successful')
        ax.axvline(s_mean, color='darkgreen', linestyle='--', label=f'Mean: {s_mean:.2f}')
        ax.axvline(s_median, color='darkgreen', linestyle=':', label=f'Median: {s_median:.2f}')
    ax.set_title('Information Ratios (Successful Quarters)')
    ax.legend()

    # Plot 2: Information Ratio (Unsuccessful)
    ax = axes[0, 1]
    if not unsuccessful_results.empty:
        u_data = unsuccessful_results['Information_Ratio']
        u_mean, u_median = u_data.mean(), u_data.median()
        ax.hist(u_data, bins=20, density=True, alpha=0.7, color='red', label='Unsuccessful')
        ax.axvline(u_mean, color='darkred', linestyle='--', label=f'Mean: {u_mean:.2f}')
        ax.axvline(u_median, color='darkred', linestyle=':', label=f'Median: {u_median:.2f}')
    ax.set_title('Information Ratios (Unsuccessful Quarters)')
    ax.legend()

    # Plot 3: Lower CI Bounds (All Quarters)
    ax = axes[1, 0]
    all_lower_ci = final_data_with_ci_result['IR_CI_Lower'].replace([np.inf, -np.inf], np.nan).dropna()
    if not all_lower_ci.empty:
        l_mean, l_median = all_lower_ci.mean(), all_lower_ci.median()
        ax.hist(all_lower_ci, bins=20, density=True, alpha=0.7, color='blue', label='All Quarters')
        ax.axvline(l_mean, color='darkblue', linestyle='--', label=f'Mean: {l_mean:.2f}')
        ax.axvline(l_median, color='darkblue', linestyle=':', label=f'Median: {l_median:.2f}')
    ax.set_title('Lower CI Bounds (All Quarters)')
    ax.legend()

    # Plot 4: Upper CI Bounds (All Quarters)
    ax = axes[1, 1]
    all_upper_ci = final_data_with_ci_result['IR_CI_Upper'].replace([np.inf, -np.inf], np.nan).dropna()
    if not all_upper_ci.empty:
        up_mean, up_median = all_upper_ci.mean(), all_upper_ci.median()
        ax.hist(all_upper_ci, bins=20, density=True, alpha=0.7, color='purple', label='All Quarters')
        ax.axvline(up_mean, color='indigo', linestyle='--', label=f'Mean: {up_mean:.2f}')
        ax.axvline(up_median, color='indigo', linestyle=':', label=f'Median: {up_median:.2f}')
    ax.set_title('Upper CI Bounds (All Quarters)')
    ax.legend()

    for ax_row in axes:
        for ax_item in ax_row:
            ax_item.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('Plots/APSO/summary_distributions.png')
    plt.close()


# --- Main Execution Block ---
if 'market_quarterly' not in globals() or 'prices_weekly' not in globals():
    print("Error: `market_quarterly` or `prices_weekly` not found in environment. Please load data first.")
else:
    final_data_with_ci_result, final_data_result, successes, failures = run_portfolio_optimization(
        market_quarterly=market_quarterly,
        prices_weekly=prices_weekly,
        n_bootstrap_samples=300,
        set_particles=200,
        set_iters_pso=50,
        set_starts=5,
        pso_w_decay_rate=3.0,
        pso_c1_decay_rate=3.0,
        pso_c2_increase_rate=1.0,
        diversification_penalty=0.5
    )

    if final_data_with_ci_result is not None:
        # --- Save Results to CSV ---
        print("\n" + "="*25 + " SAVING RESULTS TO CSV " + "="*25)
        try:
            final_data_result.to_csv("final_data_result.csv")
            print("Successfully saved results to 'final_data_result.csv'")
            final_data_with_ci_result.to_csv("final_data_with_ci_result.csv")
            print("Successfully saved detailed results to 'final_data_with_ci_result.csv'")
        except Exception as e:
            print(f"Error saving results to CSV: {e}")

        # --- Static Heatmap for Weights ---
        weight_columns = [col for col in final_data_with_ci_result.columns if col.endswith('_Weight')]
        weights_only_df = final_data_with_ci_result[weight_columns]

        print("\n" + "="*25 + " PORTFOLIO COMPOSITION HEATMAPS " + "="*25)

        plot_df = weights_only_df.T
        plot_df.index = plot_df.index.str.replace('_Weight', '')
        plot_df.columns = plot_df.columns.astype(str)

        # --- Heatmap 1: Custom 4-Tiered Colormap ---
        colors = ["#F1F8E9", "#A6D854", "#FFD92F", "#FC8D62", "#B30000"]
        nodes = [0.0, 0.25, 0.50, 0.75, 1.0]
        custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", list(zip(nodes, colors)))
        plt.figure(figsize=(20, 16))
        sns.heatmap(
            plot_df, annot=False, cmap=custom_cmap, vmin=0, vmax=100,
            linewidths=.5, cbar_kws={'label': 'Portfolio Weight (%)'}
        )
        plt.title('Portfolio Composition (4-Tiered Colorscale)', fontsize=16)
        plt.xlabel('Quarter', fontsize=12); plt.ylabel('Asset', fontsize=12)
        plt.xticks(rotation=45, ha='right'); plt.yticks(fontsize=8); plt.tight_layout()
        plt.savefig('Plots/APSO/heatmap_composition_custom.png')
        plt.close()

        # --- Heatmap 2: Standard Colorscale (Zeros as Grey) ---
        plot_df_nan = plot_df.replace(0, np.nan)
        viridis_with_bad = plt.get_cmap('viridis').copy()
        viridis_with_bad.set_bad(color='gainsboro')
        plt.figure(figsize=(20, 16))
        sns.heatmap(
            plot_df_nan, annot=False, cmap=viridis_with_bad, vmin=0, vmax=100,
            linewidths=.5, cbar_kws={'label': 'Portfolio Weight (%)'}
        )
        plt.title('Portfolio Composition (Standard Colorscale with Zeros Highlighted)', fontsize=16)
        plt.xlabel('Quarter', fontsize=12); plt.ylabel('Asset', fontsize=12)
        plt.xticks(rotation=45, ha='right'); plt.yticks(fontsize=8); plt.tight_layout()
        plt.savefig('Plots/APSO/heatmap_composition_standard.png')
        plt.close()

        # HHI Plot
        print("\n" + "="*25 + " PORTFOLIO CONCENTRATION (HHI) OVER TIME " + "="*25)
        weights_frac = weights_only_df / 100.0
        hhi = (weights_frac**2).sum(axis=1)
        num_positions = (weights_only_df > 0.01).sum(axis=1)
        num_positions[num_positions == 0] = 1
        eq_w_hhi = 1 / num_positions
        plt.figure(figsize=(12, 7))
        hhi.index = hhi.index.astype(str)
        hhi.plot(kind='line', marker='o', label='Portfolio HHI')
        eq_w_hhi.index = eq_w_hhi.index.astype(str)
        eq_w_hhi.plot(kind='line', marker='x', linestyle='--', color='red', label='Equal-Weight HHI (for context)')
        plt.title('Portfolio Concentration (HHI) Over Time', fontsize=16)
        plt.xlabel('Quarter', fontsize=12); plt.ylabel('HHI (Sum of Squared Weights)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6); plt.legend()
        plt.text(0.02, 0.02, f'Average HHI: {hhi.mean():.3f}', transform=plt.gca().transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
        plt.xticks(rotation=45, ha='right'); plt.tight_layout()
        plt.savefig('Plots/APSO/hhi_over_time.png')
        plt.close()

        # --- Summary Plots ---
        print("\n--- Generating Final Summary Plots ---")
        create_summary_distribution_plots(final_data_with_ci_result)

        # Text Summary
        print("\n" + "="*25 + " OPTIMIZATION RUN SUMMARY " + "="*25)
        total_processed = successes + failures
        if total_processed > 0:
            print(f"Total Quarters Processed:       {total_processed}")
            print(f"  - Successful Quarters:          {successes} ({successes/total_processed:.1%})")
            print(f"  - Unsuccessful (Fallback):    {failures} ({failures/total_processed:.1%})")
        else:
            print("No quarters were processed.")
        print("-" * 52)

        # Filter out non-finite values for mean calculation
        finite_ir = final_data_with_ci_result['Information_Ratio'][np.isfinite(final_data_with_ci_result['Information_Ratio'])]
        finite_ci_lower = final_data_with_ci_result['IR_CI_Lower'][np.isfinite(final_data_with_ci_result['IR_CI_Lower'])]
        finite_ci_upper = final_data_with_ci_result['IR_CI_Upper'][np.isfinite(final_data_with_ci_result['IR_CI_Upper'])]

        avg_ir = finite_ir.mean()
        median_ir = final_data_with_ci_result['Information_Ratio'].median()
        avg_lower_bound = finite_ci_lower.mean()
        median_lower_bound = final_data_with_ci_result['IR_CI_Lower'].median()
        avg_upper_bound = finite_ci_upper.mean()
        median_upper_bound = final_data_with_ci_result['IR_CI_Upper'].median()

        print(f"Average Information Ratio (All Quarters): {avg_ir:.4f}")
        print(f"Median Information Ratio (All Quarters):  {median_ir:.4f}")
        print("-" * 52)
        print(f"Average IR Lower Bound (All Quarters):  {avg_lower_bound:.4f}")
        print(f"Median IR Lower Bound (All Quarters):   {median_lower_bound:.4f}")
        print("-" * 52)
        print(f"Average IR Upper Bound (All Quarters):  {avg_upper_bound:.4f}")
        print(f"Median IR Upper Bound (All Quarters):   {median_upper_bound:.4f}")
        print("=" * 52)

    else:
        print("\nOptimization did not produce a final result.")
#%% Feature Engineering and Model Training Section
import pandas as pd
import numpy as np
import itertools
from scipy.spatial.distance import pdist, squareform
import warnings
from collections import Counter
import os  # <-- ADDED: To handle file paths

# Scikit-learn Imports
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler, PolynomialFeatures # <-- MODIFIED
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.utils import resample

# XGBoost and SHAP
import xgboost as xgb
import shap

# Plotly for Interactive 2D plotting
import plotly.graph_objects as go
import plotly.io as pio

# NetworkX for graph analysis
import networkx as nx

# Visualization Imports
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import seaborn as sns

# Using Optuna for hyperparameter tuning
import optuna
# Suppress informational messages from Optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)


# --- 0. Configuration & Setup ---
np.random.seed(42)
warnings.filterwarnings("ignore", category=UserWarning, message=".*pkg_resources is deprecated.*")

disable_plots = True
if disable_plots:
    print("--- PLOTTING IS DISABLED ---")

pio.renderers.default = "browser"

# --- 1. Global Data Handling ---
# --- MODIFIED: Load and export files from/to a 'Data' folder in the working directory ---
DATA_DIR = os.path.join(os.getcwd(), 'Data')
CSV_FILE_PATH = os.path.join(DATA_DIR, 'final_data_result.csv')
# --- End of Modification ---

try:
    final_data_result = pd.read_csv(CSV_FILE_PATH, index_col=0)
    if not isinstance(final_data_result.index, pd.DatetimeIndex):
        final_data_result.index = pd.to_datetime(final_data_result.index)
    final_data_result.index = final_data_result.index.to_period('Q')
    print(f"GLOBAL: Successfully loaded and processed data from '{CSV_FILE_PATH}'.")
except Exception as e:
    print(f"GLOBAL ERROR: Could not load data from '{CSV_FILE_PATH}': {e}")
    exit("Data not available. Exiting.")

protected_weight_features = [col for col in final_data_result.columns if col.endswith('_Weight')]
print(f"Identified {len(protected_weight_features)} protected '_Weight' features.")

# --- 2. Data Preparation and Feature Separation ---
def prepare_data_for_ml(df_input):
    print("\n--- 2. Preparing Data ---")
    TARGET_VARIABLE = 'Information_Ratio'
    if TARGET_VARIABLE not in df_input.columns:
        raise ValueError(f"Target column '{TARGET_VARIABLE}' not found.")
    df_input = df_input.dropna(subset=[TARGET_VARIABLE])
    target_df = df_input[[TARGET_VARIABLE]]
    weight_features_df = df_input[protected_weight_features]
    engineered_feature_names = [col for col in df_input.columns if col != TARGET_VARIABLE and col not in protected_weight_features]
    engineered_features_df = df_input[engineered_feature_names]
    print(f"Separated target variable. {len(engineered_features_df.columns)} features will be engineered.")
    return target_df, engineered_features_df, weight_features_df, TARGET_VARIABLE

# --- 3. Time-Series Feature Generation ---
def generate_ts_features(features_df, target_series):
    all_new_features = []
    for col in features_df.columns:
        for lag in range(1, 5):
            all_new_features.append(features_df[col].shift(lag).rename(f'{col}_lag_{lag}'))
    for lag in range(1, 3):
        all_new_features.append(target_series.shift(lag).rename(f'{target_series.name}_lag_{lag}'))
    for col in features_df.columns:
        for window in [2, 4]:
            all_new_features.append(features_df[col].rolling(window=window).mean().rename(f'{col}_rolling_mean_{window}q'))
            all_new_features.append(features_df[col].rolling(window=window).std().rename(f'{col}_rolling_std_{window}q'))
    time_features_df = pd.DataFrame(index=features_df.index)
    time_features_df['quarter_of_year'] = features_df.index.quarter
    time_features_df['year'] = features_df.index.year
    time_features_df['quarter_sin'] = np.sin(2 * np.pi * time_features_df['quarter_of_year'] / 4)
    time_features_df['quarter_cos'] = np.cos(2 * np.pi * time_features_df['quarter_of_year'] / 4)
    all_new_features.append(time_features_df)
    ts_features = pd.concat(all_new_features, axis=1).copy()
    ts_features.replace([np.inf, -np.inf], np.nan, inplace=True)
    ts_features.fillna(0, inplace=True)
    return ts_features

# --- Main Execution Script ---
target_df, engineered_features, weight_features, TARGET_VARIABLE = prepare_data_for_ml(final_data_result)

# --- 3. Train/Validation/Test Split ---
print("\n--- Data Split ---")
train_size = int(0.70 * len(target_df))
val_size = int(0.15 * len(target_df))
y_train, y_valid, y_test = target_df.iloc[:train_size], target_df.iloc[train_size:train_size + val_size], target_df.iloc[train_size + val_size:]
X_train_engineered_raw, X_valid_engineered_raw, X_test_engineered_raw = engineered_features.iloc[:train_size], engineered_features.iloc[train_size:train_size + val_size], engineered_features.iloc[train_size + val_size:]
X_train_weights, X_valid_weights, X_test_weights = weight_features.iloc[:train_size], weight_features.iloc[train_size:train_size + val_size], weight_features.iloc[train_size + val_size:]
print(f"Train: {len(y_train)} rows\nValidation: {len(y_valid)} rows\nTest: {len(y_test)} rows")

# --- 4. Generate Time-Series Features ---
print("\n--- 4. Generating Time-Series Features ---")
X_train_ts_engineered = generate_ts_features(X_train_engineered_raw, y_train[TARGET_VARIABLE])
X_train_ts = pd.concat([X_train_ts_engineered, X_train_weights], axis=1)
print(f"Generated {X_train_ts_engineered.shape[1]} time-series features.")

# --- 5. Dynamic MI Pre-selection ---
print("\n--- 5. Dynamic MI Pre-selection ---")
mi_scores = pd.Series(mutual_info_regression(X_train_ts, y_train.values.ravel(), random_state=42), index=X_train_ts.columns)
mi_selected_engineered, avg_distances, combined_scores, threshold_combined = [], np.array([]), pd.Series(dtype=float), np.inf
if not mi_scores.empty and mi_scores.sum() > 0:
    mi_scores.sort_values(ascending=False, inplace=True)
    distance_matrix = squareform(pdist(mi_scores.values.reshape(-1, 1), 'euclidean'))
    avg_distances = np.mean(distance_matrix, axis=1)
    avg_distances_series = pd.Series(avg_distances, index=mi_scores.index)
    combined_scores = mi_scores * avg_distances_series
    combined_scores.sort_values(ascending=False, inplace=True)
    if not combined_scores.empty:
        mean_combined, std_combined = combined_scores.mean(), combined_scores.std()
        threshold_combined = mean_combined + (3 * std_combined)
        candidate_features = combined_scores[combined_scores > threshold_combined].index.tolist()
        mi_selected_engineered = [f for f in candidate_features if f not in protected_weight_features]
if not mi_selected_engineered:
    engineered_mi_scores_fallback = mi_scores.drop(labels=protected_weight_features, errors='ignore')
    mi_selected_engineered = engineered_mi_scores_fallback.nlargest(50).index.tolist()
final_mi_features = mi_selected_engineered + protected_weight_features
# --- NEW: Print statement added ---
print(f"Selected {len(final_mi_features)} features after MI step ({len(mi_selected_engineered)} engineered + {len(protected_weight_features)} protected).")
X_train_mi = X_train_ts[final_mi_features]


# --- 5.5 PLOT 1: Enhanced 2D Scatter for MI with Buttons ---
if not disable_plots:
    print("\n--- 5.5 Generating Plot 1: 2D Scatter for MI ---")
    G_mi = nx.Graph()
    for feat_name in mi_scores.index: G_mi.add_node(feat_name, mi_score=mi_scores.get(feat_name, 0), avg_dist=avg_distances_series.get(feat_name, 0), combined_score=combined_scores.get(feat_name, 0))
    pos_mi = nx.spring_layout(G_mi, dim=2, seed=42)
    global_combined_score_min = combined_scores.min()
    global_combined_score_max = combined_scores.max()
    trace_all_nodes_mi = go.Scatter(x=[pos_mi[n][0] for n in G_mi.nodes()],y=[pos_mi[n][1] for n in G_mi.nodes()],mode='markers',hoverinfo='text',text=[f"{n}\nMI Score: {d['mi_score']:.4f}\nAvg MI Distance: {d['avg_dist']:.4f}\nCombined Score: {d['combined_score']:.4f}" for n, d in G_mi.nodes(data=True)],marker=dict(showscale=True,colorscale='Viridis',color=[d['combined_score'] for n, d in G_mi.nodes(data=True)],size=[d['mi_score'] * 300 + 8 for n, d in G_mi.nodes(data=True)],cmin=global_combined_score_min,cmax=global_combined_score_max,colorbar=dict(thickness=15, title=dict(text='Combined Score', side='right'), tickvals=[threshold_combined], ticktext=[f'Threshold ({threshold_combined:.2f})'])))
    trace_selected_nodes_mi = go.Scatter(x=[pos_mi[n][0] for n in final_mi_features if n in pos_mi],y=[pos_mi[n][1] for n in final_mi_features if n in pos_mi],mode='markers',hoverinfo='text',visible=False,text=[f"{n}\nMI Score: {d['mi_score']:.4f}\nAvg MI Distance: {d['avg_dist']:.4f}\nCombined Score: {d['combined_score']:.4f}" for n,d in G_mi.nodes(data=True) if n in final_mi_features],marker=dict(showscale=True,colorscale='Viridis',color=[d['combined_score'] for n,d in G_mi.nodes(data=True) if n in final_mi_features],size=[d['mi_score']*300 + 8 for n,d in G_mi.nodes(data=True) if n in final_mi_features],cmin=global_combined_score_min,cmax=global_combined_score_max,colorbar=dict(thickness=15, title=dict(text='Combined Score'), tickvals=[threshold_combined], ticktext=[f'Threshold ({threshold_combined:.2f})'])))
    fig_mi_scatter = go.Figure(data=[trace_all_nodes_mi, trace_selected_nodes_mi])
    fig_mi_scatter.update_layout(title='2D MI Analysis\n(Size = MI Score | Color = Combined Score)', updatemenus=[dict(type="buttons", direction="right", x=0.5, y=1.1, xanchor='center', yanchor='top', buttons=list([dict(label="All Features", method="update", args=[{"visible": [True, False]}]), dict(label="Selected Features", method="update", args=[{"visible": [False, True]}])]))])
    fig_mi_scatter.show()

# --- 6. Polynomial Feature Generation ---
print("\n--- 6. Generating Polynomial Features ---")
X_train_mi_engineered = X_train_mi.drop(columns=protected_weight_features)
X_train_mi_weights = X_train_mi[protected_weight_features]
poly_scaler = RobustScaler().fit(X_train_mi_engineered) # <-- MODIFIED
poly = PolynomialFeatures(degree=2, include_bias=False).fit(poly_scaler.transform(X_train_mi_engineered))
X_train_poly_engineered = pd.DataFrame(poly.transform(poly_scaler.transform(X_train_mi_engineered)), columns=poly.get_feature_names_out(X_train_mi_engineered.columns), index=X_train_mi_engineered.index)
X_train_poly = pd.concat([X_train_poly_engineered, X_train_mi_weights], axis=1)
print(f"Expanded to {X_train_poly_engineered.shape[1]} polynomial features.")

# --- 7. SULOV Selection ---
print("\n--- 7. SULOV Selection ---")
corr_matrix = X_train_poly.corr().abs()
mis_scores_sulov = pd.Series(mutual_info_regression(X_train_poly, y_train.values.ravel(), random_state=42), index=X_train_poly.columns)
correlated_pairs_for_plot = [(corr_matrix.columns[i], corr_matrix.columns[j]) for i in range(len(corr_matrix.columns)) for j in range(i + 1, len(corr_matrix.columns)) if corr_matrix.iloc[i, j] > 0.7]
features_to_remove_candidates = {c1 if mis_scores_sulov.get(c1, 0) < mis_scores_sulov.get(c2, 0) else c2 for c1, c2 in correlated_pairs_for_plot}
features_to_remove = {feat for feat in features_to_remove_candidates if feat not in protected_weight_features}
sulov_selected_features = [feat for feat in X_train_poly.columns if feat not in features_to_remove]
X_train_sulov = X_train_poly[sulov_selected_features]

# --- 7.5 PLOT 2: Enhanced 2D Scatter for SULOV with Buttons ---
if not disable_plots:
    print("\n--- 7.5 Generating Plot 2: 2D Scatter for SULOV Correlations ---")
    correlated_pairs_df = pd.DataFrame(correlated_pairs_for_plot, columns=['source', 'target'])
    correlated_pairs_df['weight'] = correlated_pairs_df.apply(lambda row: corr_matrix.loc[row['source'], row['target']], axis=1)
    G_sulov = nx.from_pandas_edgelist(correlated_pairs_df, 'source', 'target', ['weight'])
    if G_sulov.nodes():
        node_degrees = {n:d for n,d in G_sulov.degree()}; nx.set_node_attributes(G_sulov, node_degrees, 'degree'); nx.set_node_attributes(G_sulov, {n: mis_scores_sulov.get(n, 0) for n in G_sulov.nodes()}, 'mi_score')
        pos_sulov = nx.spring_layout(G_sulov, dim=2, seed=42)
        all_degrees = [d['degree'] for n, d in G_sulov.nodes(data=True)]
        global_degree_min = min(all_degrees) if all_degrees else 0
        global_degree_max = max(all_degrees) if all_degrees else 0
        edge_x_all, edge_y_all = [],[];
        for edge in G_sulov.edges(): x0, y0 = pos_sulov[edge[0]]; x1, y1 = pos_sulov[edge[1]]; edge_x_all.extend([x0,x1,None]); edge_y_all.extend([y0,y1,None])
        edge_trace_all = go.Scatter(x=edge_x_all, y=edge_y_all, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')
        node_trace_all_s = go.Scatter(x=[pos_sulov[n][0] for n in G_sulov.nodes()],y=[pos_sulov[n][1] for n in G_sulov.nodes()],mode='markers',hoverinfo='text',text=[f"{n}<br>MI: {d.get('mi_score', 0):.4f}<br>Connections: {d.get('degree', 0)}" for n,d in G_sulov.nodes(data=True)],marker=dict(showscale=True,colorscale='Plasma',color=[d.get('degree', 0) for n,d in G_sulov.nodes(data=True)],size=[d.get('mi_score', 0)*200+8 for n,d in G_sulov.nodes(data=True)],cmin=global_degree_min,cmax=global_degree_max,colorbar=dict(thickness=15, title=dict(text='Connections', side='right'))))
        G_sulov_sel = G_sulov.subgraph(sulov_selected_features)
        edge_x_sel, edge_y_sel = [], []
        for edge in G_sulov_sel.edges(): x0, y0 = pos_sulov[edge[0]]; x1, y1 = pos_sulov[edge[1]]; edge_x_sel.extend([x0,x1,None]); edge_y_sel.extend([y0,y1,None])
        edge_trace_sel = go.Scatter(x=edge_x_sel, y=edge_y_sel, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines', visible=False)
        node_trace_sel_s = go.Scatter(x=[pos_sulov[n][0] for n in G_sulov_sel.nodes()],y=[pos_sulov[n][1] for n in G_sulov_sel.nodes()],mode='markers',hoverinfo='text',visible=False,text=[f"{n}<br>MI: {d.get('mi_score', 0):.4f}<br>Connections: {d.get('degree', 0)}" for n,d in G_sulov_sel.nodes(data=True)],marker=dict(showscale=True,colorscale='Plasma',color=[d.get('degree', 0) for n,d in G_sulov_sel.nodes(data=True)],size=[d.get('mi_score', 0)*200+8 for n,d in G_sulov_sel.nodes(data=True)],cmin=global_degree_min,cmax=global_degree_max,colorbar=dict(thickness=15, title=dict(text='Connections', side='right'))))
        fig_sulov_scatter = go.Figure(data=[edge_trace_all, node_trace_all_s, edge_trace_sel, node_trace_sel_s])
        fig_sulov_scatter.update_layout(title='2D SULOV Analysis<br>(Size = MI Score | Color = # Connections)', showlegend=False, updatemenus=[dict(type="buttons", direction="right", x=0.5, y=1.1, xanchor='center', yanchor='top', buttons=list([dict(label="All Correlated Features", method="update", args=[{"visible": [True, True, False, False]}]), dict(label="Kept Features", method="update", args=[{"visible": [False, False, True, True]}])]))])
        fig_sulov_scatter.show()

# --- 8. PRE-RFE: Transform Validation Set ---
print("\n--- 8. Transforming Validation set for RFE ---")
X_valid_ts_engineered = generate_ts_features(X_valid_engineered_raw, y_valid[TARGET_VARIABLE])
X_valid_ts = pd.concat([X_valid_ts_engineered, X_valid_weights], axis=1)
X_valid_mi = X_valid_ts.reindex(columns=final_mi_features, fill_value=0)
X_valid_mi_engineered = X_valid_mi.drop(columns=protected_weight_features)
X_valid_mi_weights = X_valid_mi[protected_weight_features]
X_valid_poly_engineered = pd.DataFrame(poly.transform(poly_scaler.transform(X_valid_mi_engineered)), columns=poly.get_feature_names_out(X_valid_mi_engineered.columns), index=X_valid_mi_engineered.index)
X_valid_poly = pd.concat([X_valid_poly_engineered, X_valid_mi_weights], axis=1)
X_valid_sulov = X_valid_poly.reindex(columns=sulov_selected_features, fill_value=0)

# --- 9. Multi-Round SHAP-RFE with Optuna Tuning ---
print("\n--- 9. Hyperparameter Tuning with Optuna ---")
def objective(trial):
    pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "validation_0-rmse")
    params = {
        'objective': 'reg:squarederror', 'eval_metric': 'rmse',
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
        'random_state': 42,
        'callbacks': [pruning_callback]
    }
    model = xgb.XGBRegressor(**params)
    model.fit(X_train_sulov, y_train, eval_set=[(X_valid_sulov, y_valid)], verbose=False)
    preds = model.predict(X_valid_sulov)
    mse = mean_squared_error(y_valid, preds)
    return mse

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50, show_progress_bar=True)
best_params = study.best_params
print(f"Best parameters for RFE model found by Optuna: {best_params}")

N_ROUNDS = 10
N_ITERATIONS_PER_ROUND = 5
N_FEATURES_PER_ITERATION = 3
all_round_features = []
rfe_performance_log = []
rfe_engineered_features = [f for f in X_train_sulov.columns if f not in protected_weight_features]

print(f"\n--- Starting SHAP-RFE with Tuned Parameters ---")
for i in range(N_ROUNDS):
    print(f"\n--- Round {i+1}/{N_ROUNDS} ---")
    X_train_boot, y_train_boot = resample(X_train_sulov, y_train, random_state=i)
    features_in_play = rfe_engineered_features.copy()
    selected_this_round = []
    for j in range(N_ITERATIONS_PER_ROUND):
        print(f"   Iteration {j+1}/{N_ITERATIONS_PER_ROUND}: Finding top {N_FEATURES_PER_ITERATION} from {len(features_in_play)} candidates.")
        if not features_in_play:
            print("   No more engineered features to select from.")
            break
        current_training_features = features_in_play + protected_weight_features
        temp_model = xgb.XGBRegressor(random_state=42, **best_params)
        temp_model.fit(X_train_boot[current_training_features], y_train_boot[TARGET_VARIABLE],
                       eval_set=[(X_valid_sulov[current_training_features], y_valid[TARGET_VARIABLE])],
                       verbose=False)
        if i == 0:
            y_pred_valid = temp_model.predict(X_valid_sulov[current_training_features])
            mse = mean_squared_error(y_valid, y_pred_valid)
            rfe_performance_log.append({'features_remaining': len(current_training_features), 'validation_mse': mse})
        explainer = shap.TreeExplainer(temp_model)
        shap_values = explainer.shap_values(X_train_boot[current_training_features])
        shap_importance = pd.Series(np.abs(shap_values).mean(axis=0), index=current_training_features)
        shap_importance_engineered = shap_importance.loc[features_in_play].sort_values(ascending=False)
        top_n_features = shap_importance_engineered.head(N_FEATURES_PER_ITERATION).index.tolist()
        selected_this_round.extend(top_n_features)
        features_in_play = [f for f in features_in_play if f not in top_n_features]
    round_features = list(dict.fromkeys(selected_this_round))
    all_round_features.append(round_features)
    print(f"Round {i+1} selected {len(round_features)} unique engineered features.")

if not all_round_features: raise ValueError("Feature selection process resulted in zero features.")
stable_engineered_features = sorted(list(set(all_round_features[0]).intersection(*all_round_features[1:])))
final_selected_features = stable_engineered_features + protected_weight_features
print(f"\nFound {len(stable_engineered_features)} stable engineered features. Total final features: {len(final_selected_features)}")

# --- 10. Final Combination and Evaluation ---
print("\n--- 10. Final Combination and Evaluation ---")
X_train_final = X_train_sulov[final_selected_features]
X_valid_final = X_valid_sulov[final_selected_features]
X_test_ts_engineered = generate_ts_features(X_test_engineered_raw, y_test[TARGET_VARIABLE])
X_test_ts = pd.concat([X_test_ts_engineered, X_test_weights], axis=1)
X_test_mi = X_test_ts.reindex(columns=final_mi_features, fill_value=0)
X_test_mi_engineered = X_test_mi.drop(columns=protected_weight_features)
X_test_mi_weights = X_test_mi[protected_weight_features]
X_test_poly_engineered = pd.DataFrame(poly.transform(poly_scaler.transform(X_test_mi_engineered)), columns=poly.get_feature_names_out(X_test_mi_engineered.columns), index=X_test_mi_engineered.index)
X_test_poly = pd.concat([X_test_poly_engineered, X_test_mi_weights], axis=1)
X_test_sulov = X_test_poly.reindex(columns=sulov_selected_features, fill_value=0)
X_test_final = X_test_sulov[final_selected_features]
X_train_val_combined = pd.concat([X_train_final, X_valid_final])
y_train_val_combined = pd.concat([y_train, y_valid])
final_model = xgb.XGBRegressor(random_state=42, **best_params)
eval_set = [(X_train_val_combined, y_train_val_combined), (X_test_final, y_test)]
final_model.fit(X_train_val_combined, y_train_val_combined.values.ravel(), eval_set=eval_set, verbose=False)
y_pred_test = final_model.predict(X_test_final)
test_mse = mean_squared_error(y_test, y_pred_test)
test_rmse = np.sqrt(test_mse)
test_mae = mean_absolute_error(y_test, y_pred_test)

# Calculate MAPE, handling potential division by zero
y_true_for_mape = y_test[TARGET_VARIABLE]
mask = y_true_for_mape != 0
if np.any(mask):
    # Note: y_pred_test is a numpy array, so boolean indexing works directly
    test_mape = np.mean(np.abs((y_true_for_mape[mask] - y_pred_test[mask]) / y_true_for_mape[mask])) * 100
else:
    test_mape = np.inf # Set to infinity if all true values are zero

directional_accuracy = np.mean(np.sign(y_test[TARGET_VARIABLE]) == np.sign(y_pred_test)) * 100

print("\n" + "="*50); print("       FEATURE ENGINEERING & SELECTION COMPLETE"); print("="*50)
print(f"\n--- Final Model Performance on Test Set ---")
print(f"  Mean Squared Error (MSE):          {test_mse:.4f}")
print(f"  Root Mean Squared Error (RMSE):    {test_rmse:.4f}")
print(f"  Mean Absolute Error (MAE):         {test_mae:.4f}")
print(f"  Mean Absolute Percentage Error (MAPE): {test_mape:.2f}%")
print(f"  Directional Accuracy:              {directional_accuracy:.2f}%")

print("\n--- Final List of Stable Engineered and Protected Features ---")
for i, feature in enumerate(final_selected_features):
    marker = "[PROTECTED]" if feature in protected_weight_features else ""
    print(f"{i+1:2d}: {feature} {marker}")
print("\n--- End of Pipeline ---")

# --- 11. RFE Evaluation Plots ---
if not disable_plots:
    print("\n--- 11. Generating RFE Evaluation Plots ---")
    plt.figure(figsize=(12, 10))
    all_selected_features_union = sorted(list(set(itertools.chain.from_iterable(all_round_features))))
    if all_selected_features_union:
        heatmap_data = pd.DataFrame(0, index=all_selected_features_union, columns=[f'Round {i+1}' for i in range(N_ROUNDS)])
        for i, round_features in enumerate(all_round_features): heatmap_data.loc[round_features, f'Round {i+1}'] = 1
        cumulative_heatmap_data = heatmap_data.cumsum(axis=1)
        sns.heatmap(cumulative_heatmap_data, cmap='viridis', annot=False, cbar=True, cbar_kws={'label': 'Cumulative Number of Selections'})
        plt.title('Cumulative Engineered Feature Stability Heatmap', fontsize=16)
        plt.xlabel('Selection Round')
        plt.ylabel('Engineered Feature')
        plt.tight_layout()
        plt.show()

    plt.figure(figsize=(12, 10))
    feature_counts = Counter(itertools.chain.from_iterable(all_round_features))
    if feature_counts:
        feature_freq_df = pd.DataFrame(feature_counts.items(), columns=['Feature', 'Frequency']).sort_values(by='Frequency', ascending=True)
        norm = colors.Normalize(vmin=feature_freq_df['Frequency'].min(), vmax=feature_freq_df['Frequency'].max())
        cmap = cm.get_cmap('viridis')
        dot_colors = cmap(norm(feature_freq_df['Frequency']))
        feature_freq_df.reset_index(drop=True, inplace=True)
        plt.hlines(y=feature_freq_df.index, xmin=0, xmax=feature_freq_df['Frequency'], color=dot_colors, alpha=0.6, linewidth=2)
        plt.scatter(x=feature_freq_df['Frequency'], y=feature_freq_df.index, color=dot_colors, s=100, alpha=1, zorder=3)
        plt.yticks(feature_freq_df.index, feature_freq_df['Feature'])
        for i, row in feature_freq_df.iterrows(): plt.text(row['Frequency'] + 0.1, i, int(row['Frequency']), va='center', ha='left', fontsize=10)
        plt.title('Engineered Feature Selection Frequency', fontsize=16)
        plt.xlabel('Number of Rounds Selected')
        plt.ylabel('Engineered Feature')
        plt.xlim(0, N_ROUNDS + 1)
        plt.grid(axis='x', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

    if 'rfe_performance_log' in locals() and rfe_performance_log:
        plt.figure(figsize=(10, 6))
        perf_df = pd.DataFrame(rfe_performance_log)
        sns.lineplot(x='features_remaining', y='validation_mse', data=perf_df, marker='o')
        plt.title('RFE Performance Curve (Round 1)', fontsize=16)
        plt.xlabel('Number of Features Remaining (Engineered + Protected)')
        plt.ylabel('Validation MSE')
        plt.grid(True, linestyle='--')
        plt.gca().invert_xaxis()
        plt.show()

    if final_selected_features:
        print("\nGenerating SHAP summary plot for the final model...")
        explainer_final = shap.TreeExplainer(final_model)
        shap_values_final = explainer_final.shap_values(X_train_val_combined)
        shap.summary_plot(shap_values_final, X_train_val_combined, plot_size=(12, max(6, len(final_selected_features)*0.5)), show=False)
        plt.title("SHAP Summary for Final Model", fontsize=16)
        plt.tight_layout()
        plt.show()

    print("\nGenerating Residual Plot...")
    y_pred_series = pd.Series(y_pred_test, index=y_test.index)
    residuals = y_test.squeeze() - y_pred_series
    sns.scatterplot(x=y_pred_series, y=residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals (Actual - Predicted)")
    plt.title("Residual Plot of Final Model", fontsize=16)
    plt.grid(True)
    plt.show()

    print("\nGenerating RMSE by Epoch Plot...")
    results = final_model.evals_result()
    train_rmse = results['validation_0']['rmse']
    test_rmse = results['validation_1']['rmse']
    epochs = len(train_rmse)
    x_axis = range(0, epochs)
    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, train_rmse, label='Train RMSE')
    plt.plot(x_axis, test_rmse, label='Test RMSE')
    plt.legend()
    plt.ylabel('RMSE')
    plt.xlabel('Epoch')
    plt.title('Final Model RMSE by Epoch', fontsize=16)
    plt.grid(True)
    plt.show()

# --- 12. Export Final Features ---
print("\n--- 12. Exporting Final Features ---")
if final_selected_features:
    final_features_full_dataset = pd.concat([X_train_final, X_valid_final, X_test_final])
    # --- MODIFIED: Ensure the output directory exists and define export path ---
    os.makedirs(DATA_DIR, exist_ok=True)
    EXPORT_FILE_PATH = os.path.join(DATA_DIR, 'final_model_features.csv')
    # --- End of Modification ---
    final_features_full_dataset.to_csv(EXPORT_FILE_PATH)
    print(f"\nSuccessfully exported {len(final_features_full_dataset)} rows of {len(final_selected_features)} final features to '{EXPORT_FILE_PATH}'")
else:
    print("\nNo final features were selected, so no CSV file was exported.")
#%% Forecast
import pandas as pd
import numpy as np
import optuna
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import warnings

# --- 0. Configuration & Setup ---
np.random.seed(42)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)
print("Starting Integrated GBRT Pipeline with Full Evaluation and Plotting...")

# --- Stage 1: Load Data and Features ---
print("\n--- Stage 1: Loading All Necessary Data ---")
try:
    features_df = pd.read_csv('final_model_features.csv')
    all_model_features = features_df.columns.tolist()
except FileNotFoundError:
    print("ERROR: 'final_model_features.csv' not found.")
    exit()
weight_cols = [col for col in all_model_features if col.endswith('_Weight')]
non_weight_features = [col for col in all_model_features if not col.endswith('_Weight')]
print(f"Identified {len(weight_cols)} weight features and {len(non_weight_features)} fixed features.")
try:
    df_full = pd.read_csv('final_data_result.csv', index_col='Date', parse_dates=True)
except FileNotFoundError:
    print("ERROR: 'final_data_result.csv' not found.")
    exit()

# --- Stage 2: Create Time-Series Splits (Train/Validation) ---
print("\n--- Stage 2: Splitting Data for Training and Validation ---")
TARGET = 'Information_Ratio'
df_history = df_full.iloc[:-1].copy()
df_predict_point = df_full.iloc[-1:].copy()
X = df_history.reindex(columns=all_model_features, fill_value=0)
y = df_history[TARGET]
train_size = int(0.70 * len(X))
X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size:], y[train_size:]
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# --- Stage 3: Optuna Hyperparameter Search for GBRT ---
print("\n--- Stage 3: Finding Best GBRT Hyperparameters with Optuna ---")
def objective(trial):
    params = {
        'n_estimators': 1000, 'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 2, 5), 'subsample': trial.suggest_float('subsample', 0.6, 1.0, step=0.05),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 20), 'random_state': 42,
    }
    model = GradientBoostingRegressor(**params)
    model.fit(X_train_scaled, y_train)
    val_errors = [mean_squared_error(y_val, y_pred) for y_pred in model.staged_predict(X_val_scaled)]
    best_iteration = np.argmin(val_errors) + 1
    min_val_rmse = np.sqrt(np.min(val_errors))
    trial.set_user_attr('best_iteration', best_iteration)
    return min_val_rmse
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50, show_progress_bar=False)
best_params = study.best_params
best_params['n_estimators'] = study.best_trial.user_attrs['best_iteration']

# --- Stage 4: Evaluate Model Performance on Validation Set ---
print("\n--- Stage 4: Evaluating Model Performance on Unseen Validation Data ---")
eval_model = GradientBoostingRegressor(**best_params, random_state=42)
eval_model.fit(X_train_scaled, y_train)
val_predictions = eval_model.predict(X_val_scaled)
mse = mean_squared_error(y_val, val_predictions)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_val, val_predictions)
epsilon = 1e-8
mape = np.mean(np.abs((y_val - val_predictions) / (y_val + epsilon))) * 100
directional_accuracy = np.mean(np.sign(val_predictions) == np.sign(y_val)) * 100
print(f"  - Mean Squared Error (MSE):      {mse:.4f}")
print(f"  - Root Mean Squared Error (RMSE):  {rmse:.4f}")
print(f"  - Mean Absolute Error (MAE):     {mae:.4f}")
print(f"  - Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
print(f"  - Directional Accuracy:          {directional_accuracy:.2f}%")

# --- Stage 5: Train Final Model and Create Importance Prior ---
print("\n--- Stage 5: Training Final Model and Creating Importance Prior ---")
X_full_train_scaled = np.vstack((X_train_scaled, X_val_scaled))
y_full_train = pd.concat([y_train, y_val])
final_model = GradientBoostingRegressor(**best_params, random_state=42)
final_model.fit(X_full_train_scaled, y_full_train)
print("Final GBRT model trained on all historical data.")
feature_importances = pd.Series(final_model.feature_importances_, index=X.columns)
weight_importances = feature_importances.loc[weight_cols]
importance_based_weights = weight_importances / (weight_importances.sum() + epsilon)

# --- Stage 6: Optimize Portfolio Weights with Prior ---
print("\n--- Stage 6: Optimizing Portfolio Weights with Prior ---")
fixed_features_for_prediction = df_predict_point.reindex(columns=non_weight_features, fill_value=0)
def objective_optimization(new_weights, model, scaler, fixed_features, weight_columns, all_feature_columns, lambda_hhi, target_weights, lambda_prior):
    weights_df = pd.DataFrame([new_weights], columns=weight_columns, index=fixed_features.index)
    prediction_row = pd.concat([weights_df, fixed_features], axis=1)[all_feature_columns]
    prediction_row_scaled = scaler.transform(prediction_row)
    predicted_ir = model.predict(prediction_row_scaled)[0]
    num_assets = len(new_weights)
    hhi = np.sum(new_weights**2)
    normalized_hhi = (hhi - 1/num_assets) / (1 - 1/num_assets) if num_assets > 1 else 1.0
    prior_penalty = np.sum((new_weights - target_weights.values)**2)
    return -predicted_ir + (lambda_hhi * normalized_hhi) + (lambda_prior * prior_penalty)
DIVERSIFICATION_STRENGTH = 0.1
PRIOR_STRENGTH = 0.5
initial_guess = importance_based_weights.values
constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
bounds = tuple((0, 1) for _ in weight_cols)
print("Searching for optimal weights...")
optimization_result = minimize(
    fun=objective_optimization, x0=initial_guess,
    args=(final_model, scaler, fixed_features_for_prediction, weight_cols, X.columns, DIVERSIFICATION_STRENGTH, importance_based_weights, PRIOR_STRENGTH),
    method='SLSQP', bounds=bounds, constraints=constraints
)

# --- Stage 7: Visualize & Display Final Portfolio ---
if optimization_result.success:
    print("\n--- Stage 7: Generating Final Results ---")
    optimal_weights = optimization_result.x
    asset_names = [col.replace('_Weight', '') for col in weight_cols]

    # --- Reallocation Logic ---
    weights_series = pd.Series(optimal_weights, index=asset_names)
    threshold = 0.01
    small_weights = weights_series[weights_series < threshold]
    total_to_reallocate = small_weights.sum()
    weights_to_keep = weights_series[weights_series >= threshold]
    if total_to_reallocate > 0 and not weights_to_keep.empty:
        proportional_base = weights_to_keep.sum()
        adjusted_weights = weights_to_keep + (weights_to_keep / proportional_base) * total_to_reallocate
        final_weights = pd.Series(0.0, index=weights_series.index)
        final_weights.update(adjusted_weights)
    else:
        final_weights = weights_series

    # --- Plotting ---
    plot_df = pd.DataFrame({
        'Asset': [name.replace('_Weight', '') for name in weight_cols],
        'Prior': importance_based_weights.values,
        'Optimal': final_weights.values
    })
    plot_df['Difference'] = plot_df['Optimal'] - plot_df['Prior']

    # MODIFIED: Filter to show any asset with a meaningful prior OR optimal weight
    plot_df_filtered = plot_df[(plot_df['Prior'] > 0.001) | (plot_df['Optimal'] > 0.001)].sort_values(by='Optimal', ascending=True)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, max(8, len(plot_df_filtered) * 0.4))) # Dynamically adjust height

    y_pos = np.arange(len(plot_df_filtered))
    bar_height = 0.25

    ax.barh(y_pos + bar_height, plot_df_filtered['Prior'], height=bar_height, label='Prior (from Importance)', color='skyblue')
    ax.barh(y_pos, plot_df_filtered['Optimal'], height=bar_height, label='Adjusted Optimal (>1%)', color='mediumseagreen')
    ax.barh(y_pos - bar_height, plot_df_filtered['Difference'], height=bar_height, label='Difference (Adjusted - Prior)', color='salmon')

    ax.set_yticks(y_pos, labels=plot_df_filtered['Asset'])
    ax.invert_yaxis()
    ax.set_xlabel('Portfolio Weight')
    ax.set_title('Portfolio Allocation: Prior vs. Adjusted Optimal Weights', fontsize=16)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.axvline(0, color='grey', linewidth=0.8)
    ax.legend()

    plt.tight_layout()
    plt.savefig('portfolio_comparison.png', dpi=300)
    print("Plot saved successfully to 'portfolio_comparison.png'.")

    # --- Text Display ---
    final_predicted_ir = -objective_optimization(final_weights.values, final_model, scaler, fixed_features_for_prediction, weight_cols, X.columns, 0, importance_based_weights, 0)
    print("\n" + "="*50)
    print("MODEL-BASED OPTIMIZATION COMPLETE")
    print(f"Date for Optimization: {df_predict_point.index[0].date()}")
    print(f"Time: 07:16:52 PM CEST")
    print(f"Location: Budapest, Hungary")
    print(f"Model's Predicted Maximum Information Ratio: {final_predicted_ir:.4f}")

    print("\n--- Final Optimal Portfolio Weights (Adjusted) ---")
    results_df = pd.DataFrame({'Asset': final_weights.index, 'Optimal Weight': final_weights.values}).sort_values(by='Optimal Weight', ascending=False)
    results_df_filtered_text = results_df[results_df['Optimal Weight'] > 0]
    results_df_filtered_text['Optimal Weight'] = results_df_filtered_text['Optimal Weight'].apply(lambda x: f"{x:.2%}")
    print(results_df_filtered_text.to_string(index=False))
    print("="*50 + "\n")

else:
    print("\nOptimization failed.")
    print(f"Message: {optimization_result.message}")

print("--- End of Pipeline ---")
# %%
