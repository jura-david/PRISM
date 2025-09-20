# ##############################################################################
# #                DUAL-STRATEGY INTERLEAVED BACKTESTING ENGINE                #
# #                  (With P&L Constraint & Full Reporting)                    #
# ##############################################################################

using CSV, DataFrames, Dates, Statistics, Random, Plots, Printf, ProgressMeter, Tables, StatsPlots, GLM, PyCall, StatsBase, LinearAlgebra, PortfolioOptimiser

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
#           --- Switches & Configuration ---
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# --- CHOOSE SCRIPT MODE ---
# :simulate - Runs the full, time-consuming backtest simulation to generate weights and stats.
# :analyze  - Skips simulation and uses existing output files to generate all reports and plots.
const RUN_MODE = :simulate

const BENCHMARK = "SPY"
const TEST_MODE = false # Set to false to run the full backtest
const NUM_FORECASTS_IN_TEST_MODE = 3
const MIN_TRAINING_QUARTERS = 20
const INITIAL_CAPITAL = 1_000.0
const COMMISSION_RATE = 0.005
const REBALANCE_WEEK_WINDOW = 1:2
const RESULTS_DIR = joinpath("Plots", "Backtest")

# --- CORRECTED: File paths for logs and data checkpoints ---
const LOGS_DIR = "Logs"
const DATA_DIR = "Data"
const WEIGHTS_DIR = joinpath(DATA_DIR, "Weights")
const CHECKPOINTS_DIR = joinpath(DATA_DIR, "Checkpoints")
const APSO_CHECKPOINT_FILE = joinpath(CHECKPOINTS_DIR, "precomputed_apso_results.csv")
const WF_PROGRESS_FILE = joinpath(LOGS_DIR, "walk_forward_progress.txt")
const STATS_CSV_FILE = joinpath(DATA_DIR, "rebalancing_statistics.csv")
const STATE_FILE = joinpath(LOGS_DIR, "apso_pipeline_state.json")

# --- Include and Setup Pipeline ---
if RUN_MODE == :simulate
	println("Including the optimization pipeline from 'Portfolio_Optimization_prod.jl'...")
	include("Portfolio_Optimization_prod.jl")
	println("Pipeline included successfully.\n")
end

# ##############################################################################
# #             SECTION 0: SHARED HELPER FUNCTIONS (FOR BOTH MODES)        #
# ##############################################################################

mutable struct Portfolio
	cash::Float64
	positions::Dict{String, Float64}
end

function pivot_allocations_to_row(tall_df::DataFrame, forecast_date::Date)
	row_df = DataFrame(Date = forecast_date)
	for row in eachrow(tall_df)
		row_df[!, Symbol(row.Asset * "_Weight")] = [row.Weight * 100.0]
	end
	return isempty(names(row_df)) ? DataFrame(Date = forecast_date) : row_df[1, :]
end

function rebalance_portfolio!(portfolio::Portfolio, target_weights::DataFrameRow, prices_on_date::DataFrameRow)
	current_holdings = keys(portfolio.positions)
	target_holdings = [replace(String(s), "_Weight"=>"") for s in names(target_weights) if endswith(String(s), "_Weight")]
	all_involved_assets = union(Set(current_holdings), Set(target_holdings))

	current_positions_value = sum((get(portfolio.positions, s, 0.0) * prices_on_date[s] for s in keys(portfolio.positions) if !ismissing(prices_on_date[s])), init = 0.0)
	total_value = portfolio.cash + current_positions_value
	buys_value, sells_value = 0.0, 0.0

	for stock in all_involved_assets
		if hasproperty(prices_on_date, stock) && !ismissing(prices_on_date[stock])
			target_weight = get(target_weights, Symbol(stock * "_Weight"), 0.0) / 100.0
			target_value = total_value * target_weight
			current_value = get(portfolio.positions, stock, 0.0) * prices_on_date[stock]
			trade_value = target_value - current_value
			if trade_value > 0
				buys_value += trade_value
			else
				sells_value += -trade_value
			end
		end
	end

	commission = (buys_value + sells_value) * COMMISSION_RATE
	turnover = total_value > 1e-8 ? min(buys_value, sells_value) / total_value : 0.0

	net_value_after_commission = total_value - commission
	portfolio.cash = net_value_after_commission
	new_positions = Dict{String, Float64}()

	for stock in all_involved_assets
		if hasproperty(prices_on_date, stock) && !ismissing(prices_on_date[stock])
			target_weight = get(target_weights, Symbol(stock * "_Weight"), 0.0) / 100.0
			stock_value_alloc = net_value_after_commission * target_weight
			if stock_value_alloc > 1e-6
				price = prices_on_date[stock]
				if price > 1e-8
					new_positions[stock] = stock_value_alloc / price
					portfolio.cash -= stock_value_alloc
				end
			end
		end
	end
	portfolio.positions = new_positions
	return commission, turnover
end

function load_and_prepare_dividends(path::String)
	println("--- Loading and preparing dividend data...")
	try
		dividends_wide = CSV.read(path, DataFrame; dateformat = "yyyy-mm-dd")
		sort!(dividends_wide, :Date)
		dividend_lookup = Dict(row.Date => row for row in eachrow(dividends_wide))
		println("Dividend data loaded successfully.")
		return dividend_lookup
	catch e
		println("WARNING: Could not load or process dividend file '$path'. Running without dividend adjustments. Error: $e")
		return Dict{Date, DataFrameRow}()
	end
end

function get_average_risk_free_rate(sim_start_date::Date)
	BOND_YIELD_PATH = "/home/felhasznalo/Scrapers/Data/CSV/Bonds/Govies/Yields/Bond_Yields_10Y_US.csv"
	try
		println("--- Loading 10Y US Treasury bond yield data from: $(BOND_YIELD_PATH)")
		bond_df = CSV.read(BOND_YIELD_PATH, DataFrame; dateformat = "mm/dd/yyyy")
		# Ensure the first column is named 'Date' for filtering
		rename!(bond_df, names(bond_df)[1] => :Date)
		rename!(bond_df, names(bond_df)[2] => :Yield)

		sim_end_date = today()
		filtered_bonds = filter(row -> sim_start_date <= row.Date <= sim_end_date, bond_df)

		if !isempty(filtered_bonds)
			avg_yield = mean(skipmissing(filtered_bonds.Yield))
			RISK_FREE_RATE = avg_yield / 100.0
			println(@sprintf("--- Calculated Average Risk-Free Rate over simulation period: %.4f%% ---", avg_yield))
			return RISK_FREE_RATE
		else
			println("WARNING: No bond yield data found for the simulation period. Defaulting to 2% risk-free rate.")
			return 0.02
		end
	catch e
		println("WARNING: Could not load or process bond yield file. Defaulting to 2% risk-free rate. Error: $e")
		return 0.02
	end
end


# ##############################################################################
# #                        SECTION 1: BACKTEST SIMULATION                      #
# ##############################################################################

if RUN_MODE == :simulate

	# --- Simulation-Only Helper Functions ---
	function run_classical_optimizations(prices_slice::DataFrame, benchmark::String, target_date::Date)
		println("--- Running Classical Optimizations (Sharpe, MinVol, InvVol, MaxRet) ---")

		stock_cols_initial = [col for col in names(prices_slice) if String(col) != benchmark && String(col) != "Date"]

		valid_stock_cols = String[]
		for col in stock_cols_initial
			valid_prices = filter(x -> !ismissing(x), prices_slice[!, col])
			if length(unique(valid_prices)) >= 2
				push!(valid_stock_cols, col)
			end
		end

		if length(valid_stock_cols) < 2
			println("WARNING: Not enough valid assets for classical optimizations after filtering. Returning empty.")
			return Dict{String, DataFrame}()
		end

		returns_df = DataFrame()
		for col in valid_stock_cols
			prices = prices_slice[!, col]
			log_returns = [NaN; log.(prices[2:end] ./ prices[1:(end-1)])]
			returns_df[!, col] = coalesce.(log_returns, 0.0)
		end

		returns_matrix = Matrix(returns_df)
		returns_matrix[.!isfinite.(returns_matrix)] .= 0.0

		μ = vec(mean(returns_matrix, dims = 1)) * 52

		lw = sklearn_covariance.LedoitWolf()
		lw.fit(returns_matrix)
		Σ = convert(Matrix{Float64}, lw."covariance_") * 52

		num_assets = length(valid_stock_cols)
		results = Dict{String, DataFrame}()

		# --- Max Sharpe Ratio ---
		function sharpe_objective(weights)
			weights = convert(Vector{Float64}, weights)
			portfolio_return = dot(weights, μ)
			portfolio_volatility = sqrt(max(1e-12, weights' * Σ * weights))
			return -portfolio_return / portfolio_volatility
		end

		constraints = (py"dict"(type = "eq", fun = w -> sum(w) - 1.0))
		bounds = [(0.0, 1.0) for _ in 1:num_assets]
		initial_weights = fill(1.0 / num_assets, num_assets)

		opt_result_sharpe = sp_optimize.minimize(sharpe_objective, initial_weights, method = "SLSQP", bounds = bounds, constraints = constraints)

		if opt_result_sharpe["success"]
			optimal_weights = convert(Vector{Float64}, opt_result_sharpe["x"])
			optimal_weights ./= sum(optimal_weights)
			sharpe_weights_df = DataFrame(Asset = valid_stock_cols, Weight = optimal_weights)
			results["MaxSharpe"] = pivot_allocations_to_row(sharpe_weights_df, target_date)
		else
			println("WARNING: Max Sharpe optimization failed.")
		end

		# --- Minimum Volatility ---
		function vol_objective(weights)
			weights = convert(Vector{Float64}, weights)
			return sqrt(max(1e-12, weights' * Σ * weights))
		end

		opt_result_minvol = sp_optimize.minimize(vol_objective, initial_weights, method = "SLSQP", bounds = bounds, constraints = constraints)

		if opt_result_minvol["success"]
			optimal_weights = convert(Vector{Float64}, opt_result_minvol["x"])
			optimal_weights ./= sum(optimal_weights)
			minvol_weights_df = DataFrame(Asset = valid_stock_cols, Weight = optimal_weights)
			results["MinVol"] = pivot_allocations_to_row(minvol_weights_df, target_date)
		else
			println("WARNING: Min Vol optimization failed.")
		end

		# --- Inverse Volatility ---
		asset_vols = [sqrt(Σ[i, i]) for i in 1:num_assets]
		inv_vols = 1.0 ./ asset_vols
		inv_vol_weights = inv_vols ./ sum(inv_vols)
		inv_vol_weights_df = DataFrame(Asset = valid_stock_cols, Weight = inv_vol_weights)
		results["InvVol"] = pivot_allocations_to_row(inv_vol_weights_df, target_date)

		# --- Maximum Return ---
		if !isempty(μ)
			max_ret_idx = argmax(μ)
			max_ret_weights = zeros(num_assets)
			max_ret_weights[max_ret_idx] = 1.0
			max_ret_weights_df = DataFrame(Asset = valid_stock_cols, Weight = max_ret_weights)
			results["MaxReturn"] = pivot_allocations_to_row(max_ret_weights_df, target_date)
		end

		return results
	end

	function run_hrp_optimizations(prices_slice::DataFrame, benchmark::String, target_date::Date)
		println("--- Running HRP/HERC/NCO Optimizations ---")
		stock_cols_initial = [col for col in names(prices_slice) if String(col) != benchmark && String(col) != "Date"]

		valid_stock_cols = String[]
		for col in stock_cols_initial
			valid_prices = filter(x -> !ismissing(x), prices_slice[!, col])
			if length(unique(valid_prices)) >= 2
				push!(valid_stock_cols, col)
			end
		end

		if length(valid_stock_cols) < 2
			println("WARNING: Not enough valid assets for HRP optimisations after filtering. Returning empty.")
			return Dict{String, DataFrame}()
		end

		returns_df = DataFrame()
		for col in valid_stock_cols
			prices = prices_slice[!, col]
			log_returns = [NaN; log.(prices[2:end] ./ prices[1:(end-1)])]
			returns_df[!, col] = coalesce.(log_returns, 0.0)
		end

		returns_matrix = Matrix(returns_df)
		returns_matrix[.!isfinite.(returns_matrix)] .= 0.0

		results = Dict{String, DataFrame}()
		try
			port = Portfolio(returns = returns_matrix, assets = valid_stock_cols)

			# HRP
			hrp_weights = optimise(port, OptimiseOpt(method = "HRP"))
			hrp_df = DataFrame(Asset = valid_stock_cols, Weight = hrp_weights.weights)
			results["HRP"] = pivot_allocations_to_row(hrp_df, target_date)

			# HERC
			herc_weights = optimise(port, OptimiseOpt(method = "HERC"))
			herc_df = DataFrame(Asset = valid_stock_cols, Weight = herc_weights.weights)
			results["HERC"] = pivot_allocations_to_row(herc_df, target_date)

			# NCO
			nco_weights = optimise(port, OptimiseOpt(method = "NCO"))
			nco_df = DataFrame(Asset = valid_stock_cols, Weight = nco_weights.weights)
			results["NCO"] = pivot_allocations_to_row(nco_df, target_date)

		catch e
			println("ERROR during PortfolioOptimiser execution: $e")
		end

		return results
	end


	function load_daily_prices(path::String)
		println("--- Loading daily prices from '$path'...")
		try
			header = Tables.schema(CSV.File(path)).names
			col_types = Dict(c => Float64 for c in header if c != :Date)
			df = CSV.read(path, DataFrame; missingstring = "NA", types = col_types, dateformat = "yyyy-mm-dd")
			sort!(df, :Date)
			println("Daily prices loaded and parsed: $(nrow(df)) rows, $(ncol(df)) columns.")
			return df
		catch e
			println("FATAL ERROR: Could not load or parse daily prices file. Details: $e")
			return nothing
		end
	end

	function get_traded_stocks_for_quarter(daily_prices::DataFrame, all_stocks::Vector{String}, quarter_start_date::Date)
		prev_q_end = quarter_start_date - Day(1)
		prev_q_start = firstdayofquarter(prev_q_end)
		prices_in_prev_q = filter(row -> prev_q_start <= row.Date <= prev_q_end, daily_prices)
		traded_stocks = String[]
		if !isempty(prices_in_prev_q)
			for stock in all_stocks
				if stock in names(prices_in_prev_q) && !all(ismissing, prices_in_prev_q[:, stock])
					push!(traded_stocks, stock)
				end
			end
		end
		return traded_stocks
	end

	function find_rebalance_schedule_for_simulation(all_quarters::Vector{Date}, daily_prices::DataFrame)
		println("--- Determining all potential rebalance dates (ONCE per quarter)...")
		rebalance_schedule = Dict{Date, Date}()
		rebalance_window = REBALANCE_WEEK_WINDOW

		for i in (MIN_TRAINING_QUARTERS+1):length(all_quarters)
			q_start = all_quarters[i]
			q_end = lastdayofquarter(q_start)
			all_potential_stocks = filter(c -> c != "Date", names(daily_prices))
			traded_stocks = get_traded_stocks_for_quarter(daily_prices, all_potential_stocks, q_start)

			combined_valid_days = Date[]
			for week_offset in rebalance_window
				target_week_start = q_end + Day(1) + Week(week_offset - 1)
				target_week_end = target_week_start + Day(6)
				week_prices = filter(r -> target_week_start <= r.Date <= target_week_end, daily_prices)
				if !isempty(week_prices)
					for day_row in eachrow(week_prices)
						if all(!ismissing(day_row[s]) for s in traded_stocks)
							push!(combined_valid_days, day_row.Date)
						end
					end
				end
			end
			if !isempty(combined_valid_days)
				rebalance_day = rand(combined_valid_days)
				rebalance_schedule[rebalance_day] = q_start
			end
		end
		println("Found $(length(rebalance_schedule)) potential rebalance dates.")
		return rebalance_schedule
	end

	function save_weights_to_csv(weights_row::DataFrameRow, strategy_name::String, weights_dir::String)
		date = weights_row.Date
		assets = String[]
		weights = Float64[]

		for col_name in names(weights_row)
			if endswith(string(col_name), "_Weight")
				push!(assets, replace(string(col_name), "_Weight" => ""))
				push!(weights, weights_row[col_name] / 100.0)
			end
		end

		if isempty(assets)
			return
		end

		tall_df = DataFrame(Date = date, Asset = assets, Weight = weights)

		file_path = joinpath(weights_dir, "$(strategy_name)_weights.csv")
		is_new_file = !isfile(file_path) || filesize(file_path) == 0
		CSV.write(file_path, tall_df; append = !is_new_file, header = is_new_file)
	end

	function get_current_weights_as_row(portfolio::Portfolio, prices_on_date::DataFrameRow, rebalance_date::Date)
		positions_value = sum(get(portfolio.positions, s, 0.0) * coalesce(prices_on_date[Symbol(s)], 0.0) for s in keys(portfolio.positions))
		total_value = portfolio.cash + positions_value

		if total_value < 1e-8
			return DataFrame(Date = rebalance_date)
		end

		assets = String[]
		weights = Float64[]
		for (asset, shares) in portfolio.positions
			asset_value = shares * coalesce(prices_on_date[Symbol(asset)], 0.0)
			push!(assets, asset)
			push!(weights, (asset_value / total_value))
		end

		return pivot_allocations_to_row(DataFrame(Asset = assets, Weight = weights), rebalance_date)
	end

	function calculate_period_statistics(histories::Dict, start_date, end_date)
		stats = Dict("Rebalance_Date" => start_date)

		function get_hpr(p_slice)
			if isempty(p_slice) || nrow(p_slice) < 2
				return missing
			end
			return (p_slice.Value[end] / p_slice.Value[1]) - 1
		end

		for (name, history_df) in histories
			p_slice = filter(r -> start_date <= r.Date <= end_date, history_df)
			stats["$(name)_HPR_Period"] = get_hpr(p_slice)
		end

		return stats
	end


	# --- Main Interleaved Simulation Engine ---
	function run_interleaved_walk_forward_simulation()
		# 1. --- Initial Data Loading & Setup ---
		println("--- Loading all historical raw data...")
		mkpath(RESULTS_DIR);
		mkpath(LOGS_DIR);
		mkpath(DATA_DIR);
		mkpath(CHECKPOINTS_DIR);
		mkpath(WEIGHTS_DIR);

		daily_prices = load_daily_prices(joinpath(DATA_DIR, "prices_daily.csv"))
		if daily_prices === nothing
			;
			return;
		end

		dividend_lookup = load_and_prepare_dividends(joinpath(DATA_DIR, "prices_dividends.csv"))

		market_quarterly = CSV.read(joinpath(DATA_DIR, "market_quarterly.csv"), DataFrame; dateformat = "yyyy-mm-dd")
		prices_weekly = CSV.read(joinpath(DATA_DIR, "prices_weekly.csv"), DataFrame; dateformat = "yyyy-mm-dd")
		all_quarters = unique(firstdayofquarter.(market_quarterly.Date))

		# --- Initialize all portfolio structs ---
		portfolios = Dict(
			"PRISM_A" => Portfolio(INITIAL_CAPITAL, Dict{String, Float64}()),
			"PRISM_B" => Portfolio(INITIAL_CAPITAL, Dict{String, Float64}()),
			"BuyAndHold" => Portfolio(INITIAL_CAPITAL, Dict{String, Float64}()),
			"MaxSharpe" => Portfolio(INITIAL_CAPITAL, Dict{String, Float64}()),
			"MinVol" => Portfolio(INITIAL_CAPITAL, Dict{String, Float64}()),
			"InvVol" => Portfolio(INITIAL_CAPITAL, Dict{String, Float64}()),
			"MaxReturn" => Portfolio(INITIAL_CAPITAL, Dict{String, Float64}()),
			"EqualWeight" => Portfolio(INITIAL_CAPITAL, Dict{String, Float64}()),
			"HRP" => Portfolio(INITIAL_CAPITAL, Dict{String, Float64}()),
			"HERC" => Portfolio(INITIAL_CAPITAL, Dict{String, Float64}()),
			"NCO" => Portfolio(INITIAL_CAPITAL, Dict{String, Float64}()),
		)

		# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
		#   Phase 1: Pre-compute APSO Signals (Runs Once with Checkpointing)
		# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
		local precomputed_apso_df
		is_apso_complete = false

		num_quarters_for_apso = TEST_MODE ? min(MIN_TRAINING_QUARTERS + NUM_FORECASTS_IN_TEST_MODE, length(all_quarters)) : length(all_quarters)
		target_last_quarter = all_quarters[num_quarters_for_apso]

		if isfile(APSO_CHECKPOINT_FILE)
			println("--- Found existing APSO checkpoint file. Verifying completeness... ---")
			try
				temp_df = CSV.read(APSO_CHECKPOINT_FILE, DataFrame)
				if !isempty(temp_df)
					last_date_in_file = maximum(Date.(temp_df.Date))
					if last_date_in_file >= target_last_quarter
						println("--- APSO data is complete. Loading results. ---")
						precomputed_apso_df = temp_df
						precomputed_apso_df.Date = Date.(precomputed_apso_df.Date)
						is_apso_complete = true
					else
						println("--- APSO data is INCOMPLETE (Last quarter: $last_date_in_file, Target: $target_last_quarter). Resuming computation... ---")
					end
				else
					println("--- APSO checkpoint file is empty. Re-running computation... ---")
				end
			catch e
				println("--- WARNING: Could not read checkpoint file. Re-running computation. Error: $e ---")
			end
		end

		if !is_apso_complete
			println("\n" * "#"^60, "\n### PHASE 1: PRE-COMPUTING/RESUMING ALL HISTORICAL APSO SIGNALS ###")

			apso_training_start_date = all_quarters[1]
			apso_training_end_date = lastdayofquarter(target_last_quarter)
			println("APSO pre-computation window set from $(apso_training_start_date) to $(apso_training_end_date).")

			apso_market_slice = filter(row -> apso_training_start_date <= row.Date <= apso_training_end_date, market_quarterly)
			apso_prices_slice = filter(row -> apso_training_start_date <= row.Date <= apso_training_end_date, prices_weekly)

			apso_df, _, _, _ = run_apso_stage(
				apso_market_slice,
				apso_prices_slice;
				n_bootstrap_samples = APSO_BOOTSTRAP_SAMPLES,
				set_particles = APSO_SWARM_PARTICLES,
				set_iters_pso = APSO_ITERATIONS,
				set_starts = APSO_MULTI_STARTS,
				pso_w_decay_rate = APSO_W_DECAY,
				pso_c1_decay_rate = APSO_C1_DECAY,
				pso_c2_increase_rate = APSO_C2_INCREASE,
				diversification_penalty = APSO_DIVERSIFICATION_PENALTY,
				benchmark = BENCHMARK,
				state_file_path = STATE_FILE,
				output_csv_path = APSO_CHECKPOINT_FILE,
			)

			if apso_df === nothing || nrow(apso_df) < 2
				println("CRITICAL ERROR: Initial APSO pre-computation failed. Halting backtest.")
				return
			end
			precomputed_apso_df = apso_df
		end
		println("###      APSO PRE-COMPUTATION COMPLETE. PROCEEDING TO WALK-FORWARD.     ###\n" * "#"^60)

		# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
		#   Phase 2: Walk-Forward Simulation
		# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
		rebalance_schedule = find_rebalance_schedule_for_simulation(all_quarters, daily_prices)
		if isempty(rebalance_schedule)
			;
			return;
		end

		last_rebalance_date = nothing
		first_trade_date = nothing
		all_rebalance_dates = sort(collect(keys(rebalance_schedule)))

		first_forecast_quarter = all_quarters[MIN_TRAINING_QUARTERS+1]
		valid_rebalance_dates = filter(d -> rebalance_schedule[d] >= first_forecast_quarter, all_rebalance_dates)

		histories = Dict(name => DataFrame(Date = Date[], Value = Union{Float64, Missing}[]) for name in keys(portfolios))
		histories["Benchmark"] = DataFrame(Date = Date[], Value = Union{Float64, Missing}[])

		last_completed_date = Date(0)

		if isfile(WF_PROGRESS_FILE)
			last_date_str = read(WF_PROGRESS_FILE, String)
			try
				last_completed_date = Date(last_date_str)
				println("--- Resuming from last completed rebalance date: $(last_completed_date) ---")
			catch e
				println("WARNING: Could not parse date from progress file. Starting from scratch. Error: $e")
				if isfile(STATS_CSV_FILE)
					;
					rm(STATS_CSV_FILE);
				end
			end
		else
			if isfile(STATS_CSV_FILE)
				;
				rm(STATS_CSV_FILE);
			end
			if isdir(WEIGHTS_DIR)
				println("--- Detected a fresh run. Clearing old weights directory. ---")
				rm(WEIGHTS_DIR, recursive = true, force = true)
			end
			mkpath(WEIGHTS_DIR)
		end

		println("\n" * "#"^60, "\n###   PHASE 2: STARTING WALK-FORWARD FEATURE & FORECASTING STAGES  ###")
		num_rebalances_to_run = TEST_MODE ? min(NUM_FORECASTS_IN_TEST_MODE, length(valid_rebalance_dates)) : length(valid_rebalance_dates)

		if num_rebalances_to_run == 0
			;
			return;
		end
		if TEST_MODE
			println("###          >>> RUNNING IN TEST MODE (Max $(num_rebalances_to_run) Rebalances) <<<       ###");
		end
		println("#"^60 * "\n")

		sim_start_date = valid_rebalance_dates[1]
		sim_end_date = (num_rebalances_to_run < length(valid_rebalance_dates)) ? (valid_rebalance_dates[num_rebalances_to_run+1] - Day(1)) : daily_prices.Date[end]

		prog = Progress(num_rebalances_to_run, "Walk-Forward Backtest:")
		rebalance_counter = 0
		last_known_prices = Dict{String, Float64}()
		all_stock_names = filter(name -> name != "Date", names(daily_prices))
		last_transaction_stats = Dict{String, Any}()
		bnh_initialized = false

		for day_row in eachrow(filter(r -> sim_start_date <= r.Date <= sim_end_date, daily_prices))
			current_date = day_row.Date

			for stock in all_stock_names
				if !ismissing(day_row[stock])
					last_known_prices[stock] = day_row[stock];
				end
			end

			if haskey(dividend_lookup, current_date)
				dividend_row = dividend_lookup[current_date]
				for portfolio in values(portfolios)
					dividend_cash_received = 0.0
					for (stock, shares) in portfolio.positions
						if hasproperty(dividend_row, Symbol(stock))
							dividend_per_share = dividend_row[Symbol(stock)]
							if !ismissing(dividend_per_share) && dividend_per_share > 0
								dividend_cash_received += shares * dividend_per_share
							end
						end
					end
					portfolio.cash += dividend_cash_received
				end
			end

			if haskey(rebalance_schedule, current_date) && rebalance_counter < num_rebalances_to_run
				if current_date <= last_completed_date
					rebalance_counter += 1
					if first_trade_date === nothing
						;
						first_trade_date = current_date;
					end
					last_rebalance_date = current_date
					ProgressMeter.update!(prog, rebalance_counter, desc = "Skipping completed rebalance on $(current_date)")
				else
					rebalance_counter += 1
					if first_trade_date === nothing
						;
						first_trade_date = current_date;
					end

					if last_rebalance_date !== nothing
						period_end_date = current_date - Day(1)
						stats = calculate_period_statistics(histories, last_rebalance_date, period_end_date)

						if !isempty(stats) && !ismissing(get(stats, "PRISM_A_HPR_Period", missing))
							merge!(stats, last_transaction_stats)
							stats_df_row = DataFrame([stats])
							is_new_file = !isfile(STATS_CSV_FILE) || filesize(STATS_CSV_FILE) == 0
							CSV.write(STATS_CSV_FILE, stats_df_row; append = !is_new_file, header = is_new_file)
						end
					end

					ProgressMeter.update!(prog, rebalance_counter, desc = "Rebalance #$(rebalance_counter) on $(current_date)")
					q_start_for_forecast = rebalance_schedule[current_date]
					apso_data_for_current_step = filter(row -> row.Date <= q_start_for_forecast, precomputed_apso_df)

					if nrow(apso_data_for_current_step) < MIN_TRAINING_QUARTERS
						;
						continue;
					end

					### START: FEATURE ENGINEERING CHECKPOINTING ###
					local features_df
					fe_checkpoint_path = joinpath(CHECKPOINTS_DIR, "FE_$(current_date).csv")

					if isfile(fe_checkpoint_path)
						println("\n--- Loading cached Feature Engineering results for $(current_date)...")
						features_df = CSV.read(fe_checkpoint_path, DataFrame)
						if "Date" in names(features_df) && !(eltype(features_df.Date) <: Date)
							features_df.Date = Date.(features_df.Date)
						end
					else
						println("\n--- Running Feature Engineering for $(current_date)...")
						features_df = run_feature_engineering_stage(
							apso_data_for_current_step;
							lag_vals = FE_FEATURE_LAGS,
							target_lag_vals = FE_TARGET_LAGS,
							window_vals = FE_ROLLING_WINDOWS,
							mi_std_multiplier = FE_MI_THRESHOLD_STD_MULTIPLIER,
							mi_fallback_n = FE_MI_FALLBACK_N_FEATURES,
							poly_degree = FE_POLYNOMIAL_DEGREE,
							sulov_corr_thresh = FE_SULOV_CORR_THRESHOLD,
							optuna_trials = FE_OPTUNA_TRIALS,
							sulov_plot_seed_nodes = FE_SULOV_PLOT_SEED_NODES,
							sulov_plot_k_neighbors = FE_SULOV_PLOT_K_NEIGHBORS,
						)
						if features_df !== nothing
							println("--- Caching Feature Engineering results for $(current_date)...")
							CSV.write(fe_checkpoint_path, features_df)
						end
					end
					### END: FEATURE ENGINEERING CHECKPOINTING ###

					if features_df === nothing
						;
						continue;
					end

					### START: FORECASTING & BENCHMARK OPTIMIZATIONS ###
					local forecast_result = nothing

					all_potential_stocks = filter(c -> String(c) != "Date" && String(c) != BENCHMARK, names(daily_prices))
					tradable_stocks_in_period = get_traded_stocks_for_quarter(daily_prices, all_potential_stocks, q_start_for_forecast)

					fc_prism_a_path = joinpath(CHECKPOINTS_DIR, "FC_PRISM_A_$(current_date).csv")
					fc_prism_b_path = joinpath(CHECKPOINTS_DIR, "FC_PRISM_B_$(current_date).csv")

					if isfile(fc_prism_a_path) && isfile(fc_prism_b_path)
						println("--- Loading cached Forecasting results for $(current_date)...")
						prism_a_weights_tall = CSV.read(fc_prism_a_path, DataFrame)
						prism_b_weights_tall = CSV.read(fc_prism_b_path, DataFrame)
						forecast_result = (prism_a_weights_tall, prism_b_weights_tall)
					else
						println("--- Running Forecasting for $(current_date)...")
						forecast_result = run_forecasting_stage(
							features_df,
							prices_weekly;
							pca_variance_threshold = FC_PCA_VARIANCE_THRESHOLD,
							ridge_alpha_start = FC_RIDGE_ALPHA_START,
							ridge_alpha_end = FC_RIDGE_ALPHA_END,
							ridge_alpha_count = FC_RIDGE_ALPHA_COUNT,
							ridge_cv_folds = FC_RIDGE_CV_FOLDS,
							nomad_max_evals = FC_NOMAD_MAX_EVALS,
							n_bootstrap = FC_BOOTSTRAP_SAMPLES,
							benchmark = BENCHMARK,
							tradable_universe = tradable_stocks_in_period,
						)

						if forecast_result !== nothing
							println("--- Caching Forecasting results for $(current_date)...")
							CSV.write(fc_prism_a_path, forecast_result[1])
							CSV.write(fc_prism_b_path, forecast_result[2])
						end
					end

					target_allocations = Dict{String, DataFrameRow}()

					if forecast_result !== nothing
						prism_a_weights_tall, prism_b_weights_tall = forecast_result
						target_allocations["PRISM_A"] = pivot_allocations_to_row(prism_a_weights_tall, current_date)
						target_allocations["PRISM_B"] = pivot_allocations_to_row(prism_b_weights_tall, current_date)
					end

					prices_slice_for_opt = filter(row -> row.Date < current_date, prices_weekly)
					classical_results = run_classical_optimizations(prices_slice_for_opt, BENCHMARK, current_date)
					for (name, weights) in classical_results
						target_allocations[name] = weights
					end

					hrp_results = run_hrp_optimizations(prices_slice_for_opt, BENCHMARK, current_date)
					for (name, weights) in hrp_results
						target_allocations[name] = weights
					end

					if !isempty(tradable_stocks_in_period)
						num_assets_ew = length(tradable_stocks_in_period)
						ew_weights_df = DataFrame(Asset = tradable_stocks_in_period, Weight = fill(1.0 / num_assets_ew, num_assets_ew))
						target_allocations["EqualWeight"] = pivot_allocations_to_row(ew_weights_df, current_date)
					else
						target_allocations["EqualWeight"] = pivot_allocations_to_row(DataFrame(Asset = String[], Weight = Float64[]), current_date)
					end

					last_transaction_stats = Dict()
					for (name, portfolio) in portfolios
						if haskey(target_allocations, name)
							comm, turnover = rebalance_portfolio!(portfolio, target_allocations[name], day_row)
							last_transaction_stats["$(name)_Commission"] = comm
							last_transaction_stats["$(name)_Turnover"] = turnover
						end
					end

					if !bnh_initialized
						println("--- Initializing Buy & Hold portfolio with equal weights ---")
						rebalance_portfolio!(portfolios["BuyAndHold"], target_allocations["EqualWeight"], day_row)
						bnh_initialized = true
						save_weights_to_csv(target_allocations["EqualWeight"], "BuyAndHold", WEIGHTS_DIR)
					else
						bnh_current_weights_row = get_current_weights_as_row(portfolios["BuyAndHold"], day_row, current_date)
						save_weights_to_csv(bnh_current_weights_row, "BuyAndHold", WEIGHTS_DIR)
					end

					for (name, allocation) in target_allocations
						save_weights_to_csv(allocation, name, WEIGHTS_DIR)
					end

					open(WF_PROGRESS_FILE, "w") do f
						;
						write(f, string(current_date));
					end
					last_rebalance_date = current_date
				end
			end

			if first_trade_date !== nothing && current_date >= first_trade_date
				for (name, portfolio) in portfolios
					history_df = histories[name]
					if isempty(portfolio.positions) && portfolio.cash == INITIAL_CAPITAL
						push!(history_df, (Date = current_date, Value = INITIAL_CAPITAL))
						continue
					end
					positions_value = sum(begin
							price = ismissing(day_row[s]) ? get(last_known_prices, s, 0.0) : day_row[s]
							get(portfolio.positions, s, 0.0) * price
						end for s in keys(portfolio.positions); init = 0.0)
					val = portfolio.cash + positions_value
					push!(history_df, (Date = current_date, Value = val))
				end

				if isempty(histories["Benchmark"])
					start_price = daily_prices[findfirst(==(first_trade_date), daily_prices.Date), Symbol(BENCHMARK)]
					if ismissing(start_price) || start_price == 0
						;
						start_price = 1.0;
					end
					push!(histories["Benchmark"], (Date = current_date, Value = INITIAL_CAPITAL))
				else
					benchmark_start_val = histories["Benchmark"].Value[1]
					benchmark_start_price = daily_prices[findfirst(==(first_trade_date), daily_prices.Date), Symbol(BENCHMARK)]
					benchmark_price_today = ismissing(day_row[Symbol(BENCHMARK)]) ? get(last_known_prices, BENCHMARK, benchmark_start_price) : day_row[Symbol(BENCHMARK)]
					push!(histories["Benchmark"], (Date = current_date, Value = benchmark_start_val * (benchmark_price_today / benchmark_start_price)))
				end
			end
		end

		# 3. --- Final Reporting ---
		finish!(prog)
		if isfile(WF_PROGRESS_FILE)
			;
			rm(WF_PROGRESS_FILE);
		end
		if isdir(CHECKPOINTS_DIR)
			;
			rm(CHECKPOINTS_DIR, recursive = true, force = true);
		end
		println("\n--- Backtest complete. ---")
	end

	run_interleaved_walk_forward_simulation()

end # END OF SIMULATION MODE


# ##############################################################################
# #                       SECTION 2: ANALYSIS & REPORTING                      #
# ##############################################################################

if RUN_MODE == :analyze

	function run_analysis_and_reporting()
		println("\n" * "="^60)
		println("              RUNNING IN ANALYSIS & REPORTING MODE")
		println("="^60 * "\n")

		# --- Load all necessary data ---
		println("--- Loading simulation output files...")
		local prices_df, stats_df

		strategy_files = Dict(
			"PRISM_A" => "PRISM_A_weights.csv",
			"PRISM_B" => "PRISM_B_weights.csv",
			"BuyAndHold" => "BuyAndHold_weights.csv",
			"MaxSharpe" => "MaxSharpe_weights.csv",
			"MinVol" => "MinVol_weights.csv",
			"InvVol" => "InvVol_weights.csv",
			"MaxReturn" => "MaxReturn_weights.csv",
			"EqualWeight" => "EqualWeight_weights.csv",
			"HRP" => "HRP_weights.csv",
			"HERC" => "HERC_weights.csv",
			"NCO" => "NCO_weights.csv",
		)

		weights_data = Dict{String, DataFrame}()

		try
			prices_df = CSV.read(joinpath(DATA_DIR, "prices_daily.csv"), DataFrame; dateformat = "yyyy-mm-dd")
			stats_df = CSV.read(STATS_CSV_FILE, DataFrame; dateformat = "yyyy-mm-dd")

			if nrow(stats_df) >= 2
				println("--- Deleting the second row from statistics file to remove potential data corruption ---")
				stats_df = stats_df[Not(2), :]
			end

			for (name, file) in strategy_files
				weights_data[name] = CSV.read(joinpath(WEIGHTS_DIR, file), DataFrame; dateformat = "yyyy-mm-dd")
			end
			println("All required data files loaded successfully.")
		catch e
			println("\nFATAL ERROR: Could not load necessary data files from 'Data/' and 'Data/Weights/'.")
			println("Please run the simulation mode (`RUN_MODE = :simulate`) first to generate these files.")
			println("Details: $e")
			return
		end

		dividend_lookup = load_and_prepare_dividends(joinpath(DATA_DIR, "prices_dividends.csv"))

		# --- Define consistent color palette ---
		strategy_colors = Dict(
			"Benchmark"   => :deepskyblue,
			"PRISM_A"     => :crimson,
			"PRISM_B"     => :orange,
			"MaxSharpe"   => :sienna,
			"MinVol"      => :gold,
			"InvVol"      => :olivedrab,
			"MaxReturn"   => :hotpink,
			"BuyAndHold"  => :limegreen,
			"EqualWeight" => :mediumorchid,
			"HRP"         => :teal,
			"HERC"        => :darkcyan,
			"NCO"         => :indigo,
		)
		# Add aliases for grouped bar charts
		strategy_colors["Buy & Hold"] = strategy_colors["BuyAndHold"]
		strategy_colors["Equal-Weight"] = strategy_colors["EqualWeight"]

		# --- Reconstruct daily portfolio history from weights ---
		println("--- Reconstructing daily portfolio histories...")
		all_strategies = Dict(
			"PRISM_A" => weights_data["PRISM_A"],
			"PRISM_B" => weights_data["PRISM_B"],
			"Buy & Hold" => weights_data["BuyAndHold"],
			"Max Sharpe" => weights_data["MaxSharpe"],
			"Min Vol" => weights_data["MinVol"],
			"Inv Vol" => weights_data["InvVol"],
			"Max Return" => weights_data["MaxReturn"],
			"Equal-Weight" => weights_data["EqualWeight"],
			"HRP" => weights_data["HRP"],
			"HERC" => weights_data["HERC"],
			"NCO" => weights_data["NCO"],
		)

		portfolio_histories = Dict{String, DataFrame}()
		all_stock_names = names(prices_df, Not(:Date))

		# --- Build a complete daily price lookup to handle all missing data robustly ---
		first_rebalance_date = minimum(minimum(w.Date) for w in values(all_strategies))
		simulation_price_data = filter(r -> r.Date >= first_rebalance_date, prices_df)

		price_lookup = Dict{Date, Dict{String, Float64}}()
		last_known_prices = Dict{String, Float64}()

		# Initialize last_known_prices with the first valid price for each asset
		for symbol in all_stock_names
			first_valid_price_row = findfirst(row -> !ismissing(row[symbol]), eachrow(simulation_price_data))
			last_known_prices[symbol] = (first_valid_price_row !== nothing) ? simulation_price_data[first_valid_price_row, symbol] : 0.0
		end

		for day_row in eachrow(simulation_price_data)
			for symbol in all_stock_names
				if !ismissing(day_row[symbol])
					last_known_prices[symbol] = day_row[symbol]
				end
			end
			price_lookup[day_row.Date] = copy(last_known_prices)
		end

		for (name, weights_df) in all_strategies
			history = DataFrame(Date = Date[], Value = Union{Float64, Missing}[])
			portfolio = Portfolio(INITIAL_CAPITAL, Dict{String, Float64}())
			rebalance_dates = sort(unique(weights_df.Date))

			sim_start_date = first(rebalance_dates)
			sim_end_date = last(prices_df.Date)

			for day_row in eachrow(filter(r -> sim_start_date <= r.Date <= sim_end_date, prices_df))
				current_date = day_row.Date
				if !haskey(price_lookup, current_date)
					;
					continue;
				end
				todays_prices = price_lookup[current_date]

				if haskey(dividend_lookup, current_date)
					dividend_row = dividend_lookup[current_date]
					dividend_cash_received = 0.0
					for (stock, shares) in portfolio.positions
						if hasproperty(dividend_row, Symbol(stock))
							dividend_per_share = dividend_row[Symbol(stock)]
							if !ismissing(dividend_per_share) && dividend_per_share > 0
								dividend_cash_received += shares * dividend_per_share
							end
						end
					end
					portfolio.cash += dividend_cash_received
				end

				if current_date in rebalance_dates
					target_weights_tall = filter(r -> r.Date == current_date, weights_df)
					target_weights_wide = pivot_allocations_to_row(target_weights_tall, current_date)
					if length(target_weights_wide) > 1
						rebalance_portfolio!(portfolio, target_weights_wide, day_row)
					end
				end

				positions_value = sum(get(portfolio.positions, s, 0.0) * get(todays_prices, s, 0.0) for s in keys(portfolio.positions); init = 0.0)
				total_value = portfolio.cash + positions_value
				push!(history, (Date = current_date, Value = total_value))
			end
			portfolio_histories[name] = history
		end

		# --- Reconstruct Benchmark history with spike protection ---
		bench_history = DataFrame(Date = Date[], Value = Union{Float64, Missing}[])
		bench_prices_subset = filter(r -> r.Date >= first_rebalance_date, prices_df)
		bench_start_price = price_lookup[first_rebalance_date][BENCHMARK]

		for day_row in eachrow(bench_prices_subset)
			current_price = price_lookup[day_row.Date][BENCHMARK]
			bench_val = (current_price / bench_start_price) * INITIAL_CAPITAL
			push!(bench_history, (Date = day_row.Date, Value = bench_val))
		end
		portfolio_histories["Benchmark"] = bench_history
		println("Portfolio histories reconstructed.")

		# --- Calculate full-period metrics ---
		println("--- Calculating full-period performance and risk metrics...")
		metrics_results = []

		# Function to get periodic returns from a history
		function get_periodic_returns(history_df::DataFrame, period_days::Int)
			if nrow(history_df) < 2
				;
				return Union{Float64, Missing}[];
			end
			returns = Union{Float64, Missing}[]
			start_val = history_df.Value[1]
			start_date = history_df.Date[1]

			for i in 2:nrow(history_df)
				if (history_df.Date[i] - start_date).value >= period_days
					current_val = history_df.Value[i]
					if ismissing(current_val) || ismissing(start_val) || start_val == 0.0
						push!(returns, missing)
					else
						push!(returns, (current_val / start_val) - 1)
					end
					start_val = current_val
					start_date = history_df.Date[i]
				end
			end
			if !ismissing(history_df.Value[end]) && history_df.Value[end] != start_val && (history_df.Date[end] - start_date).value > 0
				current_val = history_df.Value[end]
				if ismissing(current_val) || ismissing(start_val) || start_val == 0.0
					push!(returns, missing)
				else
					push!(returns, (current_val / start_val) - 1)
				end
			end
			return returns
		end

		for (name, history_df) in portfolio_histories
			if nrow(history_df) < 2
				;
				continue;
			end

			# Max Drawdown
			peak = -Inf;
			max_dd = 0.0
			for val in skipmissing(history_df.Value)
				peak = max(peak, val)
				max_dd = max(max_dd, (peak - val) / peak)
			end

			# Quarterly Returns for VaR/CVaR
			quarterly_returns = get_periodic_returns(history_df, 90)
			clean_quarterly_returns = filter(x -> !ismissing(x) && isfinite(x), quarterly_returns)

			# Annualized Stdev (using daily returns, trimmed)
			daily_returns = (history_df.Value[2:end] ./ history_df.Value[1:(end-1)]) .- 1
			clean_daily_returns = filter(x -> !ismissing(x) && isfinite(x), daily_returns)
			local stdev
			if length(clean_daily_returns) > 20
				lower_quantile = percentile(clean_daily_returns, 1)
				upper_quantile = percentile(clean_daily_returns, 99)
				trimmed_returns = filter(r -> lower_quantile <= r <= upper_quantile, clean_daily_returns)
				stdev = std(trimmed_returns) * sqrt(252)
			elseif !isempty(clean_daily_returns)
				stdev = std(clean_daily_returns) * sqrt(252)
			else
				stdev = 0.0
			end

			# VaR and CVaR using periodic (quarterly) returns
			var_95 = isempty(clean_quarterly_returns) ? 0.0 : -percentile(clean_quarterly_returns, 5)
			cvar_95 = isempty(clean_quarterly_returns) || var_95 == 0.0 ? 0.0 : -mean(filter(r -> r < -var_95, clean_quarterly_returns))

			# Annualized Return for Efficient Frontier and calculations
			num_years = (last(history_df.Date) - first(history_df.Date)).value / 365.25
			first_valid_idx = findfirst(!ismissing, history_df.Value)
			last_valid_idx = findlast(!ismissing, history_df.Value)
			local ann_return = 0.0
			if num_years > 0 && first_valid_idx !== nothing && last_valid_idx !== nothing
				first_val = history_df.Value[first_valid_idx]
				last_val = history_df.Value[last_valid_idx]
				if first_val != 0
					ann_return = ((last_val / first_val)^(1/num_years) - 1)
				end
			end

			bench_hist = portfolio_histories["Benchmark"]
			aligned_bench_hist = filter(r -> r.Date in history_df.Date, bench_hist)
			if nrow(aligned_bench_hist) < 2
				beta, r_squared = NaN, NaN
			else
				# Calculate Beta and R-squared using annual returns
				annual_returns = get_periodic_returns(history_df, 365)
				bench_annual_returns = get_periodic_returns(aligned_bench_hist, 365)

				clean_annual_returns = filter(x -> !ismissing(x) && isfinite(x), annual_returns)
				clean_bench_annual_returns = filter(x -> !ismissing(x) && isfinite(x), bench_annual_returns)

				local model_data
				min_len = min(length(clean_annual_returns), length(clean_bench_annual_returns))
				if min_len > 1
					model_data = DataFrame(Y = clean_annual_returns[1:min_len], X = clean_bench_annual_returns[1:min_len])
					try
						ols = lm(@formula(Y ~ X), model_data)
						beta = coef(ols)[2]
						r_squared = r2(ols)
					catch
						beta = NaN;
						r_squared = NaN
					end
				else
					beta = NaN;
					r_squared = NaN
				end
			end

			push!(metrics_results, (
				Strategy = name,
				AnnualizedReturn = ann_return * 100,
				MaxDD = max_dd * 100,
				Stdev = stdev * 100,
				VaR_95 = var_95 * 100,
				CVaR_95 = cvar_95 * 100,
				Beta = beta,
				R_Squared = r_squared * 100,
			))
		end
		metrics_df = DataFrame(metrics_results)

		# --- Calculate additional periodic metrics (HHI, IR) ---
		println("--- Calculating periodic HHI and Information Ratio...")
		periodic_metrics = []
		unique_dates = sort(unique(stats_df.Rebalance_Date))

		strategy_to_hpr_col = Dict(
			"PRISM_A" => :PRISM_A_HPR_Period, "PRISM_B" => :PRISM_B_HPR_Period,
			"Buy & Hold" => :BuyAndHold_HPR_Period, "Max Sharpe" => :MaxSharpe_HPR_Period,
			"Min Vol" => :MinVol_HPR_Period, "Inv Vol" => :InvVol_HPR_Period,
			"Max Return" => :MaxReturn_HPR_Period, "Equal-Weight" => :EqualWeight_HPR_Period,
			"HRP" => :HRP_HPR_Period, "HERC" => :HERC_HPR_Period, "NCO" => :NCO_HPR_Period,
			"Benchmark" => :Benchmark_HPR_Period,
		)

		for i in 1:(length(unique_dates)-1)
			start_date = unique_dates[i]
			period_stats = filter(r -> r.Rebalance_Date == start_date, stats_df)
			if isempty(period_stats)
				;
				continue;
			end
			bench_hpr = get(period_stats, 1, :Benchmark_HPR_Period)

			for (name, weights_df) in all_strategies
				hpr_col = get(strategy_to_hpr_col, name, nothing)
				if hpr_col === nothing || !hasproperty(period_stats, hpr_col)
					;
					continue;
				end

				strat_hpr = get(period_stats, 1, hpr_col)
				if ismissing(strat_hpr) || ismissing(bench_hpr)
					;
					continue;
				end

				weights_for_date = filter(r -> r.Date == start_date, weights_df)
				hhi = isempty(weights_for_date) ? 0.0 : sum(weights_for_date.Weight .^ 2)

				excess_return = strat_hpr - bench_hpr
				hist_strat_hpr = filter(!ismissing, getproperty(stats_df, hpr_col))
				hist_bench_hpr = filter(!ismissing, stats_df.Benchmark_HPR_Period)
				min_len = min(length(hist_strat_hpr), length(hist_bench_hpr))
				tracking_error = std(hist_strat_hpr[1:min_len] .- hist_bench_hpr[1:min_len])
				ir = tracking_error > 1e-8 ? excess_return / tracking_error : 0.0

				push!(periodic_metrics, (Date = start_date, Strategy = name, HHI = hhi, IR = ir))
			end
		end
		periodic_metrics_df = DataFrame(periodic_metrics)

		# --- Generate Plots ---
		println("--- Generating all plots...")

		strategy_order = sort(collect(keys(portfolio_histories)))
		p1 = plot(title = "Portfolio Value Over Time", xlabel = "Date", ylabel = "Value (\$)", legend = :topleft, size = (1200, 700))
		for name in strategy_order
			plot!(p1, portfolio_histories[name].Date, portfolio_histories[name].Value, label = name, color = strategy_colors[name])
		end

		# --- Metrics Table Plot (Transposed) ---
		function create_metrics_table_plot(metrics_df)
			strategy_order = ["Benchmark", "PRISM_A", "PRISM_B", "Max Sharpe", "Min Vol", "Inv Vol", "Max Return", "HRP", "HERC", "NCO", "Buy & Hold", "Equal-Weight"]
			metric_info = [
				(:AnnualizedReturn, "Ann. Return", "%.2f%%"), (:Stdev, "Ann. Stdev", "%.2f%%"),
				(:MaxDD, "Max Drawdown", "%.2f%%"), (:VaR_95, "VaR 95% (Q)", "%.2f%%"),
				(:CVaR_95, "CVaR 95% (Q)", "%.2f%%"), (:Beta, "Beta (Ann)", "%.2f"),
				(:R_Squared, "R² (Ann)", "%.2f%%"),
			]

			df_filtered = filter(row -> row.Strategy in strategy_order, metrics_df)
			# Reorder dataframe to match the desired strategy_order
			df_map = Dict(s => i for (i, s) in enumerate(df_filtered.Strategy))
			order_indices = [df_map[s] for s in strategy_order if haskey(df_map, s)]
			df = df_filtered[order_indices, :]


			num_rows = length(metric_info)
			num_cols = nrow(df)

			p = plot(grid = false, showaxis = false, xticks = false, yticks = false, legend = false,
				title = "Performance & Risk Metrics Summary", size = (1400, 150 + num_rows * 30),
				xlims = (0.5, num_cols + 1.5), ylims = (0.5, num_rows + 2))

			for (j, strat_name) in enumerate(df.Strategy)
				annotate!(p, j + 1, num_rows + 1.5, text(strat_name, :center, :bold, 9))
			end

			for (i, (_, metric_name, _)) in enumerate(metric_info)
				y_pos = num_rows - i + 1
				annotate!(p, 1, y_pos, text(metric_name, :left, :bold, 9))
			end

			for (j, strat_name) in enumerate(df.Strategy)
				row = first(filter(r -> r.Strategy == strat_name, eachrow(df)))
				for (i, (metric_sym, _, format_str)) in enumerate(metric_info)
					y_pos = num_rows - i + 1
					x_pos = j + 1
					val = row[metric_sym]
					formatted_val = ismissing(val) ? "N/A" : Printf.format(Printf.Format(format_str), val)
					annotate!(p, x_pos, y_pos, text(formatted_val, :center, 9))
				end
			end
			return p
		end
		p2 = create_metrics_table_plot(metrics_df)

		# --- Daily Return Distribution Subplots ---
		println("--- Generating Daily Return Distribution Plots ---")

		function semi_moment(returns, target, order)
			downside_returns = filter(r -> r < target, returns)
			if isempty(downside_returns)
				;
				return 0.0;
			end
			n = length(returns)
			sum_dev = sum((target .- r)^order for r in downside_returns)
			return sum_dev / n
		end

		plot_list = []
		bench_hist = portfolio_histories["Benchmark"]
		bench_daily_returns_raw = (bench_hist.Value[2:end] ./ bench_hist.Value[1:(end-1)]) .- 1
		bench_daily_returns = filter(x -> !ismissing(x) && isfinite(x), bench_daily_returns_raw)
		target_return = isempty(bench_daily_returns) ? 0.0 : mean(bench_daily_returns)

		for name in strategy_order
			hist_df = portfolio_histories[name]
			daily_returns_raw = (hist_df.Value[2:end] ./ hist_df.Value[1:(end-1)]) .- 1
			returns = filter(x -> !ismissing(x) && isfinite(x), daily_returns_raw)

			if isempty(returns) || length(returns) < 4
				;
				continue;
			end

			μ = mean(returns);
			med = median(returns);
			sk = skewness(returns)
			kurt = kurtosis(returns) - 3

			semi_dev = sqrt(semi_moment(returns, target_return, 2))
			semi_sk = (semi_dev > 1e-8) ? semi_moment(returns, target_return, 3) / (semi_dev^3) : 0.0
			semi_kurt = (semi_dev > 1e-8) ? semi_moment(returns, target_return, 4) / (semi_dev^4) : 0.0

			stat_label = (name == "Benchmark") ? @sprintf("Skew: %.2f\nExKurtosis: %.2f\nSemi-Skew: N/A\nSemi-Kurtosis: N/A", sk, kurt) :
						 @sprintf("Skew: %.2f\nExKurtosis: %.2f\nSemi-Skew: %.2f\nSemi-Kurtosis: %.2f", sk, kurt, semi_sk, semi_kurt)

			p_dist = density(returns, label = "", title = name, color = strategy_colors[name], fill = (0, 0.3, strategy_colors[name]),
				xticks = :none, yticks = :none, xlims = quantile(returns, [0.001, 0.999]))

			vline!(p_dist, [μ], linestyle = :dash, color = :black, label = "")
			vline!(p_dist, [med], linestyle = :dot, color = :blueviolet, label = "")

			y_range = ylims(p_dist);
			x_range = xlims(p_dist)
			annotate!(p_dist, x_range[1] + 0.05 * (x_range[2] - x_range[1]), y_range[2] * 0.95, text(stat_label, :left, :top, 8))
			x_offset = (x_range[2] - x_range[1]) * 0.03
			annotate!(p_dist, μ + x_offset, y_range[2]*0.9, text(@sprintf("Mean\n%.4f%%", μ * 100), :left, :top, 8, :black))
			annotate!(p_dist, med - x_offset, y_range[2]*0.6, text(@sprintf("Median\n%.4f%%", med * 100), :right, :top, 8, :blueviolet))

			push!(plot_list, p_dist)
		end
		p3 = plot(plot_list..., layout = (4, 3), size = (1200, 1200), plot_title = "Daily Return Distributions", bottom_margin = 20*Plots.px)


		# --- Efficient Frontier Plot ---
		println("--- Generating Efficient Frontier Plot ---")
		sim_start_date = minimum(h.Date[1] for h in values(portfolio_histories))
		RISK_FREE_RATE = get_average_risk_free_rate(sim_start_date)

		sp_optimize = pyimport("scipy.optimize")
		p4 = plot()
		try
			assets_for_ef = filter(c -> String(c) != "Date" && c in names(prices_df), names(prices_df))
			ef_prices = prices_df[!, [:Date; Symbol.(assets_for_ef)]]

			ef_returns_df = DataFrame(Date = ef_prices.Date[2:end])
			for col in assets_for_ef
				ef_returns_df[!, col] = (ef_prices[2:end, col] ./ ef_prices[1:(end-1), col]) .- 1
			end
			ef_returns_matrix = Matrix(ef_returns_df[!, assets_for_ef])
			ef_returns_matrix = coalesce.(ef_returns_matrix, 0.0)
			ef_returns_matrix[.!isfinite.(ef_returns_matrix)] .= 0.0

			μ_ann_historical = vec(mean(ef_returns_matrix, dims = 1) * 252)
			Σ_ann = cov(ef_returns_matrix) * 252
			num_assets_ef = length(assets_for_ef)

			p4 = plot(xlabel = "Annualized Volatility (%)", ylabel = "Annualized Return (%)",
				title = "Efficient Frontier & Strategy Performance", legend = :topleft,
				size = (1200, 800))

			function portfolio_volatility(weights, Σ)
				return sqrt(max(1e-12, weights' * Σ * weights))
			end

			initial_weights = fill(1.0 / num_assets_ef, num_assets_ef)
			bounds = [(0.0, 1.0) for _ in 1:num_assets_ef]
			constraints_gmv = (py"dict"(type = "eq", fun = w -> sum(w) - 1.0),)
			gmv_result = sp_optimize.minimize(w -> portfolio_volatility(w, Σ_ann),
				initial_weights, method = "SLSQP",
				bounds = bounds, constraints = constraints_gmv)

			gmv_weights = gmv_result["x"]
			gmv_vol = portfolio_volatility(gmv_weights, Σ_ann)
			gmv_ret = dot(gmv_weights, μ_ann_historical)

			frontier_points = []
			target_returns_frontier = range(gmv_ret, maximum(μ_ann_historical) * 1.2, length = 100)
			for target_return in target_returns_frontier
				constraints_ef = (py"dict"(type = "eq", fun = w -> sum(w) - 1.0),
					py"dict"(type = "eq", fun = w -> dot(w, μ_ann_historical) - target_return))
				opt_result = sp_optimize.minimize(w -> portfolio_volatility(w, Σ_ann), gmv_result["x"], method = "SLSQP", bounds = bounds, constraints = constraints_ef)
				if opt_result["success"]
					push!(frontier_points, (vol = opt_result["fun"], ret = target_return))
				end
			end

			if !isempty(frontier_points)
				sort!(frontier_points, by = x -> x.vol)
				frontier_vols = [p.vol * 100 for p in frontier_points]
				frontier_rets = [p.ret * 100 for p in frontier_points]
				plot!(p4, frontier_vols, frontier_rets, label = "Efficient Frontier", color = :blue, linewidth = 3)
			end

			tangency_sharpe(w) = -(dot(w, μ_ann_historical) - RISK_FREE_RATE) / portfolio_volatility(w, Σ_ann)
			cml_result = sp_optimize.minimize(tangency_sharpe, initial_weights, method = "SLSQP", bounds = bounds, constraints = constraints_gmv)

			if cml_result["success"]
				cml_weights = cml_result["x"]
				cml_ret = dot(cml_weights, μ_ann_historical)
				cml_vol = portfolio_volatility(cml_weights, Σ_ann)
				cml_slope = (cml_ret - RISK_FREE_RATE) / cml_vol
				cml(x) = (RISK_FREE_RATE + cml_slope * (x/100)) * 100
				plot!(p4, x -> cml(x), 0, maximum(m.Stdev for m in eachrow(metrics_df)) * 1.1,
					label = "Capital Market Line", color = :black, linestyle = :solid, linewidth = 2)
			end

			for row in eachrow(metrics_df)
				scatter!(p4, [row.Stdev], [row.AnnualizedReturn],
					marker = (:circle, 8, stroke(2.0, :black)),
					label = row.Strategy, color = strategy_colors[row.Strategy])
			end

		catch e
			println("WARNING: Could not generate Efficient Frontier plot. Error: $e")
			p4 = plot(title = "Efficient Frontier Data Not Available", legend = false, grid = false, showaxis = false)
		end

		# --- HHI and IR Plots ---
		hhi_ticks = [0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.0]
		p5_hhi = plot(title = "HHI by Period", xlabel = "Date", ylabel = "HHI (log scale)", legend = :outertopright, yaxis = :log10, yticks = (hhi_ticks, string.(hhi_ticks)))
		p5_ir = plot(title = "Information Ratio by Period", xlabel = "Date", ylabel = "Information Ratio", legend = :outertopright)

		unique_strategies_periodic = sort(unique(periodic_metrics_df.Strategy))

		for name in unique_strategies_periodic
			df_strat = filter(r -> r.Strategy == name, periodic_metrics_df)
			if !isempty(df_strat)
				plot!(p5_hhi, df_strat.Date, df_strat.HHI, label = name, color = strategy_colors[name], linewidth = 2.5)
			end
		end

		strategies_for_ir = filter(s -> s != "Benchmark", unique_strategies_periodic)
		for name in strategies_for_ir
			df_strat = filter(r -> r.Strategy == name, periodic_metrics_df)
			if !isempty(df_strat)
				plot!(p5_ir, df_strat.Date, df_strat.IR, label = name, color = strategy_colors[name], linewidth = 2.5)
			end
		end

		p5 = plot(p5_hhi, p5_ir, layout = (2, 1), size = (1000, 800), plot_title = "Portfolio Dynamics")

		println("--- Displaying Plots ---")
		display(p1)
		display(p2)
		display(p3)
		display(p4)
		display(p5)

		println("\n" * "="^60)
		println("              ANALYSIS & REPORTING COMPLETE")
		println("="^60 * "\n")
	end

	run_analysis_and_reporting()

end # END OF ANALYSIS MODE
