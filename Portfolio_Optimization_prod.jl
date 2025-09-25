## Environment Configuration
# Set the Python executable path for PyCall and configure the plotting backend.
ENV["GKSwstype"] = "100"

# --- Julia Package Loading ---
# Import all required Julia packages for the entire pipeline.
using Conda
using CSV
using DataFrames
using Dates
using Distances
using Graphs
using LinearAlgebra
using MLJ
using NetworkLayout
using NOMAD
using PlotlyJS
using Plots
using Printf
using ProgressMeter
using PyCall
using Random
using RollingFunctions
using ScikitLearn
using ShapML
using ShiftedArrays
using SimpleGraphs
using StatsBase
using Statistics
using StatsPlots
using XGBoost
using GLM
using StatsModels
using JSON # Added for state management
using JLD2

# --- Python Module Declarations ---
# Declare global constants for Python modules, initialized to PyNULL.
const sp = PyNULL()
const pso = PyNULL()
const np = PyNULL()
const sklearn_covariance = PyNULL()
const skl_preprocessing = PyNULL()
const optuna = PyNULL()
const skl_feature_selection = PyNULL()
const sp_optimize = PyNULL()
const linear_model = PyNULL()
const skl_decomposition = PyNULL()

# --- Global Constants & Paths ---
const epsilon = 1e-8
const TARGET_VARIABLE = "Information_Ratio"
const FORECAST_PLOTS_DIR = joinpath("Plots", "Forecast")
const CHECKPOINT_DIR = joinpath("Data", "Checkpoints")
const EVAL_STATS_DIR = joinpath("Data", "EvalStats")
const LOGS_DIR = "Logs"
const STANDALONE_STATE_FILE = joinpath(LOGS_DIR, "standalone_pipeline_state.json")
const STANDALONE_APSO_RESULTS = joinpath(CHECKPOINT_DIR, "standalone_apso_results.csv")

# === GLOBAL & DATA PARAMETERS ===
const BENCHMARK = "SPY"
const ENABLE_PLOTTING = true      # Master switch to enable/disable all plot generation
# This switch is used by the backtester. When true, the main() function below will not run.
const BACKTEST_MODE = false      # Default to standalone mode. The backtester will override this.
const RESTART_AFTER_QUARTER_EXIT_CODE = 10

# === STAGE 1: APSO PARAMETERS ===
const APSO_BOOTSTRAP_SAMPLES = 1000
const APSO_SWARM_PARTICLES = 300
const APSO_ITERATIONS = 100
const APSO_MULTI_STARTS = 20
const APSO_W_DECAY = 3.0
const APSO_C1_DECAY = 3.0
const APSO_C2_INCREASE = 1.0
const APSO_PARETO_PENALTIES = 0.0:0.1:1.0
const APSO_MAX_REFINEMENT_ATTEMPTS = 20
const APSO_MU_INITIAL = 0.1
const APSO_MU_INCREMENT = 0.15

# === STAGE 2: FEATURE ENGINEERING PARAMETERS ===
# --- Time Series ---
const FE_FEATURE_LAGS = 1:4
const FE_TARGET_LAGS = 1:2
const FE_ROLLING_WINDOWS = [2, 4]
# --- MI Selection ---
const FE_MI_THRESHOLD_STD_MULTIPLIER = 3.0
const FE_MI_FALLBACK_N_FEATURES = 50
# --- Polynomial & SULOV ---
const FE_POLYNOMIAL_DEGREE = 2
const FE_SULOV_CORR_THRESHOLD = 0.7
const FE_SULOV_PLOT_SEED_NODES = 50
const FE_SULOV_PLOT_K_NEIGHBORS = 5
# --- RFE (Recursive Feature Elimination) ---
const FE_OPTUNA_TRIALS = 100
const RFE_N_ROUNDS = 50
const RFE_N_ITERATIONS_PER_ROUND = 25
const RFE_N_FEATURES_PER_ITERATION = 1

# === STAGE 3: FORECASTING PARAMETERS ===
const FC_OPTIMIZATION_MODE = :DYNAMIC # Options: :MANUAL, :DYNAMIC
const FC_GRID_STEP = 0.5            # Step size for lambda penalties in DYNAMIC mode. Smaller is denser but slower.
const FC_RISK_METRIC = :TrackingError # Options: :TrackingError, :Stdev, :Beta, :AvgCorrelation
const FC_PREFERENCES = Dict("w_ir" => 0.4, "w_risk" => 0.3, "w_hhi" => 0.3)
const FC_PCA_VARIANCE_THRESHOLD = 0.95
const FC_RIDGE_ALPHA_START = -2  # Exponent for logspace start (10^-2)
const FC_RIDGE_ALPHA_END = 5     # Exponent for logspace end (10^4)
const FC_RIDGE_ALPHA_COUNT = 100 # Number of alphas to generate
const FC_RIDGE_CV_FOLDS = 5
const FC_NOMAD_MAX_EVALS = 1_000_000
const FC_BOOTSTRAP_SAMPLES = 1000

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# --- Initialization Function ---
function __init__()
	println("Initializing Python modules via PyCall...")
	copy!(sp, pyimport("scipy"))
	copy!(pso, pyimport("pyswarms.single"))
	copy!(np, pyimport("numpy"))
	copy!(sklearn_covariance, pyimport("sklearn.covariance"))
	copy!(skl_preprocessing, pyimport("sklearn.preprocessing"))
	copy!(optuna, pyimport("optuna"))
	copy!(skl_feature_selection, pyimport("sklearn.feature_selection"))
	copy!(sp_optimize, pyimport("scipy.optimize"))
	copy!(linear_model, pyimport("sklearn.linear_model"))
	copy!(skl_decomposition, pyimport("sklearn.decomposition"))
	println("Python modules initialized successfully.")
end

# --- Global Configurations & Execution ---
Random.seed!(42)
gr()
Plots.default(size = (1000, 600), gridalpha = 0.3, dpi = 300)
__init__()
optuna.logging.set_verbosity(optuna.logging.WARNING)

println("\n" * "="^60)
println("GLOBAL SETUP COMPLETE: All packages and modules loaded.")
println("="^60 * "\n")

## ------------------------------------------------------------------------------------------
## --------------------------------- STAGE 1: APSO and Analysis -----------------------------
## ------------------------------------------------------------------------------------------

# --- 1.1 APSO HELPER FUNCTIONS ---

function calculate_quarterly_return(log_returns::Matrix, weights::Vector)
	adjusted_weights = sum(abs.(weights)) > 0 ? weights ./ 100.0 : zeros(length(weights))
	portfolio_weekly_returns = log_returns * adjusted_weights
	cumulative_quarterly_log_return = sum(portfolio_weekly_returns)
	return exp(cumulative_quarterly_log_return) - 1
end

function calculate_shrunk_tracking_error(weights::Vector, shrunk_excess_return_cov_matrix::Matrix)
	if sum(abs.(weights)) == 0
		return 1e-8
	end
	w = weights ./ 100.0
	portfolio_variance = w' * shrunk_excess_return_cov_matrix * w
	return sqrt(abs(portfolio_variance)) + 1e-8
end

function calculate_transaction_cost(current_weights::Vector, previous_weights::Vector; buy_cost = 0.0005, sell_cost = 0.0005)
	w_curr = current_weights ./ 100.0
	w_prev = previous_weights ./ 100.0
	delta_w = w_curr - w_prev
	buys = sum(delta_w[delta_w .> 0], init = 0.0)
	sells = sum(abs.(delta_w[delta_w .< 0]), init = 0.0)
	return (buys * buy_cost) + (sells * sell_cost)
end

function portfolio_objective_function(weights::Vector, log_returns, benchmark_log_returns, shrunk_cov_matrix, mu, diversification_penalty, previous_weights, num_stocks)
	gross_portfolio_return = calculate_quarterly_return(log_returns, weights)
	turnover_cost = calculate_transaction_cost(weights, previous_weights)
	net_portfolio_return = gross_portfolio_return - turnover_cost
	benchmark_return = exp(sum(benchmark_log_returns)) - 1
	excess_return = net_portfolio_return - benchmark_return
	tracking_error = calculate_shrunk_tracking_error(weights, shrunk_cov_matrix)
	hhi = sum((weights ./ 100.0) .^ 2)
	hhi_normalized = num_stocks > 1 ? (hhi - (1 / num_stocks)) / (1 - (1 / num_stocks)) : 1.0
	hhi_penalty = hhi_normalized
	cost = -excess_return + mu * tracking_error + diversification_penalty * hhi_penalty
	return cost
end

function adaptive_inertia_weight(iteration, max_iter; w_max = 0.9, w_min = 0.4, decay_rate = 3.0)
	if max_iter <= 1
		return w_min
	end
	t_norm = (iteration - 1) / (max_iter - 1)
	return w_min + (w_max - w_min) * exp(-decay_rate * t_norm)
end

function adaptive_cognitive_weight(iteration, max_iter; c1_initial = 2.5, c1_final = 0.5, decay_rate = 3.0)
	if max_iter <= 1
		return c1_final
	end
	t_norm = (iteration - 1) / (max_iter - 1)
	return c1_final + (c1_initial - c1_final) * exp(-decay_rate * t_norm)
end

function delayed_adaptive_social_weight(iteration, max_iter; c2_initial = 0.5, c2_final = 2.5, delay_fraction = 0.7)
	if max_iter <= 1
		return c2_initial
	end
	if (iteration - 1) < max_iter * delay_fraction
		return c2_initial
	else
		adaptation_total_iters = max_iter * (1.0 - delay_fraction)
		if adaptation_total_iters <= 0
			return c2_final
		end
		adaptation_current_iter = (iteration - 1) - (max_iter * delay_fraction)
		t_norm_adaptation = adaptation_current_iter / adaptation_total_iters
		return c2_initial + (c2_final - c2_initial) * t_norm_adaptation
	end
end

function winsorize_series(series::Vector; lower_percentile = 1, upper_percentile = 99)
	if isempty(series)
		return series
	end
	finite_series = filter(isfinite, series)
	if isempty(finite_series)
		return series
	end
	lower_bound, upper_bound = percentile(finite_series, [lower_percentile, upper_percentile])
	return clamp.(series, lower_bound, upper_bound)
end

function create_pso_objective_function(log_returns_quarterly_slice, benchmark_log_returns_quarterly_slice,
	shrunk_cov_matrix, num_stocks; mu = 1.0, diversification_penalty = 0.1, previous_weights = nothing)
	function pso_objective(particles)
		costs = zeros(size(particles, 1))
		for (i, p_py) in enumerate(eachrow(particles))
			p = convert(Vector{Float64}, p_py)
			abs_sum = sum(abs.(p))
			weights = abs_sum > 0 ? (p ./ abs_sum) .* 100.0 : zeros(length(p))
			costs[i] = portfolio_objective_function(weights, log_returns_quarterly_slice, benchmark_log_returns_quarterly_slice, shrunk_cov_matrix, mu, diversification_penalty, previous_weights, num_stocks)
		end
		return costs
	end
	return pso_objective
end

function multi_start_pso(objective_function, bounds, n_particles, dimensions, max_iter, n_starts,
	w_decay_rate, c1_decay_rate, c2_increase_rate)
	best_costs_all_starts = Float64[]
	best_positions_all_starts = Vector{Float64}[]
	all_runs_cost_history = []
	w_initial, w_final = 0.9, 0.4
	c1_initial, c1_final = 2.5, 0.5
	c2_initial, c2_final = 0.5, 2.5

	for start in 1:n_starts
		options = py"dict"(c1 = c1_initial, c2 = c2_initial, w = w_initial)
		optimizer = pso.GlobalBestPSO(n_particles = n_particles, dimensions = dimensions, options = options, bounds = bounds)
		cost_history = Float64[]

		for i in 1:max_iter
			w = adaptive_inertia_weight(i, max_iter; w_max = w_initial, w_min = w_final, decay_rate = w_decay_rate)
			c1 = adaptive_cognitive_weight(i, max_iter; c1_initial = c1_initial, c1_final = c1_final, decay_rate = c1_decay_rate)
			c2 = delayed_adaptive_social_weight(i, max_iter; c2_initial = c2_initial, c2_final = c2_final)

			optimizer.swarm.options["w"] = w
			optimizer.swarm.options["c1"] = c1
			optimizer.swarm.options["c2"] = c2

			optimizer.optimize(objective_function, iters = 1, verbose = false)
			push!(cost_history, optimizer.swarm.best_cost)
		end
		push!(best_costs_all_starts, optimizer.swarm.best_cost)
		push!(best_positions_all_starts, convert(Vector{Float64}, optimizer.swarm.best_pos))
		push!(all_runs_cost_history, cost_history)
	end

	best_idx = argmin(best_costs_all_starts)
	return best_positions_all_starts[best_idx], best_costs_all_starts, all_runs_cost_history
end

function perform_bootstrap_analysis_on_weights(log_returns, benchmark_returns, fixed_weights, n_bootstrap_samples)
	bootstrap_ir_samples = Float64[]
	original_bench_return = exp(sum(benchmark_returns)) - 1
	n_obs = size(log_returns, 1)

	for i in 1:n_bootstrap_samples
		indices = rand(1:n_obs, n_obs)
		bootstrap_log_ret = log_returns[indices, :]
		bootstrap_bench_ret = benchmark_returns[indices, :]
		bootstrap_excess_returns = bootstrap_log_ret .- bootstrap_bench_ret

		if all(var(bootstrap_excess_returns, dims = 1, corrected = false) .< 1e-10)
			bootstrap_shrunk_cov = zeros(size(log_returns, 2), size(log_returns, 2))
		else
			lw_bootstrap = sklearn_covariance.LedoitWolf()
			lw_bootstrap.fit(bootstrap_excess_returns)
			bootstrap_shrunk_cov = convert(Matrix{Float64}, lw_bootstrap."covariance_")
		end

		sim_port_ret = calculate_quarterly_return(bootstrap_log_ret, fixed_weights)
		sim_te = calculate_shrunk_tracking_error(fixed_weights, bootstrap_shrunk_cov)
		sim_ir = sim_te > 1e-8 ? (sim_port_ret - original_bench_return) / sim_te : -Inf
		push!(bootstrap_ir_samples, sim_ir)
	end
	return bootstrap_ir_samples
end

function center_plot_around_zero!(data_list)
	combined_data = vcat([d for d in data_list if d !== nothing && !isempty(d)]...)
	finite_data = filter(isfinite, combined_data)
	if !isempty(finite_data)
		max_abs = maximum(abs, finite_data)
		if max_abs > 0
			Plots.xlims!(-max_abs * 1.1, max_abs * 1.1)
		end
	end
end

function create_summary_distribution_plot(data, title_suffix, color)
	numeric_data = filter(x -> isa(x, Number) && isfinite(x), data)
	plt = Plots.plot(title = "Distribution of $(title_suffix)", xlabel = "Value", ylabel = "Density", legend = :topleft, legend_background_color = :white, legend_foreground_color = :black)
	if !isempty(numeric_data)
		d_mean = mean(numeric_data)
		d_median = median(numeric_data)
		d_std = std(numeric_data)
		d_skew = skewness(numeric_data)
		d_kurt = kurtosis(numeric_data)
		d_n = length(numeric_data)

		Plots.histogram!(plt, numeric_data, bins = 20, normalize = :pdf, alpha = 0.7, label = "Distribution", color = color)
		Plots.vline!(plt, [d_mean], color = :black, style = :dash, linewidth = 2, label = "Mean")
		Plots.vline!(plt, [d_median], color = :grey, style = :dot, linewidth = 2, label = "Median")

		Plots.scatter!(plt, [NaN], [NaN], label = "", markeralpha = 0, color = :transparent) # Spacer
		Plots.scatter!(plt, [NaN], [NaN], label = @sprintf("Mean: %.3f", d_mean), markeralpha = 0, color = :transparent)
		Plots.scatter!(plt, [NaN], [NaN], label = @sprintf("Median: %.3f", d_median), markeralpha = 0, color = :transparent)
		Plots.scatter!(plt, [NaN], [NaN], label = @sprintf("Std Dev: %.3f", d_std), markeralpha = 0, color = :transparent)
		Plots.scatter!(plt, [NaN], [NaN], label = @sprintf("Skew: %.3f", d_skew), markeralpha = 0, color = :transparent)
		Plots.scatter!(plt, [NaN], [NaN], label = @sprintf("Kurtosis: %.3f", d_kurt), markeralpha = 0, color = :transparent)
		Plots.scatter!(plt, [NaN], [NaN], label = @sprintf("N: %d", d_n), markeralpha = 0, color = :transparent)
	end
	center_plot_around_zero!([numeric_data])
	return plt
end

# --- 1.2 MAIN APSO ORCHESTRATOR ---
function run_apso_stage(market_quarterly, prices_weekly;
	n_bootstrap_samples, set_particles, set_iters_pso, set_starts,
	pso_w_decay_rate, pso_c1_decay_rate, pso_c2_increase_rate,
	diversification_penalty_grid, benchmark,
	state_file_path = nothing, output_csv_path = nothing)

	# --- Initial Data Prep ---
	if isempty(prices_weekly) || isempty(market_quarterly)
		return nothing, nothing, 0, 0
	end
	if !(eltype(prices_weekly.Date) <: Date)
		prices_weekly.Date = Date.(prices_weekly.Date)
	end
	if !(eltype(market_quarterly.Date) <: Date)
		market_quarterly.Date = Date.(market_quarterly.Date)
	end

	benchmark_col_name = benchmark
	if benchmark_col_name ∉ names(prices_weekly)
		println("ERROR: Benchmark '$(benchmark_col_name)' not found.")
		return nothing, nothing, 0, 0
	end
	master_stock_cols = [col for col in names(prices_weekly) if col != benchmark_col_name && col != "Date"]
	if isempty(master_stock_cols)
		return nothing, nothing, 0, 0
	end

	all_results_df, all_pareto_results_df = DataFrame(), DataFrame()
	quarters_to_process = unique(firstdayofquarter.(market_quarterly.Date))
	last_completed_quarter = Date(1900, 1, 1)
	local state = Dict()
	apso_loop_executed = false

	# --- Resumability Logic ---
	current_state_file = BACKTEST_MODE ? state_file_path : STANDALONE_STATE_FILE
	current_output_csv = BACKTEST_MODE ? output_csv_path : STANDALONE_APSO_RESULTS
	pareto_checkpoint_path = joinpath(CHECKPOINT_DIR, "apso_pareto_quarterly_stats.csv")

	if current_state_file !== nothing && current_output_csv !== nothing
		state = isfile(current_state_file) ? JSON.parsefile(current_state_file) : Dict()
		last_completed_quarter_str = get(get(state, "apso", Dict()), "last_completed_quarter", "1900-01-01")
		last_completed_quarter = Date(last_completed_quarter_str)

		if isfile(current_output_csv) && last_completed_quarter > Date(1900, 1, 1)
			println("--- Found existing partial APSO results. Loading from $(current_output_csv) ---")
			all_results_df = CSV.read(current_output_csv, DataFrame)
			all_results_df.Date = Date.(all_results_df.Date)
		end
		if isfile(pareto_checkpoint_path) && last_completed_quarter > Date(1900, 1, 1)
			println("--- Found existing Pareto checkpoint data. Loading from $(pareto_checkpoint_path) ---")
			all_pareto_results_df = CSV.read(pareto_checkpoint_path, DataFrame)
			all_pareto_results_df.Date = Date.(all_pareto_results_df.Date)
		end

		quarters_to_process = filter(q -> q > last_completed_quarter, quarters_to_process)
		if !isempty(all_results_df) && last_completed_quarter > Date(1900, 1, 1)
			println("--- Resuming from quarter: $(isempty(quarters_to_process) ? "N/A" : first(quarters_to_process)) ---")
		end
	end

	# --- Main Loop Setup ---
	MIN_WEEKLY_OBSERVATIONS_PER_QUARTER = 10
	successful_quarters_count, unsuccessful_quarters_count = 0, 0
	previous_quarter_weights = zeros(length(master_stock_cols))
	plot_data_aggregator = Dict("all_costs" => [], "all_histories" => [], "all_bootstrap_samples" => [])

	function save_state_and_results(current_quarter_result::DataFrame, current_pareto_df::DataFrame, quarter_start::Date)
		if current_state_file !== nothing && current_output_csv !== nothing
			CSV.write(current_output_csv, current_quarter_result; append = isfile(current_output_csv) && filesize(current_output_csv) > 0)
			CSV.write(pareto_checkpoint_path, current_pareto_df; append = isfile(pareto_checkpoint_path) && filesize(pareto_checkpoint_path) > 0)
			get!(state, "apso", Dict())["last_completed_quarter"] = string(quarter_start)
			open(current_state_file, "w") do f
				JSON.print(f, state, 4)
			end
		end
	end

	prog = Progress(length(quarters_to_process), "APSO Pareto Calculation:")
	for (i, quarter_start) in enumerate(quarters_to_process)
		apso_loop_executed = true
		ProgressMeter.update!(prog, i, showvalues = [(:quarter, quarter_start)])

		quarter_end = lastdayofquarter(quarter_start)
		prices_in_quarter = filter(row -> quarter_start <= row.Date <= quarter_end, prices_weekly)
		tradable_stocks_in_quarter = [stock for stock in master_stock_cols if hasproperty(prices_in_quarter, Symbol(stock)) && count(!ismissing, prices_in_quarter[!, Symbol(stock)]) >= MIN_WEEKLY_OBSERVATIONS_PER_QUARTER]

		if length(tradable_stocks_in_quarter) < 2
			println("Skipping quarter $(quarter_start): insufficient tradable assets.")
			blank_weights = zeros(length(master_stock_cols))
			weights_matrix = reshape(blank_weights, 1, :)
			optimal_weights_df = DataFrame(weights_matrix, Symbol.(master_stock_cols .* "_Weight"))
			optimal_weights_df.Date = [quarter_start]
			ir_df = DataFrame(Date = [quarter_start], Information_Ratio = [0.0])
			ci_df = DataFrame(Date = [quarter_start], IR_CI_Lower = [0.0], IR_CI_Upper = [0.0])
			current_quarter_result = innerjoin(ir_df, optimal_weights_df, ci_df, on = :Date)
			market_data_for_join = filter(row -> row.Date == quarter_start, select(market_quarterly, Not(r"_Weight$")))
			if !isempty(market_data_for_join)
				current_quarter_result = innerjoin(market_data_for_join, current_quarter_result, on = :Date)
			end

			blank_pareto_df = DataFrame(Date = quarter_start, Penalty = 0.0, IR = 0.0, HHI = 0.0, CI_Lower = 0.0, CI_Upper = 0.0)
			save_state_and_results(current_quarter_result, blank_pareto_df, quarter_start)
			println("--- APSO quarter for $(quarter_start) complete. Exiting with code $(RESTART_AFTER_QUARTER_EXIT_CODE) to signal restart. ---")
			exit(RESTART_AFTER_QUARTER_EXIT_CODE)
		end

		log_ret(p) = [NaN; log.(p[2:end] ./ p[1:(end-1)])]
		clean_val(x) = !isfinite(x) ? 0.0 : x
		stock_returns_quarter = DataFrame([col => winsorize_series(clean_val.(log_ret(coalesce.(prices_in_quarter[:, Symbol(col)], NaN)))) for col in tradable_stocks_in_quarter])
		benchmark_log_returns_in_quarter = winsorize_series(clean_val.(log_ret(coalesce.(prices_in_quarter[:, Symbol(benchmark_col_name)], NaN))))
		log_returns_in_quarter = Matrix(stock_returns_quarter)
		num_stocks = size(log_returns_in_quarter, 2)

		master_idx_map = Dict(name => idx for (idx, name) in enumerate(master_stock_cols))
		aligned_previous_weights = [get(master_idx_map, s, 0) > 0 ? previous_quarter_weights[master_idx_map[s]] : 0.0 for s in tradable_stocks_in_quarter]

		excess_returns = log_returns_in_quarter .- benchmark_log_returns_in_quarter
		lw = sklearn_covariance.LedoitWolf()
		lw.fit(excess_returns)
		quarterly_shrunk_cov_matrix = convert(Matrix{Float64}, lw."covariance_")

		quarterly_pareto_results = []
		for penalty_val in diversification_penalty_grid
			mu_val = APSO_MU_INITIAL
			best_fallback_solution = Dict("weights" => zeros(num_stocks), "ir" => 0.0, "ci" => (-Inf, -Inf))
			for attempt in 1:APSO_MAX_REFINEMENT_ATTEMPTS
				try
					pso_bounds = (fill(-1.0, num_stocks), fill(1.0, num_stocks))
					pso_objective_func =
						create_pso_objective_function(log_returns_in_quarter, benchmark_log_returns_in_quarter, quarterly_shrunk_cov_matrix, num_stocks; mu = mu_val, diversification_penalty = penalty_val, previous_weights = aligned_previous_weights)

					pso_best_solution, pso_final_costs, pso_cost_histories = multi_start_pso(pso_objective_func, pso_bounds, set_particles, num_stocks, set_iters_pso, set_starts, pso_w_decay_rate, pso_c1_decay_rate, pso_c2_increase_rate)
					append!(plot_data_aggregator["all_costs"], pso_final_costs)
					append!(plot_data_aggregator["all_histories"], pso_cost_histories)

					x0_sum = sum(abs.(pso_best_solution))
					x0 = x0_sum > 0 ? (pso_best_solution ./ x0_sum) .* 100.0 : zeros(num_stocks)
					slsqp_bounds = [(-100.0, 100.0) for _ in 1:num_stocks]
					constraints = py"dict"(type = "eq", fun = (w -> sum(abs.(w)) - 100.0))
					options_slsqp = py"dict"(disp = false, ftol = 1e-9, maxiter = 200)
					scipy_obj_wrapper(w) = portfolio_objective_function(convert(Vector{Float64}, w), log_returns_in_quarter, benchmark_log_returns_in_quarter, quarterly_shrunk_cov_matrix, mu_val, penalty_val, aligned_previous_weights, num_stocks)
					slsqp_result = sp.optimize.minimize(scipy_obj_wrapper, x0; method = "SLSQP", bounds = slsqp_bounds, constraints = constraints, options = options_slsqp)
					final_optimal_weights_unrounded = slsqp_result["success"] ? convert(Vector{Float64}, slsqp_result["x"]) : x0

					bootstrap_ir_samples = perform_bootstrap_analysis_on_weights(log_returns_in_quarter, benchmark_log_returns_in_quarter, final_optimal_weights_unrounded, n_bootstrap_samples)

					ci_lower, ci_upper = percentile(bootstrap_ir_samples, [2.5, 97.5])
					gross_port_return = calculate_quarterly_return(log_returns_in_quarter, final_optimal_weights_unrounded)
					bench_return = exp(sum(benchmark_log_returns_in_quarter)) - 1
					track_error = calculate_shrunk_tracking_error(final_optimal_weights_unrounded, quarterly_shrunk_cov_matrix)
					information_ratio = track_error > 1e-8 ? (gross_port_return - bench_return) / track_error : 0.0
					if ci_lower > best_fallback_solution["ci"][1]
						best_fallback_solution = Dict("weights" => final_optimal_weights_unrounded, "ir" => information_ratio, "ci" => (ci_lower, ci_upper))
					end
					if ci_lower > 0
						append!(plot_data_aggregator["all_bootstrap_samples"], bootstrap_ir_samples)
						break
					end
					mu_val += APSO_MU_INCREMENT
				catch e
					println("CRITICAL ERROR in APSO for quarter $(quarter_start): $e")
					continue
				end
			end
			final_weights_unrounded_for_penalty = best_fallback_solution["weights"]
			hhi = sum((final_weights_unrounded_for_penalty ./ 100.0) .^ 2)
			push!(
				quarterly_pareto_results,
				Dict("Date"=>quarter_start, "Penalty"=>penalty_val, "IR"=>best_fallback_solution["ir"], "HHI"=>hhi, "CI_Lower"=>best_fallback_solution["ci"][1], "CI_Upper"=>best_fallback_solution["ci"][2], "Weights"=>final_weights_unrounded_for_penalty),
			)
		end

		if isempty(quarterly_pareto_results)
			println("WARNING: No valid Pareto results for quarter $(quarter_start). Saving blank and restarting.")
			best_result_for_quarter = Dict("IR" => 0.0, "CI_Lower" => 0.0, "CI_Upper" => 0.0, "Weights" => zeros(num_stocks))
		else
			best_result_for_quarter = quarterly_pareto_results[argmax([res["CI_Lower"] for res in quarterly_pareto_results])]
		end

		final_ir = best_result_for_quarter["IR"]
		final_ci = (best_result_for_quarter["CI_Lower"], best_result_for_quarter["CI_Upper"])
		final_weights = best_result_for_quarter["Weights"]
		if final_ci[1] > 0
			successful_quarters_count += 1
		else
			unsuccessful_quarters_count += 1
		end

		full_quarter_weights = zeros(length(master_stock_cols))
		for (idx, stock_name) in enumerate(tradable_stocks_in_quarter)
			full_quarter_weights[master_idx_map[stock_name]] = final_weights[idx]
		end
		previous_quarter_weights = full_quarter_weights

		weights_matrix = reshape(full_quarter_weights, 1, :)
		optimal_weights_df = DataFrame(weights_matrix, Symbol.(master_stock_cols .* "_Weight"))
		optimal_weights_df.Date = [quarter_start]
		ir_df = DataFrame(Date = [quarter_start], Information_Ratio = [final_ir])
		ci_df = DataFrame(Date = [quarter_start], IR_CI_Lower = [final_ci[1]], IR_CI_Upper = [final_ci[2]])
		current_quarter_result = innerjoin(ir_df, optimal_weights_df, on = :Date)
		market_data_for_join = filter(row -> row.Date == quarter_start, select(market_quarterly, Not(r"_Weight$")))
		if !isempty(market_data_for_join)
			current_quarter_result = innerjoin(market_data_for_join, current_quarter_result, on = :Date)
		end
		current_quarter_result = innerjoin(current_quarter_result, ci_df, on = :Date)

		current_pareto_df = DataFrame(
			Date = [res["Date"] for res in quarterly_pareto_results],
			Penalty = [res["Penalty"] for res in quarterly_pareto_results],
			IR = [res["IR"] for res in quarterly_pareto_results],
			HHI = [res["HHI"] for res in quarterly_pareto_results],
			CI_Lower = [res["CI_Lower"] for res in quarterly_pareto_results],
			CI_Upper = [res["CI_Upper"] for res in quarterly_pareto_results],
		)

		all_results_df = vcat(all_results_df, current_quarter_result)
		all_pareto_results_df = vcat(all_pareto_results_df, current_pareto_df)

		save_state_and_results(current_quarter_result, current_pareto_df, quarter_start)
		println("--- APSO quarter for $(quarter_start) complete. Exiting with code $(RESTART_AFTER_QUARTER_EXIT_CODE) to signal restart. ---")
		exit(RESTART_AFTER_QUARTER_EXIT_CODE)
	end
	ProgressMeter.finish!(prog)

	if ENABLE_PLOTTING && !isempty(all_results_df) && apso_loop_executed
		println("\n", "="^25, " GENERATING AND SAVING APSO PLOTS ", "="^25)
		plot_dir = joinpath(pwd(), "Plots", "APSO")
		mkpath(plot_dir)
		if !isempty(all_pareto_results_df)
			avg_pareto = combine(groupby(all_pareto_results_df, :Penalty), :CI_Lower => mean => :Avg_CI_Lower, :HHI => mean => :Avg_HHI)
			sort!(avg_pareto, :Penalty)
			frontier_df = DataFrame(Penalty = Float64[], Avg_CI_Lower = Float64[], Avg_HHI = Float64[])
			max_ci_so_far = -Inf
			for row in eachrow(avg_pareto)
				if row.Avg_CI_Lower > max_ci_so_far
					push!(frontier_df, row)
					max_ci_so_far = row.Avg_CI_Lower
				end
			end
			p_pareto = plot(frontier_df.Avg_HHI, frontier_df.Avg_CI_Lower, title = "Average Pareto Frontier", xlabel = "Average HHI", ylabel = "Average Lower CI Bound", legend = false, lw = 3, marker = :circle)
			scatter!(p_pareto, avg_pareto.Avg_HHI, avg_pareto.Avg_CI_Lower, color = :grey, alpha = 0.6, marker = :x)
			hline!(p_pareto, [0], lw = 1.5, ls = :dash, color = :red)
			savefig(p_pareto, joinpath(plot_dir, "average_pareto_frontier.png"))
			println("Saved average Pareto frontier plot.")
		end
		weight_columns = [col for col in names(all_results_df) if endswith(string(col), "_Weight")]
		weights_only_df = all_results_df[:, weight_columns]
		plot_df = Matrix(weights_only_df)
		assets = [replace(string(c), "_Weight" => "") for c in weight_columns]
		quarters = string.(all_results_df.Date)
		custom_cmap = cgrad([:lightgreen, :green, :yellow, :orange, :red], [0.0, 0.25, 0.50, 0.75, 1.0])
		h1 = Plots.heatmap(quarters, assets, plot_df'; c = custom_cmap, clims = (0, 100), title = "Portfolio Composition (Custom Colors)", size = (2000, 1600), xrotation = 45, yticks = (1:length(assets), assets), ytickfontsize = 6)
		Plots.savefig(h1, joinpath(plot_dir, "heatmap_composition_custom.png"))
		plot_df_nan = replace(plot_df, 0 => NaN)
		h2 = Plots.heatmap(quarters, assets, plot_df_nan'; c = :viridis, clims = (0, 100), title = "Portfolio Composition (Zeros Hidden)", size = (2000, 1600), xrotation = 45, yticks = (1:length(assets), assets), ytickfontsize = 6)
		Plots.savefig(h2, joinpath(plot_dir, "heatmap_composition_standard.png"))
		weights_frac = weights_only_df ./ 100.0
		hhi = sum(Matrix(weights_frac .^ 2); dims = 2)[:]
		num_positions = sum(Matrix(weights_only_df .> 0.01); dims = 2)[:]
		num_positions[num_positions .== 0] .= 1
		eq_w_hhi = 1 ./ num_positions
		h3 = Plots.plot(quarters, hhi; marker = :o, label = "Portfolio HHI", title = "Portfolio Concentration (HHI) Over Time", size = (1200, 700), xrotation = 45, legend = :topright)
		Plots.plot!(h3, quarters, eq_w_hhi; marker = :x, style = :dash, c = :red, label = "Equal-Weight HHI (for context)")
		Plots.annotate!(h3, (0.02, 0.02), Plots.text("Average HHI: $(round(mean(hhi), digits=3))", :left, :bottom, 12, "white"), :axes)
		Plots.savefig(h3, joinpath(plot_dir, "hhi_over_time.png"))

		println("\n--- Generating Final Summary Subplots ---")
		successful_results = filter(row -> row.CI_Lower > 0, all_pareto_results_df)
		unsuccessful_results = filter(row -> row.CI_Lower <= 0, all_pareto_results_df)

		p_ir_s = create_summary_distribution_plot(successful_results.IR, "IRs (Successful)", :green)
		p_ir_u = create_summary_distribution_plot(unsuccessful_results.IR, "IRs (Unsuccessful)", :red)
		p_ci_l_s = create_summary_distribution_plot(successful_results.CI_Lower, "Lower CI (Successful)", :darkgreen)
		p_ci_l_u = create_summary_distribution_plot(unsuccessful_results.CI_Lower, "Lower CI (Unsuccessful)", :darkred)
		p_ci_u_s = create_summary_distribution_plot(successful_results.CI_Upper, "Upper CI (Successful)", :purple)
		p_ci_u_u = create_summary_distribution_plot(unsuccessful_results.CI_Upper, "Upper CI (Unsuccessful)", :orange)
		final_layout = Plots.plot(p_ir_s, p_ir_u, p_ci_l_s, p_ci_l_u, p_ci_u_s, p_ci_u_u; layout = (3, 2), size = (1600, 1800), plot_title = "Summary Distributions")
		Plots.savefig(final_layout, joinpath(plot_dir, "summary_distributions.png"))

		# --- Generate and Save NEW plots ---
		println("--- Generating and saving detailed optimization plots ---")

		# 1. PSO Costs Distribution Plot
		if !isempty(plot_data_aggregator["all_costs"])
			all_costs_flat = vcat(plot_data_aggregator["all_costs"]...)
			p_costs_dist = histogram(all_costs_flat, normalize = :pdf, bins = 30, label = "Cost Distribution",
				title = "Distribution of Best PSO Costs Across All Runs",
				xlabel = "Cost", ylabel = "Density", legend = :topright, alpha = 0.7)
			vline!(p_costs_dist, [minimum(all_costs_flat)], color = :red, style = :dash, lw = 2,
				label = @sprintf("Overall Best Cost: %.4f", minimum(all_costs_flat)))
			savefig(p_costs_dist, joinpath(plot_dir, "pso_costs_distribution.png"))
			println("Saved PSO costs distribution plot.")
		end

		# 2. PSO Convergence Plot
		if !isempty(plot_data_aggregator["all_histories"])
			histories = plot_data_aggregator["all_histories"]
			max_len = maximum(length.(histories))
			padded_histories = map(h -> vcat(h, fill(h[end], max_len - length(h))), histories)
			history_matrix = hcat(padded_histories...)'
			mean_conv = mean(history_matrix, dims = 1)[:]
			std_conv = std(history_matrix, dims = 1)[:]

			p_conv = plot(1:max_len, mean_conv, ribbon = std_conv, fillalpha = 0.2, lw = 2,
				label = "Mean Cost", color = :blue,
				title = "Average PSO Convergence Across All Attempts & Runs",
				xlabel = "Iteration", ylabel = "Cost")
			plot!(p_conv, 1:max_len, mean_conv, label = "", color = :blue, lw = 2)
			savefig(p_conv, joinpath(plot_dir, "pso_convergence.png"))
			println("Saved PSO convergence plot.")
		end

		# 3. Bootstrap IR Distribution
		if !isempty(plot_data_aggregator["all_bootstrap_samples"])
			all_samples = vcat(plot_data_aggregator["all_bootstrap_samples"]...)
			ir_mean = mean(all_samples)
			ir_skew = skewness(all_samples)
			ir_kurt = kurtosis(all_samples)
			ci_l, ci_u = percentile(all_samples, [2.5, 97.5])

			p_bootstrap = histogram(all_samples, normalize = :pdf, bins = 50, label = "IR Distribution",
				title = @sprintf("Bootstrap IR Distribution (Skew: %.2f, Kurtosis: %.2f)", ir_skew, ir_kurt),
				xlabel = "Information Ratio", ylabel = "Density", legend = :topright, alpha = 0.7)
			vline!(p_bootstrap, [ir_mean], style = :dash, color = :red, lw = 2, label = @sprintf("Mean IR: %.4f", ir_mean))
			vline!(p_bootstrap, [ci_l], style = :dash, color = :green, lw = 2, label = "2.5% CI")
			vline!(p_bootstrap, [ci_u], style = :dash, color = :green, lw = 2, label = "97.5% CI")
			savefig(p_bootstrap, joinpath(plot_dir, "bootstrap_ir_distribution.png"))
			println("Saved aggregate bootstrap IR distribution plot.")
		end

		function get_stats(data, name)
			numeric_data = filter(x -> isa(x, Number) && isfinite(x), data)
			if isempty(numeric_data)
				return DataFrame(Metric = name, Mean = NaN, Median = NaN, StdDev = NaN, Skewness = NaN, Kurtosis = NaN, N = 0)
			end
			d_mean, d_median, d_std, d_skew, d_kurt, d_n = mean(numeric_data), median(numeric_data), std(numeric_data), skewness(numeric_data), kurtosis(numeric_data), length(numeric_data)
			return DataFrame(Metric = name, Mean = d_mean, Median = d_median, StdDev = d_std, Skewness = d_skew, Kurtosis = d_kurt, N = d_n)
		end
		stats_df = vcat(
			get_stats(successful_results.IR, "IR_Successful"),
			get_stats(unsuccessful_results.IR, "IR_Unsuccessful"),
			get_stats(successful_results.CI_Lower, "IR_CI_Lower_Successful"),
			get_stats(unsuccessful_results.CI_Lower, "IR_CI_Lower_Unsuccessful"),
			get_stats(successful_results.CI_Upper, "IR_CI_Upper_Successful"),
			get_stats(unsuccessful_results.CI_Upper, "IR_CI_Upper_Unsuccessful"),
			get_stats(all_pareto_results_df.HHI, "HHI_All"),
		)
		mkpath(EVAL_STATS_DIR)
		stats_path = joinpath(EVAL_STATS_DIR, "apso_summary_stats.csv")
		CSV.write(stats_path, stats_df)
		println("Saved APSO summary statistics to '$(stats_path)'.")
		println("Saved all APSO stage plots to 'Plots/APSO'.")
	end

	return all_results_df, all_results_df, successful_quarters_count, unsuccessful_quarters_count
end

## ------------------------------------------------------------------------------------------
## ----------------------- STAGE 2: Feature Engineering and Selection -----------------------
## ------------------------------------------------------------------------------------------
function run_feature_engineering_stage(apso_output_df::DataFrame;
	lag_vals, target_lag_vals, window_vals,
	mi_std_multiplier, mi_fallback_n,
	poly_degree, sulov_corr_thresh,
	optuna_trials,
	sulov_plot_seed_nodes, sulov_plot_k_neighbors)
	println("\n" * "="^60)
	println("ENTERING STAGE 2: FEATURE ENGINEERING")
	println("="^60 * "\n")

	# --- 2.1 Global Data Handling (In-Memory) ---
	df = copy(apso_output_df)
	df.Date = firstdayofquarter.(df.Date)
	sort!(df, :Date)
	final_data_result = df
	println("Successfully received and processed data from APSO stage.")

	protected_weight_features = [col for col in names(final_data_result) if endswith(col, "_Weight")]
	println("Identified $(length(protected_weight_features)) protected '_Weight' features.")

	# --- 2.2 Data Preparation and Feature Separation ---
	function prepare_data_for_ml(df_input)
		println("\n--- 2.2 Preparing Data ---")
		if !(TARGET_VARIABLE in names(df_input))
			throw(ArgumentError("Target column '$(TARGET_VARIABLE)' not found."))
		end
		df_clean = dropmissing(df_input, Symbol(TARGET_VARIABLE))
		target_df = df_clean[:, [:Date, Symbol(TARGET_VARIABLE)]]
		weight_features_df = df_clean[:, [:Date; Symbol.(protected_weight_features)]]
		engineered_feature_names = [col for col in names(df_clean) if col != TARGET_VARIABLE && !(col in protected_weight_features) && col != "Date"]
		engineered_features_df = df_clean[:, [:Date; Symbol.(engineered_feature_names)]]
		println("Separated target variable. $(ncol(engineered_features_df) - 1) features will be engineered.")
		return target_df, engineered_features_df, weight_features_df
	end

	# --- 2.3 Time-Series Feature Generation ---
	function generate_ts_features(features_df, target_series)
		sort!(features_df, :Date)
		sort!(target_series, :Date)
		all_new_features_df = DataFrame(Date = features_df.Date)
		feature_cols = names(features_df, Not(:Date))

		# Lagged features
		for col in feature_cols
			for lag_val in lag_vals
				all_new_features_df[!, Symbol("$(col)_lag_$(lag_val)")] = ShiftedArrays.lag(features_df[!, col], lag_val)
			end
		end
		target_name = names(target_series)[2]
		for lag_val in target_lag_vals
			all_new_features_df[!, Symbol("$(target_name)_lag_$(lag_val)")] = ShiftedArrays.lag(target_series[!, target_name], lag_val)
		end

		# Rolling features
		for col in feature_cols
			for window in window_vals
				if nrow(features_df) >= window
					rolled_mean = rollmean(features_df[!, col], window)
					padded_mean = vcat(fill(missing, window - 1), rolled_mean)
					all_new_features_df[!, Symbol("$(col)_rolling_mean_$(window)q")] = padded_mean

					rolled_std = rollstd(features_df[!, col], window)
					padded_std = vcat(fill(missing, window - 1), rolled_std)
					all_new_features_df[!, Symbol("$(col)_rolling_std_$(window)q")] = padded_std
				else
					all_new_features_df[!, Symbol("$(col)_rolling_mean_$(window)q")] = zeros(nrow(features_df))
					all_new_features_df[!, Symbol("$(col)_rolling_std_$(window)q")] = zeros(nrow(features_df))
				end
			end
		end

		# Cyclical features
		all_new_features_df.quarter_of_year = quarterofyear.(features_df.Date)
		all_new_features_df.year = year.(features_df.Date)
		all_new_features_df.quarter_sin = sin.(2 * π * all_new_features_df.quarter_of_year / 4)
		all_new_features_df.quarter_cos = cos.(2 * π * all_new_features_df.quarter_of_year / 4)

		ts_features = all_new_features_df

		for col in names(ts_features, Not(:Date))
			ts_features[!, col] = map(x -> ismissing(x) || !isfinite(x) ? 0.0 : x, ts_features[!, col])
		end

		return ts_features
	end

	# --- Main Execution Script ---
	target_df, engineered_features, weight_features = prepare_data_for_ml(final_data_result)
	println("\n--- Data Split ---")
	n_rows = nrow(target_df)
	train_size = Int(floor(0.70 * n_rows))
	val_size = Int(floor(0.15 * n_rows))
	train_indices = 1:train_size
	valid_indices = (train_size+1):(train_size+val_size)
	test_indices = (train_size+val_size+1):n_rows
	y_train, y_valid, y_test = target_df[train_indices, :], target_df[valid_indices, :], target_df[test_indices, :]
	X_train_engineered_raw, X_valid_engineered_raw, X_test_engineered_raw = engineered_features[train_indices, :], engineered_features[valid_indices, :], engineered_features[test_indices, :]
	X_train_weights, X_valid_weights, X_test_weights = weight_features[train_indices, :], weight_features[valid_indices, :], weight_features[test_indices, :]
	println("Train: $(nrow(y_train)) rows\nValidation: $(nrow(y_valid)) rows\nTest: $(nrow(y_test)) rows")

	println("\n--- 2.4 Generating Time-Series Features ---")
	X_train_ts_engineered = generate_ts_features(X_train_engineered_raw, y_train)
	X_train_ts_unaligned = innerjoin(X_train_ts_engineered, X_train_weights, on = :Date)
	println("Generated $(ncol(X_train_ts_engineered) - 1) time-series features.")

	# --- 2.5 Final Data Alignment ---
	train_df = innerjoin(X_train_ts_unaligned, y_train, on = :Date)
	println("Data aligned. Final training set size: $(nrow(train_df)) rows.")

	y_train_vector = train_df[!, TARGET_VARIABLE]
	feature_cols = names(train_df, Not([:Date, Symbol(TARGET_VARIABLE)]))
	X_train_ts_matrix = Matrix(train_df[:, feature_cols])

	println("\n--- 2.5.1 Dynamic MI Pre-selection ---")
	println("Calculating MI scores using scikit-learn...")
	mi_scores_vec = skl_feature_selection.mutual_info_regression(X_train_ts_matrix, y_train_vector, random_state = 42)
	mi_scores_df = DataFrame(Feature = feature_cols, MI = mi_scores_vec)
	sort!(mi_scores_df, :MI, rev = true)
	mi_selected_engineered = String[]
	local final_mi_features
	local threshold_combined = Inf
	if !isempty(mi_scores_df) && sum(mi_scores_df.MI) > 0
		println("Calculating distance matrix...")
		distance_matrix = Matrix(Distances.pairwise(Euclidean(), reshape(mi_scores_df.MI, 1, :)))
		avg_distances = mean.(eachrow(distance_matrix))
		mi_scores_df.AvgDist = avg_distances
		mi_scores_df.Combined = mi_scores_df.MI .* mi_scores_df.AvgDist
		sort!(mi_scores_df, :Combined, rev = true)
		if !isempty(mi_scores_df.Combined)
			mean_combined, std_combined = mean(mi_scores_df.Combined), std(mi_scores_df.Combined)
			threshold_combined = mean_combined + (mi_std_multiplier * std_combined)
			candidate_features_df = filter(row -> row.Combined > threshold_combined, mi_scores_df)
			if !isempty(candidate_features_df)
				candidate_features = candidate_features_df.Feature
				mi_selected_engineered = [f for f in candidate_features if !(f in protected_weight_features)]
			end
		end
	end
	if isempty(mi_selected_engineered)
		println("Custom metric filter was too aggressive or failed, falling back to top $(mi_fallback_n) features.")
		engineered_mi_scores_fallback = filter(row -> !(row.Feature in protected_weight_features), sort(mi_scores_df, :MI, rev = true))
		mi_selected_engineered = first(engineered_mi_scores_fallback, mi_fallback_n).Feature
	end
	final_mi_features = [mi_selected_engineered; protected_weight_features]
	X_train_mi = train_df[:, [:Date; Symbol.(final_mi_features)]]
	println("Selected $(length(final_mi_features)) features after MI step.")

	if ENABLE_PLOTTING
		println("\n--- 2.5.1 Generating Plot 1: Static MI Analysis Subplots ---")
		mkpath("Plots/Features")
		df_all = filter(row -> !endswith(row.Feature, "_Weight"), mi_scores_df)
		df_selected = filter(row -> row.Feature in final_mi_features && !endswith(row.Feature, "_Weight"), mi_scores_df)
		if !isempty(df_all)
			min_z, max_z = extrema(df_all.Combined)
			p_mi_all = @df df_all Plots.scatter(
				:MI,
				:AvgDist,
				zcolor = :Combined,
				markersize = :MI .* 25 .+ 2,
				markerstrokewidth = 0,
				alpha = 0.7,
				seriescolor = :viridis,
				title = "All Features",
				xlabel = "MI Score",
				ylabel = "Avg. Euclidean Distance",
				clims = (min_z, max_z),
				colorbar_title = "\nCombined Score",
				label = "",
				legend = :topright,
			)
			if isfinite(threshold_combined)
				mi_range = range(minimum(df_all.MI), maximum(df_all.MI), length = 100)
				threshold_curve = threshold_combined ./ mi_range
				Plots.plot!(p_mi_all, mi_range, threshold_curve, line = (:dash, :red), label = "Selection Threshold")
			end
			p_mi_selected = @df df_selected Plots.scatter(
				:MI,
				:AvgDist,
				zcolor = :Combined,
				markersize = :MI .* 25 .+ 2,
				markerstrokewidth = 0,
				alpha = 0.7,
				seriescolor = :viridis,
				title = "Selected Features",
				xlabel = "MI Score",
				ylabel = "",
				clims = (min_z, max_z),
				legend = false,
				colorbar = false,
			)
			fig_mi_subplots = Plots.plot(p_mi_all, p_mi_selected, layout = (1, 2), plot_title = "MI Pre-selection Analysis", size = (1600, 700), margin = 10Plots.mm)
			Plots.savefig(fig_mi_subplots, "Plots/Features/mi_analysis_subplots.png")
			println("Saved MI analysis subplots to 'Plots/Features/mi_analysis_subplots.png'")
		else
			println("No data to plot for MI analysis.")
		end
	end

	println("\n--- 2.6 Generating Polynomial Features ---")
	X_train_mi_engineered_df = X_train_mi[:, filter(c -> !(string(c) in protected_weight_features), names(X_train_mi))]
	X_train_mi_weights_df = X_train_mi[:, [:Date; Symbol.(protected_weight_features)]]
	println("Using Python's RobustScaler via PyCall...")
	py_scaler = skl_preprocessing.RobustScaler()
	numeric_cols_df = X_train_mi_engineered_df[:, Not(:Date)]
	py_scaler.fit(Matrix(numeric_cols_df))
	X_train_mi_engineered_scaled_matrix = py_scaler.transform(Matrix(numeric_cols_df))
	X_train_mi_engineered_scaled = DataFrame(X_train_mi_engineered_scaled_matrix, names(numeric_cols_df))
	poly_transformer = skl_preprocessing.PolynomialFeatures(degree = poly_degree, include_bias = false)
	poly_transformer.fit(Matrix(X_train_mi_engineered_scaled))
	poly_features_matrix = poly_transformer.transform(Matrix(X_train_mi_engineered_scaled))
	poly_feature_names = poly_transformer.get_feature_names_out(names(X_train_mi_engineered_scaled))
	X_train_poly_engineered = DataFrame(poly_features_matrix, Symbol.(poly_feature_names))
	X_train_poly_engineered.Date = X_train_mi.Date
	X_train_poly = innerjoin(X_train_poly_engineered, X_train_mi_weights_df, on = :Date)
	println("Expanded to $(ncol(X_train_poly_engineered) - 1) polynomial features.")

	println("\n--- 2.7 SULOV Selection ---")
	sulov_train_df = innerjoin(X_train_poly, y_train, on = :Date)
	X_train_poly_numeric = sulov_train_df[:, Not([:Date, Symbol(TARGET_VARIABLE)])]
	y_train_poly_vector = sulov_train_df[!, TARGET_VARIABLE]
	corr_matrix = cor(Matrix(X_train_poly_numeric))
	feature_names_sulov = names(X_train_poly_numeric)
	println("Calculating MI scores for SULOV step...")
	mis_scores_sulov_vec = skl_feature_selection.mutual_info_regression(Matrix(X_train_poly_numeric), y_train_poly_vector, random_state = 42)
	mis_scores_sulov = Dict(zip(feature_names_sulov, mis_scores_sulov_vec))
	correlated_pairs_for_plot = []
	features_to_remove_candidates = Set{String}()
	for i in 1:length(feature_names_sulov), j in (i+1):length(feature_names_sulov)
		if abs(corr_matrix[i, j]) > sulov_corr_thresh
			c1, c2 = feature_names_sulov[i], feature_names_sulov[j]
			push!(correlated_pairs_for_plot, (c1, c2, corr_matrix[i, j]))
			mi1, mi2 = get(mis_scores_sulov, c1, 0), get(mis_scores_sulov, c2, 0)
			feature_to_drop = mi1 < mi2 ? c1 : c2
			push!(features_to_remove_candidates, feature_to_drop)
		end
	end
	features_to_remove = Set([feat for feat in features_to_remove_candidates if !(feat in protected_weight_features)])
	sulov_selected_features = [f for f in feature_names_sulov if !(f in features_to_remove)]
	X_train_sulov = X_train_poly[:, [:Date; Symbol.(sulov_selected_features)]]
	println("Selected $(length(sulov_selected_features)) features after SULOV.")

	if ENABLE_PLOTTING
		println("\n--- Generating SULOV Network Analysis Subplots ---")
		all_correlated_nodes = unique(vcat([p[1] for p in correlated_pairs_for_plot], [p[2] for p in correlated_pairs_for_plot]))
		all_correlated_nodes_no_weights = filter(name -> !endswith(name, "_Weight"), all_correlated_nodes)
		mi_for_correlated = filter(p -> p.first in all_correlated_nodes_no_weights, mis_scores_sulov)
		sorted_by_mi = sort(collect(mi_for_correlated), by = x->x[2], rev = true)
		seed_nodes = first([p[1] for p in sorted_by_mi], sulov_plot_seed_nodes)
		nodes_to_plot_set = Set(seed_nodes)
		feature_to_idx = Dict(name => i for (i, name) in enumerate(feature_names_sulov))
		for seed in seed_nodes
			if haskey(feature_to_idx, seed)
				seed_idx = feature_to_idx[seed]
				corrs = abs.(corr_matrix[seed_idx, :])
				corrs[seed_idx] = 0
				p = sortperm(corrs, rev = true)
				neighbor_names = feature_names_sulov[p]
				non_weight_neighbors = filter(name -> !endswith(name, "_Weight"), neighbor_names)
				top_k_neighbors = first(non_weight_neighbors, sulov_plot_k_neighbors)
				for neighbor in top_k_neighbors
					push!(nodes_to_plot_set, neighbor)
				end
			end
		end
		nodes_to_plot = collect(nodes_to_plot_set)
		if !isempty(nodes_to_plot)
			g = SimpleGraph();
			node_map = Dict{String, Int}();
			for name in nodes_to_plot
				add_vertex!(g);
				node_map[name] = nv(g);
			end
			id_to_name_map = Dict(v => k for (k, v) in node_map)
			for (src, tgt, w) in correlated_pairs_for_plot
				if haskey(node_map, src) && haskey(node_map, tgt)
					add_edge!(g, node_map[src], node_map[tgt]);
				end
			end
			pos = NetworkLayout.spring(g, C = 2.0, iterations = 100, initialtemp = 0.5)
			connection_counts_all = countmap(vcat([p[1] for p in correlated_pairs_for_plot], [p[2] for p in correlated_pairs_for_plot]))
			all_drawable_colors = [get(connection_counts_all, name, 0) for name in nodes_to_plot]
			min_z, max_z = isempty(all_drawable_colors) ? (0, 1) : extrema(all_drawable_colors)
			function plot_network_subplot(title, features_to_show; show_colorbar = false)
				drawable_nodes = intersect(Set(nodes_to_plot), Set(features_to_show))
				node_indices = Set([node_map[f] for f in drawable_nodes])
				p = Plots.plot(title = title, legend = false, aspect_ratio = :equal, axis = ([], false), grid = false)
				for edge in edges(g)
					if src(edge) in node_indices && dst(edge) in node_indices
						Plots.plot!(p, [pos[src(edge)][1], pos[dst(edge)][1]], [pos[src(edge)][2], pos[dst(edge)][2]], color = :dimgray, lw = 1.0, alpha = 0.7)
					end
				end
				if !isempty(node_indices)
					indices = collect(node_indices);
					names = [id_to_name_map[i] for i in indices];
					colors = [get(connection_counts_all, name, 0) for name in names];
					sizes = [get(mis_scores_sulov, name, 0) * 150 + 5 for name in names];
					Plots.scatter!(
						p, [pos[i][1] for i in indices], [pos[i][2] for i in indices],
						marker_z = colors, markersize = sizes, seriescolor = :viridis, markerstrokewidth = 0.5,
						alpha = 0.6, colorbar = show_colorbar, colorbar_title = show_colorbar ? "\n# Connections" : "",
						clims = (min_z, max_z),
					)
				end
				return p
			end
			p_all = plot_network_subplot("Top Correlated Features", nodes_to_plot, show_colorbar = true)
			selected_no_weights = filter(f -> !endswith(f, "_Weight"), sulov_selected_features)
			p_selected = plot_network_subplot("Kept Features (Post-SULOV)", selected_no_weights)
			fig_sulov = Plots.plot(p_all, p_selected, layout = (1, 2), plot_title = "SULOV Network Analysis (Size=MI, Color=#Connections)", size = (1800, 900), margin = 10Plots.mm)
			Plots.savefig(fig_sulov, "Plots/Features/sulov_analysis_subplots.png")
			println("Saved SULOV analysis subplots.")
		end
	end

	println("\n--- 2.8 Transforming Validation set for RFE ---")
	X_valid_ts_engineered = generate_ts_features(X_valid_engineered_raw, y_valid)
	X_valid_ts = innerjoin(X_valid_ts_engineered, X_valid_weights, on = :Date)
	X_valid_mi_unaligned = X_valid_ts
	X_valid_mi = DataFrame(Date = X_valid_mi_unaligned.Date)
	n_valid = nrow(X_valid_mi)
	for feat in final_mi_features
		if Symbol(feat) in names(X_valid_mi_unaligned)
			X_valid_mi[!, Symbol(feat)] = X_valid_mi_unaligned[!, Symbol(feat)]
		else
			X_valid_mi[!, Symbol(feat)] = fill(0.0, n_valid)
		end
	end
	X_valid_mi_engineered_df = X_valid_mi[:, filter(c -> !(string(c) in protected_weight_features), names(X_valid_mi))]
	X_valid_mi_weights_df = X_valid_mi[:, [:Date; Symbol.(protected_weight_features)]]
	X_valid_numeric_cols_df = X_valid_mi_engineered_df[:, Not(:Date)]
	X_valid_mi_engineered_scaled_matrix = py_scaler.transform(Matrix(X_valid_numeric_cols_df))
	X_valid_mi_engineered_scaled = DataFrame(X_valid_mi_engineered_scaled_matrix, names(X_valid_numeric_cols_df))
	X_valid_poly_engineered_matrix = poly_transformer.transform(Matrix(X_valid_mi_engineered_scaled))
	X_valid_poly_engineered = DataFrame(X_valid_poly_engineered_matrix, Symbol.(poly_feature_names))
	X_valid_poly_engineered.Date = X_valid_mi.Date
	X_valid_poly = innerjoin(X_valid_poly_engineered, X_valid_mi_weights_df, on = :Date)
	X_valid_sulov_unaligned = X_valid_poly
	X_valid_sulov = DataFrame(Date = X_valid_sulov_unaligned.Date)
	for feat in sulov_selected_features
		if Symbol(feat) in names(X_valid_sulov_unaligned)
			X_valid_sulov[!, Symbol(feat)] = X_valid_sulov_unaligned[!, Symbol(feat)]
		else
			X_valid_sulov[!, Symbol(feat)] = fill(0.0, n_valid)
		end
	end

	println("\n--- 2.9 Multi-Round SHAP-RFE with Optuna Tuning ---")
	rfe_train_df = innerjoin(X_train_sulov, y_train, on = :Date)
	rfe_valid_df = innerjoin(X_valid_sulov, y_valid, on = :Date)

	X_train_sulov_matrix = Matrix(rfe_train_df[:, names(X_train_sulov, Not(:Date))])
	y_train_vec = rfe_train_df[!, TARGET_VARIABLE]
	X_valid_sulov_matrix = Matrix(rfe_valid_df[:, names(X_train_sulov, Not(:Date))])
	y_valid_vec = rfe_valid_df[!, TARGET_VARIABLE]

	function objective(trial)
		params = Dict(
			"objective" => "reg:squarederror", "eval_metric" => "rmse",
			"n_estimators" => trial.suggest_int("n_estimators", 100, 1000, step = 100),
			"learning_rate" => trial.suggest_float("learning_rate", 0.01, 0.3, log = true),
			"max_depth" => trial.suggest_int("max_depth", 3, 8),
			"subsample" => trial.suggest_float("subsample", 0.6, 1.0),
			"colsample_bytree" => trial.suggest_float("colsample_bytree", 0.6, 1.0),
			"reg_alpha" => trial.suggest_float("reg_alpha", 1e-8, 1.0, log = true),
			"reg_lambda" => trial.suggest_float("reg_lambda", 1e-8, 1.0, log = true),
			"random_state" => 42,
		)
		dtrain = DMatrix(X_train_sulov_matrix, y_train_vec)
		dvalid = DMatrix(X_valid_sulov_matrix, y_valid_vec)
		watchlist = Dict("validation" => dvalid)
		model = xgboost(dtrain; watchlist = watchlist, num_round = params["n_estimators"],
			eta = params["learning_rate"], max_depth = params["max_depth"], subsample = params["subsample"],
			colsample_bytree = params["colsample_bytree"], alpha = params["reg_alpha"], lambda = params["reg_lambda"],
			objective = params["objective"], eval_metric = params["eval_metric"],
			seed = params["random_state"], verbose_eval = false,
		)
		preds = XGBoost.predict(model, X_valid_sulov_matrix)
		mse = mean((y_valid_vec .- preds) .^ 2)
		return mse
	end
	study = optuna.create_study(direction = "minimize")
	study.optimize(objective, n_trials = optuna_trials)
	best_params_py = study.best_params
	best_params_optuna = Dict(k => v for (k, v) in best_params_py)
	best_params = Dict(
		:num_round => best_params_optuna["n_estimators"], :eta => best_params_optuna["learning_rate"],
		:max_depth => best_params_optuna["max_depth"], :subsample => best_params_optuna["subsample"],
		:colsample_bytree => best_params_optuna["colsample_bytree"], :alpha => best_params_optuna["reg_alpha"],
		:lambda => best_params_optuna["reg_lambda"], :objective => "reg:squarederror",
		:eval_metric => "rmse", :seed => 42, :verbose_eval => false,
	)
	println("Best parameters for RFE model found by Optuna: $(best_params_optuna)")

	all_round_features = []
	rfe_performance_log = []
	rfe_engineered_features = [f for f in sulov_selected_features if !(f in protected_weight_features)]
	println("\n--- Starting SHAP-RFE with Tuned Parameters ---")
	for i in 1:RFE_N_ROUNDS
		println("\n--- Round $i/$RFE_N_ROUNDS ---")
		boot_indices = StatsBase.sample(1:nrow(rfe_train_df), nrow(rfe_train_df), replace = true)
		train_boot_df = rfe_train_df[boot_indices, :]
		features_in_play = copy(rfe_engineered_features)
		selected_this_round = []
		for j in 1:RFE_N_ITERATIONS_PER_ROUND
			if isempty(features_in_play)
				break
			end
			current_training_features_sym = Symbol.([features_in_play; protected_weight_features])
			X_train_boot_iter_df = train_boot_df[:, current_training_features_sym]
			y_train_boot_iter = train_boot_df[!, TARGET_VARIABLE]
			X_valid_iter_df = rfe_valid_df[:, current_training_features_sym]
			y_valid_iter = rfe_valid_df[!, TARGET_VARIABLE]
			dtrain_iter = DMatrix(Matrix(X_train_boot_iter_df), y_train_boot_iter)
			dvalid_iter = DMatrix(Matrix(X_valid_iter_df), y_valid_iter)
			temp_model = xgboost(dtrain_iter; watchlist = Dict("validation" => dvalid_iter), best_params...)
			if i == 1
				y_pred_valid = XGBoost.predict(temp_model, Matrix(X_valid_iter_df))
				mse = mean((y_valid_iter .- y_pred_valid) .^ 2)
				push!(rfe_performance_log, Dict("features_remaining" => length(current_training_features_sym), "validation_mse" => mse))
			end
			reference_data = X_train_boot_iter_df[StatsBase.sample(1:nrow(X_train_boot_iter_df), min(50, nrow(X_train_boot_iter_df)), replace = false), :]
			shap_df = ShapML.shap(model = temp_model, explain = X_train_boot_iter_df, reference = reference_data, predict_function = (m, d) -> XGBoost.predict(m, Matrix(d)))
			shap_importance = combine(groupby(shap_df, :feature_name), :shap_effect => (x -> mean(abs.(x))) => :Importance)
			rename!(shap_importance, :feature_name => :Feature)
			shap_importance_engineered = filter(row -> row.Feature in features_in_play, shap_importance)
			sort!(shap_importance_engineered, :Importance, rev = true)
			top_n_features = first(shap_importance_engineered, RFE_N_FEATURES_PER_ITERATION).Feature
			append!(selected_this_round, top_n_features)
			features_in_play = [f for f in features_in_play if !(f in top_n_features)]
		end
		push!(all_round_features, unique(selected_this_round))
	end
	if isempty(all_round_features)
		throw(ErrorException("Feature selection failed."))
	end
	stable_engineered_features = sort(collect(intersect(Set.(all_round_features)...)))
	final_selected_features = [stable_engineered_features; protected_weight_features]
	println("\nFound $(length(stable_engineered_features)) stable engineered features. Total final features: $(length(final_selected_features))")

	println("\n--- 2.10 Final Combination and Evaluation ---")
	X_train_final = X_train_sulov[:, [:Date; Symbol.(final_selected_features)]]
	X_valid_final = X_valid_sulov[:, [:Date; Symbol.(final_selected_features)]]
	X_test_ts_engineered = generate_ts_features(X_test_engineered_raw, y_test)
	X_test_ts = innerjoin(X_test_ts_engineered, X_test_weights, on = :Date)
	X_test_final = X_test_ts[:, [:Date; Symbol.(final_selected_features)]]

	X_train_val_combined = vcat(X_train_final, X_valid_final);
	sort!(X_train_val_combined, :Date)
	y_train_val_combined = vcat(y_train, y_valid);
	sort!(y_train_val_combined, :Date)
	final_train_df = innerjoin(X_train_val_combined, y_train_val_combined, on = :Date)
	final_test_df = innerjoin(X_test_final, y_test, on = :Date)

	X_train_val_df = final_train_df[:, Not([:Date, Symbol(TARGET_VARIABLE)])];
	y_train_val_vec = final_train_df[!, TARGET_VARIABLE]
	X_test_final_df = final_test_df[:, Not([:Date, Symbol(TARGET_VARIABLE)])];
	y_test_vec = final_test_df[!, TARGET_VARIABLE]

	for col in names(X_train_val_df)
		if col ∉ names(X_test_final_df)
			X_test_final_df[!, col] .= 0.0
		end
	end
	X_test_final_df = X_test_final_df[:, names(X_train_val_df)]

	dtrain_final = DMatrix(Matrix(X_train_val_df), y_train_val_vec)
	dtest_final = DMatrix(Matrix(X_test_final_df), y_test_vec)
	final_model = xgboost(dtrain_final; watchlist = Dict("train" => dtrain_final, "test" => dtest_final), best_params...)
	y_pred_test = XGBoost.predict(final_model, Matrix(X_test_final_df))

	test_mse = mean((y_test_vec .- y_pred_test) .^ 2)
	denominator = abs.(y_test_vec) .+ abs.(y_pred_test)
	smape_terms = ifelse.(denominator .== 0, 0.0, 2 .* abs.(y_pred_test .- y_test_vec) ./ denominator)

	println("\n" * "="^50);
	println("FEATURE ENGINEERING & SELECTION COMPLETE");
	println("="^50)
	println("\n--- Final Model Performance on Test Set ---")
	@printf("  Mean Squared Error (MSE):         %.4f\n", test_mse)
	@printf("  Root Mean Squared Error (RMSE):   %.4f\n", sqrt(test_mse))
	@printf("  Mean Absolute Error (MAE):        %.4f\n", mean(abs.(y_test_vec .- y_pred_test)))
	@printf("  Symmetric MAPE (SMAPE):           %.2f%%\n", mean(smape_terms) * 100)
	@printf("  Directional Accuracy:             %.2f%%\n", mean(sign.(y_test_vec) .== sign.(y_pred_test)) * 100)
	println("\n--- Final List of Stable Engineered and Protected Features ---")
	for (i, feature) in enumerate(final_selected_features)
		marker = feature in protected_weight_features ? "[PROTECTED]" : ""
		@printf("%2d: %s %s\n", i, feature, marker)
	end

	if ENABLE_PLOTTING
		println("\n--- Generating Feature Engineering Evaluation Plots ---")
		if !isempty(all_round_features)
			all_selected_union = sort(unique(vcat(all_round_features...)))
			if !isempty(all_selected_union)
				p1 = Plots.heatmap(
					1:RFE_N_ROUNDS, all_selected_union,
					cumsum([feat in round_features for feat in all_selected_union, round_features in all_round_features], dims = 2),
					c = :viridis, title = "Feature Stability Heatmap", xlabel = "Selection Round",
					ylabel = "Engineered Feature", yflip = true, size = (1200, max(600, length(all_selected_union) * 20)),
				)
				Plots.savefig(p1, "Plots/Features/rfe_stability_heatmap.png")
			end
			feature_counts = countmap(vcat(all_round_features...))
			if !isempty(feature_counts)
				freq_df = DataFrame(Feature = collect(keys(feature_counts)), Frequency = collect(values(feature_counts)));
				sort!(freq_df, :Frequency)
				p2 = @df freq_df Plots.scatter(
					:Frequency, 1:nrow(freq_df), yticks = (1:nrow(freq_df), :Feature),
					legend = false, title = "Feature Selection Frequency", xlabel = "Number of Rounds Selected",
					ylabel = "Engineered Feature", size = (1000, max(600, nrow(freq_df) * 25)),
				)
				Plots.savefig(p2, "Plots/Features/rfe_frequency_lollipop.png")
			end
		end
		if !isempty(rfe_performance_log)
			perf_df = DataFrame(rfe_performance_log);
			sort!(perf_df, :features_remaining, rev = true)
			p3 = @df perf_df Plots.plot(:features_remaining, :validation_mse, marker = :o, title = "RFE Performance Curve (Round 1)", xlabel = "# Features Remaining", ylabel = "Validation MSE", xflip = true, legend = false)
			Plots.savefig(p3, "Plots/Features/rfe_performance_curve.png")
		end
		residuals = y_test_vec - y_pred_test
		p4 = Plots.scatter(y_pred_test, residuals, alpha = 0.6, legend = false, xlabel = "Predicted Values", ylabel = "Residuals", title = "Residual Plot of Final Model");
		Plots.hline!(p4, [0], color = :red, linestyle = :dash)
		Plots.savefig(p4, "Plots/Features/final_model_residuals.png")
		if !isempty(final_selected_features)
			ref_final = X_train_val_df[StatsBase.sample(1:nrow(X_train_val_df), min(50, nrow(X_train_val_df)), replace = false), :]
			shap_df = ShapML.shap(model = final_model, explain = X_train_val_df, reference = ref_final, predict_function = (m, d) -> XGBoost.predict(m, Matrix(d)))
			plot_df = select(shap_df, :feature_name => :Feature, :shap_effect => :SHAP_Value);
			plot_df = filter(row -> !endswith(row.Feature, "_Weight"), plot_df)
			if !isempty(plot_df)
				p5 = @df plot_df StatsPlots.violin(string.(:Feature), :SHAP_Value, group = :Feature, legend = false, permute = (:x, :y), title = "SHAP Summary", size = (1000, max(600, length(unique(plot_df.Feature)) * 30)));
				@df plot_df StatsPlots.dotplot!(p5, string.(:Feature), :SHAP_Value, group = :Feature, marker = (:circle, 3, 0.3, stroke(0)), permute = (:x, :y), legend = false)
				Plots.savefig(p5, "Plots/Features/final_model_shap_summary.png")
			end
		end
		println("Saved all Feature Engineering plots to 'Plots/Features'.")
	end

	# --- 2.11 Return Final Features DataFrame ---
	final_train_val_df_no_target = final_train_df[:, Not(Symbol(TARGET_VARIABLE))]
	final_test_df_no_target = final_test_df[:, Not(Symbol(TARGET_VARIABLE))]

	final_features_full_dataset = vcat(final_train_val_df_no_target, final_test_df_no_target)
	sort!(final_features_full_dataset, :Date)
	final_export_df = innerjoin(final_features_full_dataset, target_df, on = :Date)
	select!(final_export_df, :Date, Symbol(TARGET_VARIABLE), Not([:Date, Symbol(TARGET_VARIABLE)]))
	println("\nSuccessfully created final features DataFrame in-memory.")
	return final_export_df
end

## ------------------------------------------------------------------------------------------
## --------------------------------- STAGE 3: Forecasting -----------------------------------
## ------------------------------------------------------------------------------------------

# --- 3.1 Forecasting Helper Functions ---

"""
Generates a grid of preference weights for the Pareto frontier analysis.
Each combination [w_ir, w_risk, w_hhi] sums to 1.
"""
function generate_pareto_grid(step::Float64)
	grid = Vector{Dict{String, Float64}}()
	for w_ir in 0:step:1
		for w_risk in 0:step:(1-w_ir)
			w_hhi = 1.0 - w_ir - w_risk
			# Ensure the final weight is approximately correct due to floating point math
			if w_hhi >= -1e-9
				push!(grid, Dict("w_ir" => w_ir, "w_risk" => w_risk, "w_hhi" => round(w_hhi, digits = 2)))
			end
		end
	end
	return grid
end

"""
Creates an interactive 3D plot of the Pareto frontier results.
"""
function plot_pareto_frontier(pareto_results)
	println("\n--- Generating 3D Interactive Pareto Frontier Plot ---")

	# Extract data for plotting
	risks = [res.risk for res in pareto_results]
	hhis = [res.hhi for res in pareto_results]
	irs = [res.ir for res in pareto_results]

	# Create hover text for each point
	hover_texts = [
		"IR: $(round(res.ir, digits=3))<br>Risk (TE): $(round(res.risk, digits=4))<br>HHI: $(round(res.hhi, digits=3))<br>---<br>w_ir: $(res.prefs["w_ir"])<br>w_risk: $(res.prefs["w_risk"])<br>w_hhi: $(res.prefs["w_hhi"])"
		for res in pareto_results
	]

	# Define the 3D scatter plot trace
	trace = PlotlyJS.scatter3d(
		x = risks,
		y = hhis,
		z = irs,
		mode = "markers",
		hoverinfo = "text",
		text = hover_texts,
		marker = attr(
			size = 5,
			color = irs,                # Color by Information Ratio
			colorscale = "Viridis",     # Colorscale
			colorbar = attr(title = "Information Ratio"),
			showscale = true,
		),
	)

	# Define the layout for the plot
	layout = PlotlyJS.Layout(
		title = "3D Pareto Frontier: IR vs. Risk vs. HHI",
		scene = attr(
			xaxis_title = "Risk (Tracking Error)",
			yaxis_title = "HHI (Concentration)",
			zaxis_title = "Predicted Information Ratio",
		),
		margin = attr(l = 0, r = 0, b = 0, t = 40),
	)

	# Create the plot object and save it as an interactive HTML file
	p = PlotlyJS.plot(trace, layout)
	path = joinpath(FORECAST_PLOTS_DIR, "pareto_frontier_3d.html")
	PlotlyJS.savefig(p, path)
	println("Saved interactive Pareto plot to '$(path)'")
end


function build_prediction_row(full_weights_vector, fixed_pca_features_matrix, master_weight_cols, pca_feature_names, all_model_features)
	fixed_features_df = DataFrame(fixed_pca_features_matrix, pca_feature_names)
	weights_df = DataFrame(permutedims(full_weights_vector), Symbol.(master_weight_cols))
	base_features_df = hcat(fixed_features_df, weights_df)
	interaction_df = DataFrame()
	for pca_col in pca_feature_names, w_col in master_weight_cols
		interaction_name = pca_col * "_x_" * w_col
		interaction_df[!, Symbol(interaction_name)] = base_features_df[!, Symbol(pca_col)] .* base_features_df[!, Symbol(w_col)]
	end
	full_prediction_row = hcat(base_features_df, interaction_df)
	return full_prediction_row[:, Symbol.(all_model_features)]
end

function objective_optimization(tradable_weights, model, scaler, fixed_pca_features_matrix, master_weight_cols, tradable_weight_cols, pca_feature_names, all_model_features, cov_matrix, lambda_cov, lambda_hhi, target_weights, lambda_prior)
	# Reconstruct the full weight vector from only the tradable weights
	full_weights_vector = zeros(length(master_weight_cols))
	tradable_indices = [findfirst(==(w), master_weight_cols) for w in tradable_weight_cols]
	full_weights_vector[tradable_indices] = tradable_weights

	prediction_row_df = build_prediction_row(full_weights_vector, fixed_pca_features_matrix, master_weight_cols, pca_feature_names, all_model_features)
	prediction_row_scaled = scaler.transform(Matrix(prediction_row_df))
	predicted_ir = model.predict(prediction_row_scaled)[1]

	# Penalties are only calculated on the tradable assets
	tracking_error_variance = tradable_weights' * cov_matrix * tradable_weights
	hhi = sum(tradable_weights .^ 2)
	normalized_hhi = length(tradable_weights) > 1 ? (hhi - 1/length(tradable_weights)) / (1 - 1/length(tradable_weights)) : 1.0
	prior_penalty = sum((tradable_weights .- target_weights) .^ 2)

	return -predicted_ir + (lambda_cov * tracking_error_variance) + (lambda_hhi * normalized_hhi) + (lambda_prior * prior_penalty)
end

function nomad_objective_function(
	hyperparams::Vector{Float64},
	preferences::Dict,
	model,
	scaler,
	fixed_pca_features_matrix,
	master_weight_cols,
	tradable_weight_cols,
	pca_feature_names,
	all_model_features,
	excess_cov_matrix,
	absolute_cov_matrix,
	correlation_matrix, # Argument for the new risk metric
	asset_returns_matrix,
	benchmark_returns_vector,
	prior_weights,
	dynamic_bounds,
)
	lambda_cov, lambda_hhi, lambda_prior = hyperparams[1], hyperparams[2], hyperparams[3]

	sum_to_one_constraint = py"dict"(type = "eq", fun = w -> sum(w) - 1)
	constraints_list = [sum_to_one_constraint]

	res = sp_optimize.minimize(
		objective_optimization,
		prior_weights,
		args = (model, scaler, fixed_pca_features_matrix, master_weight_cols, tradable_weight_cols, pca_feature_names, all_model_features, excess_cov_matrix, lambda_cov, lambda_hhi, prior_weights, lambda_prior),
		method = "SLSQP",
		bounds = dynamic_bounds,
		constraints = constraints_list,
	)
	if !res["success"]
		return (false, true, [1e9]);
	end
	w_opt = res["x"]
	clean_ir = -objective_optimization(w_opt, model, scaler, fixed_pca_features_matrix, master_weight_cols, tradable_weight_cols, pca_feature_names, all_model_features, excess_cov_matrix, 0, 0, prior_weights, 0)

	# Calculate the selected risk metric
	local risk_value
	if FC_RISK_METRIC == :TrackingError
		risk_value = sqrt(abs(w_opt' * excess_cov_matrix * w_opt))
	elseif FC_RISK_METRIC == :AvgCorrelation
		# This calculates the portfolio's weighted average pairwise correlation.
		# A lower value indicates better diversification.
		risk_value = w_opt' * correlation_matrix * w_opt
	elseif FC_RISK_METRIC == :Stdev
		risk_value = sqrt(abs(w_opt' * absolute_cov_matrix * w_opt))
	elseif FC_RISK_METRIC == :Beta
		portfolio_returns = asset_returns_matrix * w_opt
		model_data = DataFrame(Y = portfolio_returns, X = benchmark_returns_vector)
		ols = lm(@formula(Y ~ X), model_data)
		risk_value = coef(ols)[2]
	else
		risk_value = 0.0 # Default case
	end

	hhi = sum(w_opt .^ 2)
	score = -(preferences["w_ir"] * clean_ir) + (preferences["w_risk"] * risk_value) + (preferences["w_hhi"] * hhi)
	return (true, true, [score])
end


# --- 3.2 Main Forecasting Orchestrator ---
function run_forecasting_stage(final_features_df::DataFrame, prices_weekly_df::DataFrame;
	pca_variance_threshold, ridge_alpha_start, ridge_alpha_end, ridge_alpha_count, ridge_cv_folds,
	nomad_max_evals, n_bootstrap, benchmark,
	tradable_universe::Vector{String})

	println("\n" * "="^60)
	println("ENTERING STAGE 3: FORECASTING")
	println("Selected Risk Metric for Optimization: $(FC_RISK_METRIC)")
	println("Received a tradable universe of $(length(tradable_universe)) assets for the forecast.")
	println("="^60 * "\n")

	println("\n--- Stage 3.1: Loading Data ---")
	mkpath(FORECAST_PLOTS_DIR)
	df_full = final_features_df

	all_features = filter(c -> c != "Date" && c != TARGET_VARIABLE, names(df_full))
	master_weight_cols = filter(col -> endswith(col, "_Weight"), all_features)
	non_weight_features = filter(col -> !endswith(col, "_Weight"), all_features)
	df_history = df_full[1:(end-1), :]
	df_predict_point = df_full[end:end, :]
	if isempty(df_history) || isempty(df_predict_point)
		println("ERROR: Not enough data for train/predict split.")
		return nothing, nothing
	end
	y_history = df_history[:, Symbol(TARGET_VARIABLE)]

	println("\n--- Stage 3.2: Calculating Covariance Matrices ---")
	local excess_cov_matrix, absolute_cov_matrix, correlation_matrix, asset_returns_matrix, benchmark_returns_vector
	tradable_weight_cols = [w * "_Weight" for w in tradable_universe if w * "_Weight" in master_weight_cols]

	try
		log_ret(p) = [NaN; log.(p[2:end] ./ p[1:(end-1)])]
		clean_val(x) = !isfinite(x) ? 0.0 : x
		stock_weekly_log_returns = DataFrame(Date = prices_weekly_df.Date)
		for col in tradable_universe
			if hasproperty(prices_weekly_df, Symbol(col))
				prices = coalesce.(prices_weekly_df[:, Symbol(col)], NaN)
				stock_weekly_log_returns[!, col] = clean_val.(log_ret(prices))
			end
		end
		benchmark_prices = coalesce.(prices_weekly_df[:, Symbol(benchmark)], NaN)
		rsp_weekly_log_returns = DataFrame(Date = prices_weekly_df.Date, Benchmark = clean_val.(log_ret(benchmark_prices)))

		min_date, max_date = minimum(df_history.Date), maximum(df_history.Date)
		stock_returns_filtered = filter(row -> min_date <= row.Date <= max_date, stock_weekly_log_returns)
		benchmark_returns_filtered = filter(row -> min_date <= row.Date <= max_date, rsp_weekly_log_returns)

		combined_df = innerjoin(stock_returns_filtered, benchmark_returns_filtered, on = :Date)

		asset_returns_matrix = Matrix(combined_df[:, Symbol.(tradable_universe)])
		benchmark_returns_vector = combined_df.Benchmark
		excess_returns_matrix = asset_returns_matrix .- benchmark_returns_vector

		lw_excess = sklearn_covariance.LedoitWolf()
		lw_excess.fit(excess_returns_matrix)
		excess_cov_matrix = convert(Matrix{Float64}, lw_excess."covariance_")

		lw_abs = sklearn_covariance.LedoitWolf()
		lw_abs.fit(asset_returns_matrix)
		absolute_cov_matrix = convert(Matrix{Float64}, lw_abs."covariance_")

		# Calculate the asset correlation matrix
		asset_std_devs = sqrt.(diag(absolute_cov_matrix))
		outer_prod_std_devs = asset_std_devs * asset_std_devs'
		# Add epsilon to prevent division by zero for assets with no variance
		correlation_matrix = absolute_cov_matrix ./ (outer_prod_std_devs .+ epsilon)
		println("Successfully calculated correlation matrix for the :AvgCorrelation risk metric.")

		println("Successfully calculated shrunk covariance matrices.")
	catch e
		println("ERROR: Failed to calculate covariance matrices. Details: $e")
		return nothing, nothing
	end

	println("\n--- Stage 3.3: PCA & Interaction Features ---")
	X_history_non_weight = Matrix(df_history[:, Symbol.(non_weight_features)])
	pca_scaler = skl_preprocessing.RobustScaler()
	X_history_non_weight_scaled = pca_scaler.fit_transform(X_history_non_weight)
	pca = skl_decomposition.PCA(n_components = pca_variance_threshold, random_state = 42)
	X_history_pca = pca.fit_transform(X_history_non_weight_scaled)
	n_components = size(X_history_pca, 2)
	println("PCA applied. Reduced $(length(non_weight_features)) features to $n_components components.")
	pca_feature_names = ["PC_$(i)" for i in 1:n_components]

	X_history_pca_df = DataFrame(X_history_pca, Symbol.(pca_feature_names))
	X_history_weights_df = df_history[:, Symbol.(master_weight_cols)]
	X_base_history = hcat(X_history_pca_df, X_history_weights_df)

	interaction_df = DataFrame()
	for pca_col in pca_feature_names, w_col in master_weight_cols
		interaction_name = pca_col * "_x_" * w_col
		interaction_df[!, Symbol(interaction_name)] = X_base_history[!, Symbol(pca_col)] .* X_base_history[!, Symbol(w_col)]
	end

	X_interactive = hcat(X_base_history, interaction_df)
	local all_model_features = names(X_interactive)
	for col in names(X_interactive)
		X_interactive[!, col] = coalesce.(X_interactive[!, col], 0.0)
	end

	train_size = Int(floor(0.70 * nrow(X_interactive)))
	X_train, y_train = X_interactive[1:train_size, :], y_history[1:train_size]
	X_val, y_val = X_interactive[(train_size+1):end, :], y_history[(train_size+1):end]
	model_scaler = skl_preprocessing.RobustScaler();
	X_train_scaled = model_scaler.fit_transform(Matrix(X_train));
	X_val_scaled = model_scaler.transform(Matrix(X_val))

	println("\n--- Stage 3.4: Training Ridge Model ---")
	alpha_grid = np.logspace(ridge_alpha_start, ridge_alpha_end, ridge_alpha_count)
	ridge_cv_model = linear_model.RidgeCV(alphas = alpha_grid, cv = ridge_cv_folds)
	ridge_cv_model.fit(X_train_scaled, y_train)
	println("RidgeCV model trained. Best alpha found: $(ridge_cv_model.alpha_)")

	println("\n--- Stage 3.5: Evaluating Model Performance ---")
	val_predictions = ridge_cv_model.predict(X_val_scaled)
	mse = mean((y_val .- val_predictions) .^ 2)
	@printf "  - Validation MSE: %.4f\n" mse

	println("\n--- Stage 3.6: Training Final Model ---")
	X_full_train_scaled = vcat(X_train_scaled, X_val_scaled)
	y_full_train = vcat(y_train, y_val)
	final_model = linear_model.Ridge(alpha = ridge_cv_model.alpha_)
	final_model.fit(X_full_train_scaled, y_full_train)
	println("Final Ridge model trained.")

	X_predict_non_weight = Matrix(df_predict_point[:, Symbol.(non_weight_features)])
	X_predict_non_weight_scaled = pca_scaler.transform(X_predict_non_weight)
	fixed_pca_features_matrix = pca.transform(X_predict_non_weight_scaled)

	# --- Common setup for optimization ---
	model_coeffs = DataFrame(Feature = all_model_features, Coeff = final_model.coef_)
	main_prior_weights_all = [abs(get(model_coeffs[model_coeffs.Feature .== w, :Coeff], 1, 0.0)) for w in master_weight_cols]
	tradable_indices = [i for (i, w_col) in enumerate(master_weight_cols) if w_col in tradable_weight_cols]
	prior_weights_tradable = main_prior_weights_all[tradable_indices]
	prior_weights_tradable ./= (sum(prior_weights_tradable) + epsilon)
	dynamic_bounds = [(0.0, 1.0) for _ in 1:length(tradable_universe)]
	local best_hyperparams

	if FC_OPTIMIZATION_MODE == :MANUAL
		println("\n--- Stage 3.7: Tuning with NOMAD.jl (MANUAL Mode) ---")
		lower_bounds, upper_bounds, x0 = [0.01, 0.01, 0.01], [5.0, 5.0, 5.0], [0.5, 0.5, 0.5]
		opts = NOMAD.NomadOptions(max_bb_eval = nomad_max_evals, display_degree = 1)
		nomad_obj_func =
			(x) -> nomad_objective_function(
				x, FC_PREFERENCES, final_model, model_scaler, fixed_pca_features_matrix, master_weight_cols,
				tradable_weight_cols, pca_feature_names, all_model_features, excess_cov_matrix, absolute_cov_matrix,
				correlation_matrix, asset_returns_matrix, benchmark_returns_vector, prior_weights_tradable, dynamic_bounds,
			)
		p = NOMAD.NomadProblem(3, 1, ["OBJ"], nomad_obj_func; lower_bound = lower_bounds, upper_bound = upper_bounds, options = opts)
		result = NOMAD.solve(p, x0)
		best_hyperparams = hasproperty(result, :x_best_feas) ? result.x_best_feas : result.x
		println("Using manually set preferences: $(FC_PREFERENCES)")

	elseif FC_OPTIMIZATION_MODE == :DYNAMIC
		println("\n--- Stage 3.7: Dynamic Tuning with Pareto Frontier (DYNAMIC Mode) ---")
		pareto_grid = generate_pareto_grid(FC_GRID_STEP)
		println("Generated a Pareto grid with $(length(pareto_grid)) preference combinations.")
		pareto_results = []

		prog = Progress(length(pareto_grid), "Evaluating Pareto Grid:")
		for prefs in pareto_grid
			lower_bounds, upper_bounds, x0 = [0.01, 0.01, 0.01], [5.0, 5.0, 5.0], [0.5, 0.5, 0.5]
			# Reduce NOMAD evaluations per point to manage total runtime
			evals_per_point = max(100, Int(floor(nomad_max_evals / length(pareto_grid))))
			opts = NOMAD.NomadOptions(max_bb_eval = evals_per_point, display_degree = 0, quiet = true)
			nomad_obj_func_grid =
				(x) -> nomad_objective_function(
					x, prefs, final_model, model_scaler, fixed_pca_features_matrix, master_weight_cols,
					tradable_weight_cols, pca_feature_names, all_model_features, excess_cov_matrix, absolute_cov_matrix,
					correlation_matrix, asset_returns_matrix, benchmark_returns_vector, prior_weights_tradable, dynamic_bounds,
				)
			p = NOMAD.NomadProblem(3, 1, ["OBJ"], nomad_obj_func_grid; lower_bound = lower_bounds, upper_bound = upper_bounds, options = opts)
			result = NOMAD.solve(p, x0)
			current_best_hyperparams = hasproperty(result, :x_best_feas) ? result.x_best_feas : result.x

			# Resolve the final weights for this preference set
			final_sum_to_one = py"dict"(type = "eq", fun = w -> sum(w) - 1)
			final_constraints = [final_sum_to_one]
			final_opt_res = sp_optimize.minimize(
				objective_optimization, prior_weights_tradable,
				args = (
					final_model, model_scaler, fixed_pca_features_matrix, master_weight_cols, tradable_weight_cols,
					pca_feature_names, all_model_features, excess_cov_matrix, current_best_hyperparams[1],
					current_best_hyperparams[2], prior_weights_tradable, current_best_hyperparams[3],
				),
				method = "SLSQP", bounds = dynamic_bounds, constraints = final_constraints,
			)

			if final_opt_res["success"]
				optimal_weights = final_opt_res["x"]
				predicted_ir =
					-objective_optimization(optimal_weights, final_model, model_scaler, fixed_pca_features_matrix, master_weight_cols, tradable_weight_cols, pca_feature_names, all_model_features, excess_cov_matrix, 0, 0, prior_weights_tradable, 0)
				risk = sqrt(abs(optimal_weights' * excess_cov_matrix * optimal_weights))
				hhi = sum(optimal_weights .^ 2)
				push!(pareto_results, (ir = predicted_ir, risk = risk, hhi = hhi, prefs = prefs, hyperparams = current_best_hyperparams))
			end
			next!(prog)
		end

		if isempty(pareto_results)
			error("Pareto frontier analysis failed to produce any valid results.")
		end

		# Select the best result from the frontier based on the highest Information Ratio
		best_point = pareto_results[argmax([res.ir for res in pareto_results])]
		best_hyperparams = best_point.hyperparams

		println("\n--- Pareto Frontier Optimal Point Selection ---")
		@printf "Selected by maximizing predicted Information Ratio.\n"
		@printf "  - Optimal Predicted IR:   %.4f\n" best_point.ir
		@printf "  - Resulting Risk (TE):    %.6f\n" best_point.risk
		@printf "  - Resulting HHI:          %.4f\n" best_point.hhi
		println("  - Optimal Preference Weights: ", best_point.prefs)

		if ENABLE_PLOTTING && !isempty(pareto_results)
			plot_pareto_frontier(pareto_results)
		end

	else
		error("Invalid FC_OPTIMIZATION_MODE specified.")
	end


	println("\n--- Stage 3.7B: Bootstrapping ---")
	bootstrap_weights_matrix = Matrix{Float64}(undef, length(tradable_universe), n_bootstrap)
	prog = Progress(n_bootstrap, "Running Bootstrap Simulations:")
	for i in 1:n_bootstrap
		bootstrap_indices = rand(1:nrow(X_interactive), nrow(X_interactive))
		X_sample = X_interactive[bootstrap_indices, :]
		y_sample = y_history[bootstrap_indices]

		scaler_sample = skl_preprocessing.RobustScaler()
		X_sample_scaled = scaler_sample.fit_transform(Matrix(X_sample));
		bootstrap_model = linear_model.Ridge(alpha = ridge_cv_model.alpha_)
		bootstrap_model.fit(X_sample_scaled, y_sample)

		model_coeffs_sample = DataFrame(Feature = all_model_features, Coeff = bootstrap_model.coef_)
		bootstrap_prior_weights_all = [abs(get(model_coeffs_sample[model_coeffs_sample.Feature .== w, :Coeff], 1, 0.0)) for w in master_weight_cols]
		bootstrap_prior_tradable = bootstrap_prior_weights_all[tradable_indices]
		bootstrap_prior_tradable ./= (sum(bootstrap_prior_tradable) + epsilon)

		sum_to_one_constraint_boot = py"dict"(type = "eq", fun = w -> sum(w) - 1)
		constraints_list_boot = [sum_to_one_constraint_boot]

		bootstrap_opt_result = sp_optimize.minimize(objective_optimization, bootstrap_prior_tradable,
			args = (
				bootstrap_model,
				scaler_sample,
				fixed_pca_features_matrix,
				master_weight_cols,
				tradable_weight_cols,
				pca_feature_names,
				all_model_features,
				excess_cov_matrix,
				best_hyperparams[1],
				best_hyperparams[2],
				bootstrap_prior_tradable,
				best_hyperparams[3],
			),
			method = "SLSQP", bounds = dynamic_bounds, constraints = constraints_list_boot)

		if bootstrap_opt_result["success"]
			bootstrap_weights_matrix[:, i] = bootstrap_opt_result["x"]
		else
			bootstrap_weights_matrix[:, i] .= NaN
		end
		next!(prog)
	end

	clean_weights = copy(bootstrap_weights_matrix)
	clean_weights[.!isfinite.(clean_weights)] .= NaN
	avg_weights = vec([mean(filter(!isnan, row)) for row in eachrow(clean_weights)])
	std_dev_weights = vec([std(filter(!isnan, row)) for row in eachrow(clean_weights)])
	avg_weights = coalesce.(avg_weights, 0.0)
	std_dev_weights = coalesce.(std_dev_weights, 0.0)
	num_valid_runs = [count(!isnan, row) for row in eachrow(clean_weights)];
	std_err_weights = std_dev_weights ./ sqrt.(max.(1, num_valid_runs))
	ci_margin = 1.96 .* std_err_weights

	println("\n--- Stage 3.8: Final Optimization ---")
	final_sum_to_one = py"dict"(type = "eq", fun = w -> sum(w) - 1)
	final_constraints = [final_sum_to_one]
	final_opt_result = sp_optimize.minimize(
		objective_optimization,
		prior_weights_tradable,
		args = (
			final_model,
			model_scaler,
			fixed_pca_features_matrix,
			master_weight_cols,
			tradable_weight_cols,
			pca_feature_names,
			all_model_features,
			excess_cov_matrix,
			best_hyperparams[1],
			best_hyperparams[2],
			prior_weights_tradable,
			best_hyperparams[3],
		),
		method = "SLSQP",
		bounds = dynamic_bounds,
		constraints = final_constraints,
	)

	optimal_weights_tradable = final_opt_result["success"] ? final_opt_result["x"] : avg_weights ./ (sum(avg_weights) + epsilon)
	if sum(optimal_weights_tradable) > epsilon
		optimal_weights_tradable ./= sum(optimal_weights_tradable);
	end
	if sum(avg_weights) > epsilon
		avg_weights ./= sum(avg_weights);
	end

	final_weights_full = zeros(length(master_weight_cols))
	avg_weights_full = zeros(length(master_weight_cols))

	final_weights_full[tradable_indices] = optimal_weights_tradable
	avg_weights_full[tradable_indices] = avg_weights

	asset_names_master = [replace(w, "_Weight" => "") for w in master_weight_cols]
	final_weights_df = DataFrame(Asset = asset_names_master, Weight = final_weights_full)
	bootstrap_weights_df = DataFrame(Asset = asset_names_master, Weight = avg_weights_full)

	println("\n" * "="^50)
	println("MODEL-BASED OPTIMIZATION COMPLETE (TUNED WITH NOMAD.JL)")
	final_ir = -objective_optimization(optimal_weights_tradable, final_model, model_scaler, fixed_pca_features_matrix, master_weight_cols, tradable_weight_cols, pca_feature_names, all_model_features, excess_cov_matrix, 0, 0, prior_weights_tradable, 0)
	final_te = sqrt(abs(optimal_weights_tradable' * excess_cov_matrix * optimal_weights_tradable))
	final_hhi = sum(optimal_weights_tradable .^ 2)
	@printf "Model's Predicted Information Ratio: %.4f\n" final_ir
	@printf "Portfolio's Predicted Tracking Error:  %.6f\n" final_te
	@printf "Portfolio's HHI (Concentration):       %.4f\n" final_hhi

	println("\n--- Final Optimal Portfolio Weights ---")
	results_text_df = sort(DataFrame(Asset = tradable_universe, Weight = final_weights_tradable), :Weight, rev = true)
	results_text_df_filtered = filter(row -> row.Weight > 1e-4, results_text_df)
	results_text_df_filtered.Weight = [@sprintf("%.2f%%", w * 100) for w in results_text_df_filtered.Weight]
	println(results_text_df_filtered)

	println("\n--- Average Bootstrapped Portfolio Weights ---")
	results_bootstrap_df = sort(DataFrame(Asset = tradable_universe, Weight = avg_weights, CIMargin = ci_margin), :Weight, rev = true)
	results_to_print = filter(row -> row.Weight > 1e-4, copy(results_bootstrap_df))
	results_to_print.Weight = [@sprintf("%.2f%%", w * 100) for w in results_to_print.Weight]
	results_to_print.CIMargin = [@sprintf("±%.2f%%", e * 100) for e in results_to_print.CIMargin]
	rename!(results_to_print, :CIMargin => Symbol("95% CI Margin"))
	println(results_to_print)
	println("="^50 * "\n")

	# --- Save Forecasting Results ---
	forecast_stats_df = DataFrame(
		Metric = ["Predicted_IR", "Predicted_TE", "Predicted_HHI"],
		Value = [final_ir, final_te, final_hhi],
	)
	final_weights_to_save = filter(row -> row.Weight > 1e-4, results_text_df)
	mkpath(EVAL_STATS_DIR)
	stats_path = joinpath(EVAL_STATS_DIR, "forecasting_results.csv")
	weights_path = joinpath(EVAL_STATS_DIR, "forecasting_weights.csv")
	CSV.write(stats_path, forecast_stats_df)
	CSV.write(weights_path, final_weights_to_save)
	println("Saved forecasting results and weights to '$(EVAL_STATS_DIR)'.")

	if ENABLE_PLOTTING
		feature_importance_df = DataFrame(Feature = all_model_features, Importance = abs.(final_model.coef_))
		if !isempty(feature_importance_df.Importance) && any(x -> x > 0, feature_importance_df.Importance)
			temp_df = filter(row -> row.Importance > 0, feature_importance_df)
			bin_edges = percentile(temp_df.Importance, 0:10:100)
			bin_edges[1] = 0;
			bin_edges[end] = Inf
			get_bin_label(imp, edges) = "P$((findfirst(x -> imp <= x, edges) - 2) * 10)-P$((findfirst(x -> imp <= x, edges) - 1) * 10)"
			feature_importance_df.PercentileBin = get_bin_label.(feature_importance_df.Importance, [bin_edges])
			histogram_df = combine(groupby(feature_importance_df, :PercentileBin), nrow => :Frequency)
			get_sort_key(bin_label) = parse(Int, split(replace(bin_label, "P" => ""), "-")[1])
			histogram_df.SortKey = get_sort_key.(histogram_df.SortKey);
			sort!(histogram_df, :SortKey)
			p_hist = Plots.bar(
				histogram_df.PercentileBin,
				histogram_df.Frequency,
				title = "Distribution of Feature Importances by Percentile",
				xlabel = "Percentile Group",
				ylabel = "Number of Features",
				legend = false,
				xrotation = 45,
				bottom_margin = 75*Plots.px,
				size = (1000, 700),
			)
			display(p_hist)
		end

		plot_df = DataFrame(Asset = tradable_universe, CurrentWeight = optimal_weights_tradable, AvgWeight = avg_weights, CIMargin = ci_margin)
		sort!(plot_df, :CurrentWeight, rev = true)
		num_assets = nrow(plot_df);
		offset = 0.1
		p_alloc = Plots.plot(
			size = (1200, 800),
			title = "Portfolio Allocation with Bootstrapped Statistics",
			xlabel = "Asset",
			ylabel = "Portfolio Weight",
			yformatter = y -> @sprintf("%.1f%%", y * 100),
			xticks = (1:num_assets, plot_df.Asset),
			xrotation = 45,
			legend = :topright,
			bottom_margin = 75*Plots.px,
		)
		Plots.scatter!(p_alloc, (1:num_assets) .- offset, plot_df.CurrentWeight, label = "Final Optimal Weight", marker = :square, color = :red)
		Plots.scatter!(p_alloc, (1:num_assets) .+ offset, plot_df.AvgWeight, yerror = plot_df.CIMargin, label = "Bootstrapped Avg. Weight & 95% CI", marker = :circle, color = :blue)
		display(p_alloc)
		println("Forecasting plots displayed.")
	end

	return final_weights_df, bootstrap_weights_df
end

## ------------------------------------------------------------------------------------------
## --------------------------------- PIPELINE ORCHESTRATOR ----------------------------------
## ------------------------------------------------------------------------------------------
function main()
	println("\n" * "#"^60)
	println("############  STARTING FULL END-TO-END PIPELINE  ############")
	println("#"^60 * "\n")

	# --- Initial Data Loading ---
	println("--- Loading initial raw data files ---")
	local market_quarterly, prices_weekly
	try
		market_quarterly_path = joinpath("Data", "market_quarterly.csv")
		prices_weekly_path = joinpath("Data", "prices_weekly.csv")
		market_quarterly = CSV.read(market_quarterly_path, DataFrame)
		prices_weekly = CSV.read(prices_weekly_path, DataFrame)
		if "Date" in names(market_quarterly) && !(eltype(market_quarterly.Date) <: Date)
			market_quarterly.Date = Date.(market_quarterly.Date)
		end
		if "Date" in names(prices_weekly) && !(eltype(prices_weekly.Date) <: Date)
			prices_weekly.Date = Date.(prices_weekly.Date)
		end
		println("Initial raw data loaded successfully.")
	catch e
		println("FATAL ERROR: Could not load initial data files. Pipeline cannot start.")
		println("Details: $e")
		return
	end

	# --- Data Slicing for Training and Forecasting ---
	sort!(market_quarterly, :Date)
	last_quarter_date = market_quarterly[end, :Date]

	market_quarterly_train = market_quarterly[1:(end-1), :]
	prices_weekly_train = filter(row -> row.Date < last_quarter_date, prices_weekly)

	println("\n--- Data has been sliced. Last quarter removed for separate forecast ---")
	println("Training data runs from $(first(market_quarterly_train.Date)) to $(last(market_quarterly_train.Date))")
	println("Forecasting for quarter starting: $last_quarter_date")

	# --- Stage 1: Run APSO on Training Data ---
	final_data_with_ci_result, final_data_result, successes, failures = run_apso_stage(
		market_quarterly_train,
		prices_weekly_train;
		n_bootstrap_samples = APSO_BOOTSTRAP_SAMPLES,
		set_particles = APSO_SWARM_PARTICLES,
		set_iters_pso = APSO_ITERATIONS,
		set_starts = APSO_MULTI_STARTS,
		pso_w_decay_rate = APSO_W_DECAY,
		pso_c1_decay_rate = APSO_C1_DECAY,
		pso_c2_increase_rate = APSO_C2_INCREASE,
		diversification_penalty_grid = APSO_PARETO_PENALTIES,
		benchmark = BENCHMARK,
	)

	if final_data_result === nothing
		println("APSO stage failed to produce results. Halting pipeline.")
		return
	end
	println("\nAPSO stage complete. Passing results to Feature Engineering...")

	# --- Prepare data for Feature Engineering by combining training output and forecast point ---
	stock_cols = [col for col in names(prices_weekly) if col != BENCHMARK && col != "Date"]
	forecast_row = copy(market_quarterly[end:end, :])
	forecast_row.Information_Ratio = [0.0]
	for col in stock_cols
		forecast_row[!, Symbol(col * "_Weight")] .= 0.0
	end

	# Ensure the forecast row has the same columns as the APSO output before concatenating
	for col in names(final_data_result)
		if col ∉ names(forecast_row)
			forecast_row[!, col] .= 0.0 # Add missing columns with a default value
		end
	end
	select!(forecast_row, names(final_data_result))

	data_for_fe = vcat(final_data_result, forecast_row)
	println("\nForecast quarter data prepared and appended for feature transformation.")

	# --- Stage 2: Run Feature Engineering ---
	final_features_df = run_feature_engineering_stage(
		data_for_fe;
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

	if final_features_df === nothing
		println("Feature Engineering stage failed to produce results. Halting pipeline.")
		return
	end
	println("\nFeature Engineering stage complete. Passing results to Forecasting...")

	# --- Stage 3: Run Forecasting ---
	println("\n--- RUN MODE: STANDARD ---")
	println("P&L constraint is not applied.")

	average_weights_df = run_forecasting_stage(
		final_features_df,
		prices_weekly; # Pass the full weekly prices for covariance calculation
		pca_variance_threshold = FC_PCA_VARIANCE_THRESHOLD,
		ridge_alpha_start = FC_RIDGE_ALPHA_START,
		ridge_alpha_end = FC_RIDGE_ALPHA_END,
		ridge_alpha_count = FC_RIDGE_ALPHA_COUNT,
		ridge_cv_folds = FC_RIDGE_CV_FOLDS,
		nomad_max_evals = FC_NOMAD_MAX_EVALS,
		n_bootstrap = FC_BOOTSTRAP_SAMPLES,
		benchmark = BENCHMARK,
	)

	println("\n" * "#"^60)
	println("#############    FULL END-TO-END PIPELINE COMPLETE   ##############")
	println("#"^60 * "\n")

	if average_weights_df === nothing
		println("Forecasting stage did not produce final weights.")
	end
end

# --- Execute main() only if not in backtest mode and run directly ---
if !isdefined(Main, :BACKTEST_MODE) || !Main.BACKTEST_MODE
	if abspath(PROGRAM_FILE) == @__FILE__
		main()
	end
end
