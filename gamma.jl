# Clear garbage collection
GC.gc()

# Load required packages
using DataFrames, Dates, CSV, Plots, Statistics, DataFramesMeta, Distributions, Measures, Printf, Formatting

# Include helper functions (assuming a helper.jl file exists with compute_cumulative_demand)
include("helper.jl")
using .Helper

# Load and preprocess sales data
df = CSV.read("sales_data.csv", DataFrame, stringtype=String)
df = dropmissing(df)
@transform!(df, :converted_date = tryparse.(Date, :date, dateformat"m/d/yyyy"))
sort!(df, :converted_date)
@rsubset!(df, !ismissing(:converted_date))

# Log date range
min_date = minimum(df.converted_date)
max_date = maximum(df.converted_date)
println("Date Range: $min_date to $max_date")
println("Total days: ", Dates.value(max_date - min_date) + 1)

# Assign seasons and weeks
df.year = [year(d) for d in df.converted_date]
df.month = [month(d) for d in df.converted_date]
df.onseason = [(m in [10, 11, 12, 1, 2, 3, 4] ? 1 : 0) for m in df.month]
df.season_id = cumsum(diff([df.onseason[1]; df.onseason]) .!= 0)
df.week_of_season = zeros(Int, nrow(df))
for season in unique(df.season_id)
    mask = (df.season_id .== season) .& (df.onseason .== 1)
    if any(mask)
        start_date = minimum(df.converted_date[mask])
        df.week_of_season[mask] .= div.(df.converted_date[mask] .- start_date, Day(7)) .+ 1
    end
end

# Aggregate to weekly sales
df__on_season = filter(row -> row.onseason == 1, df)
df__weekly = combine(groupby(df__on_season, [:season_id, :week_of_season]), :sales => sum => :weekly_sales)

# Visualization of weekly sales
df_season8 = filter(row -> row.season_id == 8, df__weekly)
df__weekly = filter(row -> row.season_id in [2, 3, 4, 5, 6, 7], df__weekly)
df_all_weekly = vcat(df__weekly, df_season8)

plt = plot(
    df_all_weekly.week_of_season,
    df_all_weekly.weekly_sales,
    group=df_all_weekly.season_id,
    title="Weekly Sales by Season",
    xlabel="Week of Season",
    ylabel="Weekly Sales",
    legend=:topright,
    marker=:circle
)

# Calculate and plot cumulative sales
df_all_weekly.weekly_sales = coalesce.(df_all_weekly.weekly_sales, 0.0)
df_all_weekly = @transform(groupby(df_all_weekly, :season_id), :cum_sales = cumsum(:weekly_sales))

plt_cuml = plot(
    df_all_weekly.week_of_season,
    df_all_weekly.cum_sales,
    group=df_all_weekly.season_id,
    title="Cumulative Sales by Season",
    xlabel="Week of Season",
    ylabel="Cumulative Sales",
    legend=:topleft,
    marker=:circle
)

# Free up memory
df, df_all_weekly = nothing, nothing

### Fit Prior Distributions for Each Week of Season ###
weekly_dists__prior = Dict()

for w in 1:31
    # Define focal weeks for smoothing
    focal_weeks = if w == 1
        [w, w + 1, w + 2]
    elseif w == 31
        [w - 2, w - 1, w]
    else
        [w - 1, w, w + 1]
    end

    # Extract historical sales data for focal weeks
    week_data = filter(row -> row.week_of_season in focal_weeks, df__weekly).weekly_sales

    # Compute Gamma distribution parameters
    mean_sales = mean(week_data)
    var_sales = var(week_data)
    if var_sales == 0  # Handle zero variance
        var_sales = 1e-6
    end
    α_prior = mean_sales^2 / var_sales
    θ_prior = var_sales / mean_sales
    prior_dist = Gamma(α_prior, θ_prior)

    # Store prior distribution
    weekly_dists__prior[w] = prior_dist
end

### Compute Probabilistic Cumulative Demand ###
current_week = 21
cumulative_dist = compute_cumulative_demand(weekly_dists__prior, current_week, df__weekly)

### Posterior Update Using Season 8 Orders ###
weekly_dists__posterior = weekly_dists__prior
weeks_with_data = Int[]
drift_values = Float64[]

# Window parameters
window_half_width = 2
min_points = 2

for w in 1:31
    # Define dynamic window
    window = max(1, w - window_half_width):min(31, w + window_half_width)
    
    # Extract Season 8 data within the window
    week_data_new = filter(row -> row.week_of_season in window, df_season8).weekly_sales

    if isempty(week_data_new)
        println("Week $w: No new data in window $window")
        continue
    end

    week_data_new = coalesce.(week_data_new, 0.0)
    n = length(week_data_new)
    mu_data = mean(week_data_new)
    if n < min_points
        sigma_data = 0.1  # Default for insufficient data
        println("Week $w (window $window, $n points): mu_data = $mu_data, sigma_data = $sigma_data (default)")
    else
        sigma_data = max(std(week_data_new), 0.01)
        println("Week $w (window $window, $n points): mu_data = $mu_data, sigma_data = $sigma_data")
    end

    # Adjust likelihood variance
    sigma_likelihood = n >= 2 ? sigma_data / sqrt(n) : sigma_data

    # Retrieve prior distribution moments
    prior_dist = weekly_dists__prior[w]
    mu_prior = mean(prior_dist)
    sigma_prior = std(prior_dist)

    # Bayesian update using normal approximation
    precision_prior = 1 / sigma_prior^2
    precision_likelihood = 1 / sigma_likelihood^2
    precision_posterior = precision_prior + precision_likelihood
    mu_posterior = (mu_prior * precision_prior + mu_data * precision_likelihood) / precision_posterior
    sigma_posterior = sqrt(1 / precision_posterior)

    # Convert to Gamma parameters
    α_post = (mu_posterior / sigma_posterior)^2
    θ_post = sigma_posterior^2 / mu_posterior
    weekly_dists__posterior[w] = Gamma(α_post, θ_post)

    # Calculate relative drift
    relative_drift = abs(mu_prior) < 0.01 ? 0.0 : (mu_posterior - mu_prior) / mu_prior
    push!(weeks_with_data, w)
    push!(drift_values, relative_drift)
end

# Compute base aggregate drift
base_drift = isempty(drift_values) ? 0.0 : mean(drift_values)

# Update weeks without new data
for w in 1:31
    if !(w in weeks_with_data)
        prior_dist = weekly_dists__prior[w]
        α_prior = shape(prior_dist)
        θ_prior = scale(prior_dist)
        
        min_distance = isempty(weeks_with_data) ? 0 : minimum(abs.(w .- weeks_with_data))
        adjusted_drift = base_drift * exp(-0.1 * min_distance)
        θ_adjusted = θ_prior * (1 + adjusted_drift)
        weekly_dists__posterior[w] = Gamma(α_prior, θ_adjusted)
    end
end

### Compute Posterior Cumulative Demand ###
cumulative_dist = compute_cumulative_demand(weekly_dists__posterior, current_week, df__weekly)

### Inventory Analysis ###
inventory_level = 6000

# Define x-axis range
x_min = max(0, quantile(cumulative_dist, 0.01))
x_max = max(inventory_level * 1.1, quantile(cumulative_dist, 0.99))
x_vals = range(x_min, x_max, length=100)

# Compute stockout probability
inventory_cdf = cdf(cumulative_dist, inventory_level)
inventory_percentile = inventory_cdf * 100
stockout_probability = (1 - inventory_cdf) * 100

### Visualization: CDF with Inventory Level ###
plt = plot(
    x_vals,
    cdf.(cumulative_dist, x_vals),
    title="Cumulative Demand CDF (Week $current_week to End of Season)",
    xlabel="Cumulative Sales",
    ylabel="Cumulative Probability",
    label="Cumulative Demand CDF",
    linewidth=2,
    size=(800, 600),
    legendfontsize=10,
    grid=true,
    legend=:outertopright,
    leftmargin=10mm,
    rightmargin=10mm,
    topmargin=10mm,
    bottommargin=10mm
)

# Add inventory level line
vline!([inventory_level], label="Inventory Level ($inventory_level)", linestyle=:dash, color=:red)

# Add annotations
annotate!([
    (inventory_level * 1.01, 0.18, text("Parametric (Gamma) Model:", 8, :red, :left)),
    (inventory_level * 1.01, 0.12, text("Inventory =  $(format("{:,}", inventory_level))", 8, :red, :left)),
    (inventory_level * 1.01, 0.08, text("Demand Coverage = $(@sprintf("%.0f", inventory_percentile))%", 8, :red, :left)),
    (inventory_level * 1.01, 0.04, text("Stockout Probability = $(@sprintf("%.0f", stockout_probability))%", 8, :red, :left))
])

display(plt)