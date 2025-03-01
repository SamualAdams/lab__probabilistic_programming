GC.gc()

using DataFrames, Dates, CSV, Plots, Statistics, DataFramesMeta, Distributions, Measures, Printf, Formatting
include("helper.jl")
using .Helper

# Load and preprocess data
df = CSV.read("hbs_us.csv", DataFrame, stringtype=String)
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

# Visualization
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
display(plt)

# Calculate cumulative sales
# Replace missing weekly_sales with 0 for cumsum
df_all_weekly.weekly_sales = coalesce.(df_all_weekly.weekly_sales, 0.0)

# Calculate cumulative sales
df_all_weekly = @transform(groupby(df_all_weekly, :season_id), :cum_sales = cumsum(:weekly_sales))

# Plot cumulative sales by season
plt_cum = plot(
    df_all_weekly.week_of_season,
    df_all_weekly.cum_sales,
    group=df_all_weekly.season_id,
    title="Cumulative Sales by Season",
    xlabel="Week of Season",
    ylabel="Cumulative Sales",
    legend=:topleft,
    marker=:circle
)
# display(plt_cum)

df, df_all_weekly = nothing, nothing  # Free up memory

### --- Fit Prior Distributions for Each Week of Season --- ###
weekly_dists__prior = Dict()

# initialize priors
for w in 1:31
    # define "focal weeks" for smoothing, handling edge weeks (1 and 31)
    focal_weeks = if w == 1
        [w, w + 1, w + 2]
    elseif w == 31
        [w - 2, w - 1, w]
    else
        [w - 1, w, w + 1]
    end

    # extract historical sales data for focal weeks
    week_data = filter(row -> row.week_of_season in focal_weeks, df__weekly).weekly_sales

    # compute parameters for prior distributions
    mu_prior, sigma_prior = mean(week_data), max(std(week_data), .01) # ensure nonzero std deviation with max function
    prior_dist = Normal(mu_prior, sigma_prior)

    # store prior in dict before updating
    weekly_dists__prior[w] = prior_dist

end

### --- Compute Probabilistic Cumulative Demand --- ###
current_week = 21  # Define current week for forecasting
cumulative_dist = compute_cumulative_demand(weekly_dists__prior, current_week, df__weekly)

weekly_dists__posterior = weekly_dists__prior

### --- Posterior Update Using Season 8 Orders --- ###
weeks_with_data = Int[]
drift_values = Float64[]

# Define window parameters
window_half_width = 2  # e.g., ±2 weeks
min_points = 2         # Minimum points needed to compute sigma

for w in 1:31
    # Define the dynamic window (e.g., w-2 to w+2, clamped to 1-31)
    window = max(1, w - window_half_width):min(31, w + window_half_width)
    
    # Extract new evidence from Season 8 within the window
    week_data_new = filter(row -> row.week_of_season in window, df_season8).weekly_sales

    # If no data exists in the window, skip it
    if isempty(week_data_new)
        println("Week $w: No new data in window $window")
        continue
    end

    # Handle missing values by treating them as 0
    week_data_new = coalesce.(week_data_new, 0.0)

    # Compute mu and sigma based on windowed data
    mu_data = mean(week_data_new)
    if length(week_data_new) < min_points
        # Too few points to compute a reliable sigma
        sigma_data = 0.1  # Default fallback
        println("Week $w (window $window, $(length(week_data_new)) points): mu_data = $mu_data, sigma_data = $sigma_data (default)")
    else
        sigma_data = max(std(week_data_new), 0.01)
        println("Week $w (window $window, $(length(week_data_new)) points): mu_data = $mu_data, sigma_data = $sigma_data")
    end

    # Retrieve prior distribution
    prior_dist = weekly_dists__prior[w]
    mu_prior, sigma_prior = mean(prior_dist), std(prior_dist)

    # Ensure sigma_data is valid (mostly redundant now, but kept for robustness)
    if isnan(sigma_data) || sigma_data ≤ 0
        sigma_data = 0.1
    end

    # Perform Bayesian update using normal-normal conjugate
    mu_posterior = (sigma_prior^2 * mu_data + sigma_data^2 * mu_prior) / (sigma_prior^2 + sigma_data^2)
    sigma_posterior = sqrt((1 / sigma_prior^2 + 1 / sigma_data^2)^-1)

    # Update posterior distribution
    weekly_dists__posterior[w] = Normal(mu_posterior, sigma_posterior)

    # Calculate and store relative drift
    if abs(mu_prior) < 0.01
        relative_drift = 0.0
    else
        relative_drift = (mu_posterior - mu_prior) / mu_prior
    end
    push!(weeks_with_data, w)
    push!(drift_values, relative_drift)
end

# Compute base aggregate drift
base_drift = isempty(drift_values) ? 0.0 : mean(drift_values)

# Update weeks without new data with decayed drift
for w in 1:31
    if !haskey(weekly_dists__posterior, w) || weekly_dists__posterior[w] == weekly_dists__prior[w]
        mu_prior, sigma_prior = mean(weekly_dists__prior[w]), std(weekly_dists__prior[w])
        
        # Calculate distance to nearest week with data
        if isempty(weeks_with_data)
            adjusted_drift = 0.0
        else
            min_distance = minimum(abs.(w .- weeks_with_data))
            # Apply exponential decay (e.g., decay rate = 0.1 per week)
            adjusted_drift = base_drift * exp(-0.1 * min_distance)
        end
        
        mu_adjusted = mu_prior * (1 + adjusted_drift)
        weekly_dists__posterior[w] = Normal(mu_adjusted, sigma_prior)
    end
end

### --- Compute Probabilistic Cumulative Demand --- ###
cumulative_dist = compute_cumulative_demand(weekly_dists__posterior, current_week, df__weekly)

### --- inventory analysis --- ###
inventory_level = 7191  # Set current inventory level

# Define x-axis range for visualization
x_min = max(0, quantile(cumulative_dist, 0.01))  # Ensure non-negative min
x_max = max(inventory_level * 1.1, quantile(cumulative_dist, 0.99))
x_vals = range(x_min, x_max, length=100)

# Compute probability of stockout (demand exceeding inventory)
inventory_cdf = cdf(cumulative_dist, inventory_level)
inventory_percentile = inventory_cdf * 100
stockout_probability = (1 - inventory_cdf) * 100

### --- Visualization: CDF with Inventory Level --- ###
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

# Add a vertical line at inventory level
vline!([inventory_level], label="Inventory Level ($inventory_level)", linestyle=:dash, color=:red)

# Add annotations for inventory risk analysis
annotate!([
    (inventory_level * 1.01, 0.12, text("Inventory: $(format("{:,}", inventory_level))", 8, :red, :left)),
    (inventory_level * 1.01, 0.08, text("Demand Coverage: $(@sprintf("%.0f", inventory_percentile))%", 8, :red, :left)),
    (inventory_level * 1.01, 0.04, text("Stockout Probability: $(@sprintf("%.0f", stockout_probability))%", 8, :red, :left))
])

display(plt)