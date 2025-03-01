# Consolidated Julia Script for KDE-based Demand Forecasting and Inventory Analysis

# --- Import Necessary Packages ---
using DataFrames
using Dates
using CSV
using Plots
using Statistics
using DataFramesMeta
using KernelDensity
using Interpolations
using Random
using Measures
using Printf
using Formatting

# --- Helper Functions ---
"""
    rand_kde(kde::UnivariateKDE, n::Int)
Sample `n` values from a univariate KDE.
"""
function rand_kde(kde::UnivariateKDE, n::Int)
    x = range(minimum(kde.x), maximum(kde.x), length=1000)
    pdf_vals = pdf(kde, x)
    pdf_vals /= sum(pdf_vals)  # Normalize
    cdf_vals = cumsum(pdf_vals)
    cdf_vals /= cdf_vals[end]  # Ensure CDF reaches 1
    interp = interpolate((cdf_vals,), x, Gridded(Linear()))
    return interp.(rand(n))
end

"""
    cdf_kde(kde::UnivariateKDE, x)
Compute the CDF value at `x` for a univariate KDE.
"""
function cdf_kde(kde::UnivariateKDE, x)
    grid_x = range(minimum(kde.x), maximum(kde.x), length=1000)
    pdf_vals = pdf(kde, grid_x)
    cdf_vals = cumsum(pdf_vals .* step(grid_x))
    cdf_vals /= cdf_vals[end]  # Normalize to 1
    interp = interpolate((grid_x,), cdf_vals, Gridded(Linear()))
    return clamp(interp(x), 0.0, 1.0)  # Ensure bounds
end

# --- Data Loading and Preprocessing ---
# Load data from CSV
df = CSV.read("hbs_us.csv", DataFrame, stringtype=String)
df = dropmissing(df)

# Parse dates
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

# --- Visualization: Weekly Sales by Season ---
df_season8 = filter(row -> row.season_id == 8, df__weekly)
df__weekly_hist = filter(row -> row.season_id in [2, 3, 4, 5, 6, 7], df__weekly)
df_all_weekly = vcat(df__weekly_hist, df_season8)
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

# --- Fit Prior Distributions with KDE ---
weekly_kdes__prior = Dict()
for w in 1:31
    focal_weeks = if w == 1
        [w, w + 1, w + 2]
    elseif w == 31
        [w - 2, w - 1, w]
    else
        [w - 1, w, w + 1]
    end
    week_data = filter(row -> row.week_of_season in focal_weeks, df__weekly_hist).weekly_sales
    if length(week_data) > 1
        weekly_kdes__prior[w] = kde(week_data)
    else
        weekly_kdes__prior[w] = nothing
    end
end

# --- Posterior Update with KDE ---
weekly_kdes__posterior = Dict()
window_half_width = 2
for w in 1:31
    focal_weeks = if w == 1
        [w, w + 1, w + 2]
    elseif w == 31
        [w - 2, w - 1, w]
    else
        [w - 1, w, w + 1]
    end
    hist_data = filter(row -> row.week_of_season in focal_weeks, df__weekly_hist).weekly_sales
    window = max(1, w - window_half_width):min(31, w + window_half_width)
    new_data = filter(row -> row.week_of_season in window, df_season8).weekly_sales
    new_data = coalesce.(new_data, 0.0)
    combined_data = vcat(hist_data, new_data)
    if length(combined_data) > 1
        weekly_kdes__posterior[w] = kde(combined_data)
    else
        weekly_kdes__posterior[w] = weekly_kdes__prior[w] !== nothing ? weekly_kdes__prior[w] : nothing
    end
end

# --- Compute Cumulative Demand with KDE ---
"""
    compute_cumulative_demand(weekly_kdes, current_week, df__weekly; n_samples=10000)
Compute cumulative demand distribution from `current_week` to end of season using KDE.
"""
function compute_cumulative_demand(weekly_kdes, current_week, df__weekly; n_samples=10000)
    future_weeks = current_week:31
    weekly_samples = [rand_kde(weekly_kdes[w], n_samples) for w in future_weeks if haskey(weekly_kdes, w) && weekly_kdes[w] !== nothing]
    cumulative_samples = sum(weekly_samples, dims=1)[:]
    cumulative_kde = kde(cumulative_samples)
    
    x_vals = range(minimum(cumulative_samples), maximum(cumulative_samples), length=100)
    pdf_vals = pdf(cumulative_kde, x_vals)
    cdf_vals = cumsum(pdf_vals .* step(x_vals))
    cdf_vals /= cdf_vals[end]  # Normalize to 1
    
    # Plot CDF
    plt = plot(
        x_vals, cdf_vals,
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
    display(plt)
    
    return cumulative_kde
end

# Execute for a specific week
current_week = 21
cumulative_kde = compute_cumulative_demand(weekly_kdes__posterior, current_week, df__weekly_hist)

# --- Inventory Analysis ---
inventory_level = 7191
inventory_cdf = cdf_kde(cumulative_kde, inventory_level)
inventory_percentile = inventory_cdf * 100
stockout_probability = (1 - inventory_cdf) * 100

# Plot CDF with Inventory Level
x_vals = range(minimum(cumulative_kde.x), maximum(cumulative_kde.x), length=100)
cdf_vals = [cdf_kde(cumulative_kde, x) for x in x_vals]
plt = plot(
    x_vals, cdf_vals,
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
vline!([inventory_level], label="Inventory Level ($inventory_level)", linestyle=:dash, color=:red)
annotate!([
    (inventory_level * 1.01, 0.12, text("Inventory: $(format("{:,}", inventory_level))", 8, :red, :left)),
    (inventory_level * 1.01, 0.08, text("Demand Coverage: $(@sprintf("%.0f", inventory_percentile))%", 8, :red, :left)),
    (inventory_level * 1.01, 0.04, text("Stockout Probability: $(@sprintf("%.0f", stockout_probability))%", 8, :red, :left))
])
display(plt)

println("Inventory Level: $inventory_level")
println("Demand Coverage: $(@sprintf("%.2f", inventory_percentile))%")
println("Stockout Probability: $(@sprintf("%.2f", stockout_probability))%")