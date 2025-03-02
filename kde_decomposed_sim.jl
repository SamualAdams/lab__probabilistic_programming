# --- Package Imports ---
using DataFrames
using Dates
using CSV
using Statistics
using KernelDensity
using PlotlyJS
using DataFramesMeta
using Printf
using Interpolations
using Random

# --- Helper Functions ---

"""
    adaptive_rolling_mean(series::Vector{Float64}, window::Int=7)
Computes a centered rolling mean with a minimum of 1 period.
"""
function adaptive_rolling_mean(series::Vector{Float64}, window::Int=7)
    n = length(series)
    result = zeros(Float64, n)
    half_window = div(window, 2)
    for i in 1:n
        start_idx = max(1, i - half_window)
        end_idx = min(n, i + half_window)
        result[i] = mean(series[start_idx:end_idx])
    end
    return result
end

"""
    rolling_std(series::Vector{Float64}, window::Int=5)
Computes a centered rolling standard deviation with a minimum of 1 period.
"""
function rolling_std(series::Vector{Float64}, window::Int=5)
    n = length(series)
    result = fill(NaN, n)
    half_window = div(window, 2)
    for i in 1:n
        start_idx = max(1, i - half_window)
        end_idx = min(n, i + half_window)
        if end_idx - start_idx + 1 >= 1
            result[i] = std(series[start_idx:end_idx])
        end
    end
    return result
end

"""
    calculate_cumulative_sales(df::DataFrame, week_start::Int, week_end::Int)
Calculates cumulative sales for a specified time horizon.
"""
function calculate_cumulative_sales(df::DataFrame, week_start::Int, week_end::Int)
    df_range = filter(row -> week_start <= row.week_of_season <= week_end, df)
    return transform(groupby(df_range, :season_id), :smoothed_sales => cumsum => :cumulative_sales)
end

"""
    compute_interval_demand_cdf(df::DataFrame, week_start::Int, week_end::Int, inventory_level::Float64)
Computes CDF for interval demand forecast and returns x, probabilities, and intersection probability.
"""
function compute_interval_demand_cdf(df::DataFrame, week_start::Int, week_end::Int, inventory_level::Float64)
    
    if week_end <= week_start
        println("Error: week_end must be greater than week_start.")
        return nothing, nothing, nothing
    end

    display(df)

    data_start = filter(row -> row.week_of_season == week_start, df)
    data_end = filter(row -> row.week_of_season == week_end, df)
    if isempty(data_start)
        println("data start is missing")
    elseif isempty(data_end)
        println("data end is missing")
        return nothing, nothing, nothing
    end
    row_start = data_start[1, :]
    row_end = data_end[1, :]
    interval_demand = [row_end[Symbol("p$(p)_cumulative")] - row_start[Symbol("p$(p)_cumulative")] for p in 1:100]
    probabilities = [p / 100 for p in 1:100]
    closest_idx = argmin(abs.(interval_demand .- inventory_level))
    intersection_prob = probabilities[closest_idx]
    return interval_demand, probabilities, intersection_prob

    println("fin")
end

"""
    adaptive_kde_by_week(df::DataFrame, value_column::Symbol, window::Int=7)
Fits KDEs to smoothed data for each week using a sliding window.
"""
function adaptive_kde_by_week(df::DataFrame, value_column::Symbol, window::Int=7)
    weeks = sort(unique(df.week_of_season))
    half = div(window, 2)
    kde_dict = Dict()
    for (i, week) in enumerate(weeks)
        valid_weeks = weeks[max(1, i - half):min(length(weeks), i + half)]
        values = filter(row -> row.week_of_season in valid_weeks, df)[!, value_column]
        if length(values) >= 2
            smoothed = adaptive_rolling_mean(values, window)
            kde_dict[week] = KernelDensity.kde(smoothed)
        else
            kde_dict[week] = nothing
        end
    end
    return kde_dict
end

"""
    sample_kde_distribution(kde_models::Dict, num_samples::Int=1000)
Samples from KDE models to estimate percentiles, mean, and standard deviation.
"""

function sample_from_kde(kde_model::UnivariateKDE, num_samples::Int)
    # Define a fine grid over the support of the KDE
    x = range(minimum(kde_model.x), maximum(kde_model.x), length=1000)
    # Compute the PDF on this grid
    pdf_values = KernelDensity.pdf(kde_model, x)
    # Approximate the CDF via cumulative sum (normalize to 1)
    cdf_values = cumsum(pdf_values .* step(x))
    cdf_values ./= cdf_values[end]  # Ensure CDF ends at 1
    # Create an inverse CDF using interpolation
    inverse_cdf = LinearInterpolation(cdf_values, x, extrapolation_bc=Throw())
    # Sample uniformly from [0, 1] and map to the inverse CDF
    u = rand(num_samples)
    samples = inverse_cdf.(u)
    return samples
end

function sample_kde_distribution(kde_models::Dict, num_samples::Int=1000)
    percentiles = 1:100
    stats = DataFrame(week_of_season=Int[], mean=Float64[], std_dev=Float64[])
    for p in percentiles
        stats[!, Symbol("p$p")] = Float64[]
    end
    for (week, kde_model) in kde_models
        if !isnothing(kde_model)
            # Use custom sampling function instead of rand
            samples = sample_from_kde(kde_model, num_samples)
            row = Dict(:week_of_season => week, 
                       :mean => mean(samples), 
                       :std_dev => std(samples))
            for p in percentiles
                row[Symbol("p$p")] = quantile(samples, p / 100)
            end
        else
            row = Dict(:week_of_season => week, 
                       :mean => NaN, 
                       :std_dev => NaN)
            for p in percentiles
                row[Symbol("p$p")] = NaN
            end
        end
        push!(stats, row)
    end
    return stats
end

"""
    Dataset(df::DataFrame; with_nulls::Bool=false, record_frequency::String="D", analysis_frequency::Union{Nothing, String}=nothing, smooth_window::Union{Nothing, Int}=nothing)
Constructs a Dataset object, performing data preprocessing.
"""
mutable struct Dataset
    df::DataFrame
    with_nulls::Bool
    record_frequency::String
    analysis_frequency::Union{Nothing, String}
    smooth_window::Union{Nothing, Int}
end

function Dataset(df::DataFrame; with_nulls::Bool=false, record_frequency::String="D", analysis_frequency::Union{Nothing, String}=nothing, smooth_window::Union{Nothing, Int}=nothing)
    dfc = deepcopy(df)
    sort!(dfc, :date)
    freq = record_frequency
    if !isnothing(analysis_frequency) && analysis_frequency != record_frequency
        freq = analysis_frequency
        grouped = groupby(dfc, :date)
        dfc = combine(grouped, :value => sum => :value)
    end
    if with_nulls
        min_date = minimum(dfc.date)
        max_date = maximum(dfc.date)
        full_range = Date(min_date):Day(1):Date(max_date)
        df_full = DataFrame(date=collect(full_range))
        dfc = leftjoin(df_full, dfc, on=:date)
        dfc[!, :value] = coalesce.(dfc.value, 0.0)
    else
        dropmissing!(dfc, :value)
    end
    if !isnothing(smooth_window)
        valvec = dfc[!, :value]
        val_smooth = adaptive_rolling_mean(valvec, smooth_window)
        half_window = div(smooth_window, 2)
        N = length(val_smooth)
        if N > 2 * half_window
            dfc[!, :value] = val_smooth
            dfc = dfc[(half_window+1):(N-half_window), :]
        else
            @warn("Smoothing window too large for dataset, using original data")
        end
    end
    return Dataset(dfc, with_nulls, record_frequency, analysis_frequency, smooth_window)
end

# --- Main Script ---

# Load dataset (assumes sales_data.csv has 'date' and 'sales' columns)
df_hbs_blitz = CSV.read("sales_data.csv", DataFrame)
rename!(df_hbs_blitz, "date" => "date", "sales" => "value")
df_hbs_blitz[!, :date] = Date.(df_hbs_blitz.date, "m/d/yyyy")
df = df_hbs_blitz[:, [:value, :date]]

# Instantiate Dataset
focal_dataset = Dataset(df, with_nulls=false, record_frequency="D", analysis_frequency="D", smooth_window=nothing)
df_test = deepcopy(focal_dataset.df)
sort!(df_test, :date)

# Create main DataFrame
df = DataFrame(
    sales = Float64.(df_test.value),
    time_numeric = Int.(Dates.value.(df_test.date .- df_test.date[1])),
    onseason_flag = [month(d) in [10, 11, 12, 1, 2, 3, 4] ? 1 : 0 for d in df_test.date],
    season_start = [(month(d) == 1 && day(d) == 1) for d in df_test.date],
    week_of_year = Int.(week.(df_test.date)),
    month = Int.(month.(df_test.date)),
    year = Int.(year.(df_test.date)),
    date = df_test.date
)


# Assign season_id
df[!, :season_id] = cumsum(df.season_start)

# Filter on-season data
df_on_season = filter(row -> row.onseason_flag == 1, df)

# Compute seasonal indicators
if !isempty(df_on_season)
    df_on_season = transform(groupby(df_on_season, :season_id), :sales => (x -> 1:length(x)) => :day_of_season)
    df_on_season[!, :week_of_season] = div.(df_on_season.day_of_season .- 1, 7) .+ 1
else
    println("Warning: No on-season data found, creating empty DataFrame")
    df_on_season = DataFrame(season_id=Int[], sales=Float64[], time_numeric=Int[], onseason_flag=Int[],
    season_start=Bool[], week_of_year=Int[], month=Int[], year=Int[], date=Date[],
    day_of_season=Int[], week_of_season=Int[])
end

CSV.write("scratch.csv", df_on_season)
# Group by season and week
df_grouped = combine(groupby(df_on_season, [:season_id, :week_of_season, :onseason_flag]),
                     :sales => sum => :sales,
                     :date => first => :date)

# Add smoothed, normalized, and cumulative sales
if !isempty(df_grouped)
    df_grouped[!, :smoothed_sales] = adaptive_rolling_mean(df_grouped.sales, 7)
    df_grouped = transform(groupby(df_grouped, :season_id),
                           :smoothed_sales => (x -> (x .- mean(x)) ./ std(x)) => :normalized_sales,
                           :smoothed_sales => cumsum => :cumulative_sales)
else
    println("Warning: No grouped data, creating empty DataFrame")
    df_grouped = DataFrame(season_id=Int[], week_of_season=Int[], onseason_flag=Int[],
                          sales=Float64[], date=Date[], smoothed_sales=Float64[],
                          normalized_sales=Float64[], cumulative_sales=Float64[])
end

display(df_grouped)

# Parameters
week_start = 21
week_end = 31
inventory_level = 6000.0  # Adjusted to approximate 7,193 with 90% probability
seasons = unique(df_grouped.season_id)

# Process latest season with KDE and percentiles
latest_season_id = maximum(seasons)
df_focal = filter(row -> row.season_id == latest_season_id, df_grouped)
if !isempty(df_focal)
    kde_models = adaptive_kde_by_week(df_focal, :normalized_sales, 7)
    percentile_df = sample_kde_distribution(kde_models, 1000)

    reduction_factor = 0.95
    df_focal[!, :root_sales] = df_focal.sales .* reduction_factor
    df_focal[!, :baseline_sales] = df_focal.smoothed_sales .* reduction_factor
    df_focal[!, :season_id] .= latest_season_id * 1000

    display(df_focal)

    # df_focal[!, :time_numeric] = df_focal.time_numeric .+ season_length .+ 7
    # season_length = maximum(df_focal.time_numeric) - minimum(df_focal.time_numeric)

    df_focal[!, :baseline_sales_std] = rolling_std(df_focal.root_sales, 5)

    for p in 1:100
        df_focal[!, Symbol("p$p")] = map(w -> percentile_df[percentile_df.week_of_season .== w, Symbol("p$p")][1],
                                         df_focal.week_of_season) .* df_focal.baseline_sales_std .+ df_focal.baseline_sales
        df_focal[!, Symbol("p$p")] = max.(df_focal[!, Symbol("p$p")], 0)
    end

    for col in [:sales; [Symbol("p$p") for p in 1:100]]
        df_focal[!, Symbol("$(col)_cumulative")] = cumsum(df_focal[!, col])
    end
else
    println("Warning: No data for latest season, skipping KDE and percentiles")
end

display(names(df_focal))

# Compute forecast CDF
x_forecast, cdf_forecast, intersection_prob_forecast = if !isempty(df_focal)
    compute_interval_demand_cdf(df_focal, week_start, week_end, inventory_level)
else
    nothing, nothing, nothing
end

# Ensure PlotlyJS is imported (add this at the top of your script if not already present)
using PlotlyJS

# Compute additional metrics for annotation
inventory_percentile = round(intersection_prob_forecast * 100, digits=0)
stockout_probability = 100 - inventory_percentile

# Create forecast CDF trace
forecast_trace = PlotlyJS.scatter(
    x = x_forecast,
    y = cdf_forecast,
    mode = "lines",
    line = PlotlyJS.attr(color="gold", width=2),
    name = "Cumulative Demand CDF"
)

# Create inventory vertical line trace
inventory_trace = PlotlyJS.scatter(
    x = [inventory_level, inventory_level],
    y = [0, intersection_prob_forecast],
    mode = "lines",
    line = PlotlyJS.attr(color="red", dash="dash"),
    name = "Inventory Level ($inventory_level)"
)

# Define dynamic x-axis range
x_range = [minimum(x_forecast) - 0.1*(maximum(x_forecast)-minimum(x_forecast)),
           maximum(x_forecast) + 0.1*(maximum(x_forecast)-minimum(x_forecast))]

# Define layout with desired styling
layout = PlotlyJS.Layout(
    title = "Cumulative Demand CDF (Week $week_start to End of Season)",
    xaxis = PlotlyJS.attr(
        title = "Cumulative Sales",
        range = x_range,
        tickformat = ","
    ),
    yaxis = PlotlyJS.attr(
        title = "Cumulative Probability",
        range = [0, 1],
        tickformat = ".0%"
    ),
    width = 800,
    height = 600,
    legend = PlotlyJS.attr(x=1.05, y=1, xanchor="left", yanchor="top", font=PlotlyJS.attr(size=10)),
    margin = PlotlyJS.attr(l=40, r=40, t=40, b=40),
    template = "plotly_dark",
    annotations = [
        PlotlyJS.attr(
            x = inventory_level * 1.01,
            y = 0.18,
            xref = "x",
            yref = "paper",
            text = "KDE Model Results:",
            font = PlotlyJS.attr(size=8, color="red"),
            showarrow = false,
            align = "left"
        ),
        PlotlyJS.attr(
            x = inventory_level * 1.01,
            y = 0.12,
            xref = "x",
            yref = "paper",
            text = "Inventory = $(round(inventory_level))",
            font = PlotlyJS.attr(size=8, color="red"),
            showarrow = false,
            align = "left"
        ),
        PlotlyJS.attr(
            x = inventory_level * 1.01,
            y = 0.08,
            xref = "x",
            yref = "paper",
            text = "Demand Coverage = $(inventory_percentile)%",
            font = PlotlyJS.attr(size=8, color="red"),
            showarrow = false,
            align = "left"
        ),
        PlotlyJS.attr(
            x = inventory_level * 1.01,
            y = 0.04,
            xref = "x",
            yref = "paper",
            text = "Stockout Probability = $(stockout_probability)%",
            font = PlotlyJS.attr(size=8, color="red"),
            showarrow = false,
            align = "left"
        )
    ]
)

# Create and display the plot
fig = PlotlyJS.plot([forecast_trace, inventory_trace], layout)
display(fig)

# Print additional information
println("Inventory Level: $inventory_level")
println("Demand Coverage: $(inventory_percentile)%")
println("Stockout Probability: $(stockout_probability)%")