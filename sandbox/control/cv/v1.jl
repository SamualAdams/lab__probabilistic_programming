#!/usr/bin/env julia

using Random, Distributions, CSV, DataFrames

##############################
# Part 1: Data Generation
##############################
total_periods = 100
# Gamma distribution parameters (adjust as desired)
shape = 2.0
scale = 500.0
gamma_dist = Gamma(shape, scale)

# Generate "Actuals" from the gamma distribution (rounded for neatness)
actuals = [round(rand(gamma_dist)) for _ in 1:total_periods]
cuml_actuals = cumsum(actuals)

##############################
# Part 2: Model Function (Naive Forecast)
##############################
# This is the only function in our script.
function naive_forecast(train_data::Vector{Float64}, horizon::Int)
    # Naively forecast by using the last observed value over the forecast horizon.
    forecast_value = train_data[end]
    return fill(forecast_value, horizon)
end

##############################
# Part 3: Assemble Table, Set Fold IDs, Compute Forecasts & Output File
##############################
# Hardcode fold boundaries (fold_starts indices). These produce 4 folds:
# Fold 1: periods 1–32, Fold 2: 33–65, Fold 3: 66–98, Fold 4: 99–100.
fold_starts = [1, 33, 66, 99]
num_folds = length(fold_starts)

# Prepare columns (we retain the given schema)
Period              = collect(1:total_periods)
Forecast            = Vector{Union{Missing,Float64}}(missing, total_periods)
Residual            = Vector{Union{Missing,Float64}}(missing, total_periods)
flag_start_fin      = zeros(Int, total_periods)   # 1 if this period starts a fold
flag_fold           = zeros(Int, total_periods)   # flag for fold start (after first)
Fold_ID             = zeros(Int, total_periods)
Forecast_Fold_2     = Vector{Union{Missing,Float64}}(missing, total_periods)
Cuml_Forecast_Fold_2= Vector{Union{Missing,Float64}}(missing, total_periods)
Forecast_Fold_3     = Vector{Union{Missing,Float64}}(missing, total_periods)
Cuml_Forecast_Fold_3= Vector{Union{Missing,Float64}}(missing, total_periods)

# Assign fold IDs and flag the start of each fold.
current_fold = 1
next_fold_index = 2
for i in 1:total_periods
    if next_fold_index <= num_folds && i == fold_starts[next_fold_index]
        current_fold = next_fold_index
        flag_start_fin[i] = 1
        flag_fold[i] = 1
        next_fold_index += 1
    elseif i == fold_starts[1]
        flag_start_fin[i] = 1
    end
    Fold_ID[i] = current_fold
end

# Primary Forecast: Naively use the previous period's actual (for periods 2 onward)
for i in 2:total_periods
    Forecast[i] = actuals[i-1]
    Residual[i] = actuals[i] - Forecast[i]
end

# Compute additional forecast folds using our model.
# For Fold 2 (periods 33–65): training data = periods 1–32.
if num_folds >= 2
    start_idx = fold_starts[2]
    end_idx = (num_folds >= 3 ? fold_starts[3] - 1 : total_periods)
    horizon = end_idx - start_idx + 1
    train_data = actuals[1:start_idx-1]
    fc = naive_forecast(train_data, horizon)
    for (j, idx) in enumerate(start_idx:end_idx)
        Forecast_Fold_2[idx] = fc[j]
    end
    # Cumulative forecast within Fold 2
    cf2 = cumsum(skipmissing(Forecast_Fold_2[start_idx:end_idx]))
    for (j, idx) in enumerate(start_idx:end_idx)
        Cuml_Forecast_Fold_2[idx] = cf2[j]
    end
end

# For Fold 3 (periods 66–98): training data = periods 1–65.
if num_folds >= 3
    start_idx = fold_starts[3]
    end_idx = (num_folds >= 4 ? fold_starts[4] - 1 : total_periods)
    horizon = end_idx - start_idx + 1
    train_data = actuals[1:start_idx-1]
    fc = naive_forecast(train_data, horizon)
    for (j, idx) in enumerate(start_idx:end_idx)
        Forecast_Fold_3[idx] = fc[j]
    end
    # Cumulative forecast within Fold 3
    cf3 = cumsum(skipmissing(Forecast_Fold_3[start_idx:end_idx]))
    for (j, idx) in enumerate(start_idx:end_idx)
        Cuml_Forecast_Fold_3[idx] = cf3[j]
    end
end

# Assemble the final DataFrame (retaining our schema)
df = DataFrame(
    Period               = Period,
    Actuals              = actuals,
    Forecast             = Forecast,
    Residual             = Residual,
    "FLAG - Start/Fin"   = flag_start_fin,
    "FLAG - Fold"        = flag_fold,
    "Fold ID"            = Fold_ID,
    "Cuml. Actuals"      = cuml_actuals,
    "Forecast Fold 2"    = Forecast_Fold_2,
    "Cuml. Forecast Fold 2" = Cuml_Forecast_Fold_2,
    "Forecast Fold 3"    = Forecast_Fold_3,
    "Cuml. Forecast Fold 3" = Cuml_Forecast_Fold_3
)

# Write the DataFrame to a tab-delimited file.
output_file = "forecast_data.tsv"
CSV.write(output_file, df; delim='\t')
println("File written to ", output_file)
