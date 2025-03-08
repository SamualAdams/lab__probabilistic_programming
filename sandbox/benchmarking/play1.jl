using Distributions, Random, Plots, Statistics, DataFrames

# Parameters
n_actuals = 100          # Total number of actual data points
n_forecast = 50          # Number of folds for cross-validation
forecast_horizon = 4     # Number of steps to forecast ahead

# Generate synthetic actual data from Gamma distribution
Random.seed!(123)
actual = rand(Gamma(2, 1), n_actuals)  # Shape=2, Scale=1, no trend

# Initialize storage
df_forecasts = DataFrame()             # To store forecasts for each fold
squared_errors = Float64[]             # To collect all individual squared errors

# Rolling cross-validation loop
for fold in 1:n_forecast
    # Define training window (expands with each fold)
    train_size = n_actuals - n_forecast + fold - 1
    train = actual[1:train_size]

    # Define test window (up to 4 steps or until data ends)
    test_range = train_size + 1 : min(train_size + forecast_horizon, n_actuals)
    test_values = actual[test_range]

    # Fit normal distribution model to training data
    mu = mean(train)
    sigma = std(train)
    model = Normal(mu, sigma)

    # Generate forecasts for the test period
    Random.seed!(222 + fold)  # Ensure reproducibility per fold
    forecast = rand(model, length(test_range))

    # Store forecasts in DataFrame, padding with missing if needed
    df_forecasts[:, "Fold_$fold"] = vcat(forecast, fill(missing, forecast_horizon - length(forecast)))

    # Compute and store individual squared errors
    se = (forecast .- test_values) .^ 2
    append!(squared_errors, se)
end

# Compute overall RMSE across all individual forecast errors
rmse = sqrt(mean(squared_errors))
println("Cross-Validation RMSE (4-step): ", rmse)

# Plot actuals
plot(1:n_actuals, actual, seriestype=:line, label="Actual (Gamma)", linewidth=2, color=:blue,
     title="Rolling Cross-Validation: 4-Step Forecast\nRMSE: $(round(rmse, digits=2))", size=(1000, 600))

# Overlay forecast paths
for fold in 1:n_forecast
    preds = collect(skipmissing(df_forecasts[:, "Fold_$fold"]))
    if !isempty(preds)
        time_steps = n_actuals - n_forecast + fold : n_actuals - n_forecast + fold + length(preds) - 1
        if fold == 1
            plot!(time_steps, preds, seriestype=:line, label="Forecasts", color=:red, alpha=0.2, linewidth=0.5)
        else
            plot!(time_steps, preds, seriestype=:line, label=nothing, color=:red, alpha=0.2, linewidth=0.5)
        end
    end
end

# Display the plot (implicit in an interactive environment like Jupyter or REPL with Plots.jl)