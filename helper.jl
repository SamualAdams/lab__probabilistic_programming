module Helper

using DataFrames, Dates, CSV, Plots, Statistics, DataFramesMeta, Distributions, Measures, Printf, Formatting

export compute_cumulative_demand


function compute_cumulative_demand(weekly_dists, current_week, df__weekly)
    # Collect future weeks (current_week to 31)
    future_weeks = current_week:31
    n_weeks = length(future_weeks)

    # Compute cumulative means and variances incrementally by week for the forecast
    cumulative_means = cumsum([mean(weekly_dists[w]) for w in future_weeks])
    cumulative_variances = cumsum([var(weekly_dists[w]) for w in future_weeks])
    cumulative_stds = sqrt.(cumulative_variances)

    # Final cumulative distribution (for return)
    mu_cumulative = cumulative_means[end]
    sigma_cumulative = cumulative_stds[end]
    cumulative_dist = Normal(mu_cumulative, sigma_cumulative)

    # Compute percentile bounds for shading
    percentiles = Dict(
        10 => [quantile(Normal(mu, sigma), 0.10) for (mu, sigma) in zip(cumulative_means, cumulative_stds)],
        25 => [quantile(Normal(mu, sigma), 0.25) for (mu, sigma) in zip(cumulative_means, cumulative_stds)],
        75 => [quantile(Normal(mu, sigma), 0.75) for (mu, sigma) in zip(cumulative_means, cumulative_stds)],
        90 => [quantile(Normal(mu, sigma), 0.90) for (mu, sigma) in zip(cumulative_means, cumulative_stds)]
    )

    # Compute historical cumulative demand by season_id for the same weeks
    historical_data = filter(row -> row.week_of_season in future_weeks, df__weekly)
    historical_cumulative = combine(
        groupby(historical_data, [:season_id, :week_of_season]),
        :weekly_sales => sum => :sales_sum
    )
    historical_cumulative = sort(historical_cumulative, :week_of_season)
    historical_cumulative = transform(
        groupby(historical_cumulative, :season_id),
        :sales_sum => cumsum => :cum_sales
    )

    # Set PlotlyJS as the backend for interactive plots
    plotlyjs()

    # Plot: Start with just the axes and styling
    plt = plot(
        title="Cumulative Demand from Week $current_week to End of Season",
        xlabel="Week of Season",
        ylabel="Cumulative Sales",
        grid=true,
        size=(800, 600),
        legend=:topleft,
        leftmargin=10mm,
        rightmargin=10mm,
        topmargin=10mm,
        bottommargin=10mm
    )

    # Overlay historical cumulative demand last (top layer)
    for group in groupby(historical_cumulative, :season_id)
        season_id = group.season_id[1]  # Extract season ID properly
        plot!(
            group.week_of_season, group.cum_sales;
            label="Season $season_id",  # Correctly formatted label
            linewidth=3,
            alpha=1
        )
    end

    # Add forecast percentile shading first (bottom layer)
    plot!(future_weeks, percentiles[90], fillrange=percentiles[10],
        fillalpha=0.12, color=:yellow, label="10th-90th Percentile", linewidth=0)
    plot!(future_weeks, percentiles[75], fillrange=percentiles[25],
        fillalpha=0.12, color=:red, label="25th-75th Percentile", linewidth=0)

    # Add forecast expected demand next (middle layer)
    plot!(future_weeks, cumulative_means,
        label="Forecasted Expected Demand",
        linewidth=2,
        color=:blue,
        linestyle=:dash
    )

    # Display the interactive plot
    display(plt)

    return cumulative_dist
end


end