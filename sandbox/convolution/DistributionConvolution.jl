module DistributionConvolution

using Distributions
using FFTW
using Plots
using KernelDensity

export convolve_distributions, plot_convolution

# convolve_distributions(; dists=nothing, pdfs=nothing, cdfs=nothing, x=nothing, sigma_range=4, N=4096)
# Compute the convolution of multiple distributions or precomputed PDFs using FFT.
#
# Arguments:
# - dists: Vector of continuous univariate distributions (optional, default: nothing).
# - pdfs: N × K matrix of precomputed PDFs (optional, default: nothing).
# - cdfs: N × K matrix of precomputed CDFs (optional, default: nothing).
# - x: Grid points (required if pdfs/cdfs provided, default: nothing).
# - sigma_range: Number of standard deviations for the grid range (default: 4).
# - N: Number of grid points (default: 4096).
#
# Returns:
# - x: Grid points.
# - conv_pdf: Convolved PDF.
# - conv_cdf: Convolved CDF.
# - pdfs: Matrix of individual PDFs.
# - cdfs: Matrix of individual CDFs.
#
# Notes:
# - Either dists or (pdfs, cdfs, x) must be provided.
# - If dists is provided, pdfs/cdfs/x are computed internally.
# - If pdfs/cdfs/x are provided, they are used directly.
#
# Example:
# dists = [Exponential(1.0), Normal(10.0, 3.0)]
# x, conv_pdf, conv_cdf, pdfs, cdfs = convolve_distributions(dists=dists, sigma_range=5)
function convolve_distributions(; dists::Union{Vector{<:ContinuousUnivariateDistribution}, Nothing}=nothing, 
                                pdfs::Union{Matrix{<:Real}, Nothing}=nothing, 
                                cdfs::Union{Matrix{<:Real}, Nothing}=nothing, 
                                x::Union{AbstractVector{<:Real}, Nothing}=nothing, 
                                sigma_range::Real=4, N::Int=4096)
    # Validate inputs
    if isnothing(dists) && (isnothing(pdfs) || isnothing(cdfs) || isnothing(x))
        throw(ArgumentError("Either dists or (pdfs, cdfs, x) must be provided"))
    end
    if sigma_range <= 0
        throw(ArgumentError("sigma_range must be positive"))
    end
    if N < 2
        throw(ArgumentError("N must be at least 2"))
    end

    # If dists provided, compute pdfs/cdfs/x
    if !isnothing(dists)
        K = length(dists)
        if K == 0
            throw(ArgumentError("Distribution list cannot be empty"))
        end

        # Calculate total mean and standard deviation
        total_mean = sum(mean, dists)
        total_var = sum(var, dists)
        total_std = sqrt(total_var)

        # Define the grid
        x_min = total_mean - sigma_range * total_std
        x_max = total_mean + sigma_range * total_std
        x = range(x_min, x_max, length=N)
        delta = (x_max - x_min) / (N - 1)

        # Discretize PDFs and compute CDFs
        pdfs = zeros(N, K)
        cdfs = zeros(N, K)
        for k in 1:K
            pdf_vals = pdf(dists[k], x)
            pdf_vals ./= (sum(pdf_vals) * delta)  # Normalize PDF
            pdfs[:, k] = pdf_vals
            cdfs[:, k] = cdf(dists[k], x)
        end
    else
        # Use provided pdfs/cdfs/x
        if size(pdfs) != size(cdfs)
            throw(ArgumentError("pdfs and cdfs must have the same dimensions"))
        end
        if size(pdfs, 1) != length(x)
            throw(ArgumentError("First dimension of pdfs/cdfs must match length of x"))
        end
        N = length(x)
        delta = (x[end] - x[1]) / (N - 1)
    end

    # FFT convolution
    fft_pdfs = fft(pdfs, 1)
    fft_result = prod(fft_pdfs, dims=2)
    conv_result = real(ifft(fft_result, 1))[:, 1]
    conv_result ./= (sum(conv_result) * delta)  # Normalize convolved PDF

    # Compute convolved CDF
    conv_cdf = cumsum(conv_result) * delta
    conv_cdf ./= conv_cdf[end]  # Normalize to reach 1

    return x, conv_result, conv_cdf, pdfs, cdfs
end

# plot_convolution(x, conv_pdf, conv_cdf, pdfs, cdfs, labels::Vector{String})
# Generate side-by-side plots of the convolved PDF and CDF, overlaid with individual PDFs and CDFs.
#
# Arguments:
# - x: Grid points.
# - conv_pdf: Convolved PDF.
# - conv_cdf: Convolved CDF.
# - pdfs: Matrix of individual PDFs.
# - cdfs: Matrix of individual CDFs.
# - labels: Vector of strings to label each distribution.
#
# Example:
# x, conv_pdf, conv_cdf, pdfs, cdfs = convolve_distributions(dists=[Exponential(1.0), Normal(10.0, 3.0)])
# labels = ["Exponential(1.0)", "Normal(10.0, 3.0)"]
# plot_convolution(x, conv_pdf, conv_cdf, pdfs, cdfs, labels)
function plot_convolution(x, conv_pdf, conv_cdf, pdfs, cdfs, labels::Vector{String})
    K = size(pdfs, 2)
    if K != length(labels)
        error("Number of labels must match number of distributions")
    end

    # Plot PDFs
    p1 = plot(x, conv_pdf,
              label="Convolution PDF",
              xlabel="x",
              ylabel="Density",
              title="PDFs",
              lw=2,
              legend=:topright)
    for k in 1:K
        plot!(p1, x, pdfs[:, k], label=labels[k], lw=1, ls=:dash)
    end

    # Plot CDFs
    p2 = plot(x, conv_cdf,
              label="Convolution CDF",
              xlabel="x",
              ylabel="Cumulative Probability",
              title="CDFs",
              lw=2,
              legend=:bottomright)
    for k in 1:K
        plot!(p2, x, cdfs[:, k], label=labels[k], lw=1, ls=:dash)
    end

    # Display plots
    plot(p1, p2, layout=(1, 2), size=(1000, 400))
end

end # module