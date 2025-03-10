# Include the module
include("DistributionConvolution.jl")
using .DistributionConvolution
using Distributions
using KernelDensity

# Define theoretical distributions (to generate samples)
theoretical_dists = [Exponential(1.0), Normal(10.0, 3.0), Gamma(5.0, 2.0)]
labels = ["Exponential(1.0)", "Normal(10.0, 3.0)", "Gamma(5.0, 2.0)"]
K = length(theoretical_dists)

# Parameters for grid and sampling
N = 2^12  # 4096 points
n_samples = 10000  # Number of samples per distribution
sigma_range = 4

# Calculate total mean and std for grid (using theoretical distributions)
total_mean = sum(mean, theoretical_dists)
total_var = sum(var, theoretical_dists)
total_std = sqrt(total_var)
x_min = total_mean - sigma_range * total_std
x_max = total_mean + sigma_range * total_std
x = range(x_min, x_max, length=N)
delta = (x_max - x_min) / (N - 1)

# Sample data and create KDEs
pdfs = zeros(N, K)
cdfs = zeros(N, K)
for k in 1:K
    # Sample data
    samples = rand(theoretical_dists[k], n_samples)
    # Compute KDE
    kde_obj = kde(samples, x)
    pdf_vals = kde_obj.density
    pdf_vals ./= (sum(pdf_vals) * delta)  # Normalize KDE PDF
    pdfs[:, k] = pdf_vals
    # Compute exact CDF for comparison (or approximate via KDE if desired)
    cdfs[:, k] = cdf(theoretical_dists[k], x)
end

# Compute convolution using precomputed pdfs/cdfs/x
x, conv_pdf, conv_cdf, pdfs, cdfs = convolve_distributions(pdfs=pdfs, cdfs=cdfs, x=x)

# Verify normalization and mean
integral = sum(conv_pdf) * delta
conv_mean = sum(x .* conv_pdf) * delta
println("Integral of convolved PDF (should be ~1): ", integral)
println("Mean of convolved PDF (should be ~21): ", conv_mean)

# Plot results
plot_convolution(x, conv_pdf, conv_cdf, pdfs, cdfs, labels)