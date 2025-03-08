using Distributions
using FFTW
using Plots
using KernelDensity

# Define distributions
dists = [Exponential(1.0), Normal(10.0, 3.0), Gamma(5.0, 2.0)]
K = length(dists)

# Grid parameters
total_mean = sum(mean, dists)  # ~21.0
total_std = sqrt(sum(var, dists))  # ~√20 ≈ 4.47
N = 2^12  # 4096 points
sigma_range = 4
x_min = total_mean - sigma_range * total_std  # ~7.6
x_max = total_mean + sigma_range * total_std  # ~34.4
x = range(x_min, x_max, length=N)
delta = (x_max - x_min) / (N - 1)

# Discretize PDFs and compute CDFs for individual distributions
pdfs = zeros(N, K)
cdfs = zeros(N, K)
for k in 1:K
    pdf_vals = pdf(dists[k], x)
    pdf_vals ./= (sum(pdf_vals) * delta)  # Normalize to integrate to 1
    pdfs[:, k] = pdf_vals
    cdfs[:, k] = cdf(dists[k], x)  # Exact CDF from Distributions.jl
end

# FFT Convolution for PDF
fft_pdfs = fft(pdfs, 1)
fft_result = prod(fft_pdfs, dims=2)
conv_result = real(ifft(fft_result, 1))[:, 1]
conv_result ./= (sum(conv_result) * delta)  # Normalize convolution PDF

sum(conv_result) * delta

# Compute convolution CDF
conv_cdf = cumsum(conv_result) * delta
conv_cdf ./= conv_cdf[end]  # Normalize to reach 1

# Print mean for verification
conv_mean = sum(x .* conv_result) * delta
println("Mean of normalized conv_result: ", conv_mean)

# Plot PDFs
p1 = plot(x, conv_result, 
          label="Convolution PDF",
          xlabel="x",
          ylabel="Density",
          title="PDFs",
          lw=2,
          legend=:topright)
plot!(p1, x, pdfs[:, 1], label="Exponential(1.0)", lw=1, ls=:dash)
plot!(p1, x, pdfs[:, 2], label="Normal(10.0, 3.0)", lw=1, ls=:dash)
plot!(p1, x, pdfs[:, 3], label="Gamma(5.0, 2.0)", lw=1, ls=:dash)

# Plot CDFs
p2 = plot(x, conv_cdf, 
          label="Convolution CDF",
          xlabel="x",
          ylabel="Cumulative Probability",
          title="CDFs",
          lw=2,
          legend=:bottomright)
plot!(p2, x, cdfs[:, 1], label="Exponential(1.0)", lw=1, ls=:dash)
plot!(p2, x, cdfs[:, 2], label="Normal(10.0, 3.0)", lw=1, ls=:dash)
plot!(p2, x, cdfs[:, 3], label="Gamma(5.0, 2.0)", lw=1, ls=:dash)

# Display both plots side by side
plot(p1, p2, layout=(1, 2), size=(1000, 400))