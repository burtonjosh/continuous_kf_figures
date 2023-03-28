loading_path = string(@__DIR__, "/../../data/")
saving_path_figure = string(@__DIR__, "/../../output/figures/")
saving_path_data = string(@__DIR__, "/../../output/data/")

using StatsPlots, Random, DelimitedFiles, DelayedKalmanFilter, StochasticDelayDiffEq, Statistics
using Turing, LinearAlgebra, BenchmarkTools, Optim, Pathfinder
using DifferentialEquations
using HDF5, MCMCChains, MCMCChainsStorage
Random.seed!(25)

begin
    black = RGBA{Float64}(colorant"#000000")
    TolVibrantOrange = RGBA{Float64}(colorant"#EE7733")
    TolVibrantBlue = RGBA{Float64}(colorant"#0077BB")
    TolVibrantMagenta = RGBA{Float64}(colorant"#EE3377")
end;

hillr(X, v, K, n) = v * (K^n) / (X^n + K^n)

function hes_model_drift(du,u,h,p,t)
    P₀, n, μₘ, μₚ, αₘ, αₚ, τ = p
    du[1] = hillr(h(p,t-τ;idxs=2),αₘ,P₀,n) - μₘ*u[1]
    du[2] = αₚ*u[1] - μₚ*u[2]
end

function hes_model_noise(du,u,h,p,t)
    P₀, n, μₘ, μₚ, αₘ, αₚ, τ = p
    du[1] = sqrt(max(0.,hillr(h(p,t-τ;idxs=2),αₘ,P₀,n) + μₘ*u[1]))
    du[2] = sqrt(max(0.,αₚ*u[1] + μₚ*u[2]))
end

h(p, t; idxs::Int) = 1.0;

# p = [3407.99, 5.17, log(2)/30, log(2)/90, 15.86, 1.27, 30.];
tspan=(0.,1720.);

function make_protein(p, measurement_std; init = [30., 500.])
    prob = SDDEProblem(hes_model_drift, hes_model_noise, init, h, tspan, p; saveat=10);
    sol = solve(prob,RKMilCommute());

    unobserved_data = Array(sol)[:,100:end];
    # measurement_std = 0.1*mean(unobserved_data[2,:])

    protein = unobserved_data[2,:] + measurement_std*randn(length(unobserved_data[2,:]));
    times = 0:10:730

    return hcat(times,protein)
end

measurement_std = 600.0

chain_1 = h5open(string(saving_path_data, "1_cells_three_params_single.h5"), "r") do f
    read(f, Chains)
end

chain_10 = h5open(string(saving_path_data, "10_cells_three_params_single.h5"), "r") do f
    read(f, Chains)
end

chain_20 = h5open(string(saving_path_data, "20_cells_three_params_single.h5"), "r") do f
    read(f, Chains)
end

chain_30 = h5open(string(saving_path_data, "30_cells_three_params_single.h5"), "r") do f
    read(f, Chains)
end

chain_40 = h5open(string(saving_path_data, "40_cells_three_params_single.h5"), "r") do f
    read(f, Chains)
end

function posterior_predictive_plot_sync(chain)
    samples = Array(chain)
    plot()
    for sample in eachrow(samples)
        p = [sample[1], sample[2], log(2)/30, log(2)/90, 15.86, 1.27, sample[3]];
        post_protein = make_protein(p, measurement_std)
        plot!(post_protein[:,1], post_protein[:,2], label=false, color=TolVibrantBlue, alpha=0.1)
    end
    plot!([6000], label="Posterior prediction", color=TolVibrantBlue, lw=2)
    true_p = [3407.99, 5.17, log(2)/30, log(2)/90, 15.86, 1.27, 30.]
    true_protein = make_protein(true_p, measurement_std)
    plot!(true_protein[:,1],true_protein[:,2], label="True parameters", color=TolVibrantMagenta, lw=2)

    plot!(
        xlabel="Time (minutes)",
        ylabel="Protein molecule number",
        title="40 Cells"
    )
end

posterior_predictive_plot_sync(chain_40)
savefig(string(saving_path_figure,"posterior_predictive_chain_40.pdf"))

# function posterior_predictive_plot(chain)
#     samples = Array(chain)
#     plot()
#     for sample in eachrow(samples)
#         p = [sample[1], sample[2], log(2)/30, log(2)/90, 15.86, 1.27, sample[3]];
#         init = float.([rand(1:1000,1)[1],rand(10:100000,1)[1]])
#         post_protein = make_protein(p, measurement_std; init=init)
#         plot!(post_protein[:,1], post_protein[:,2], label=false, color=TolVibrantBlue, alpha=0.1)
#     end
#     plot!([6000], label="Posterior prediction", color=TolVibrantBlue, lw=2)
#     true_p = [3407.99, 5.17, log(2)/30, log(2)/90, 15.86, 1.27, 30.]
#     true_protein = make_protein(true_p, measurement_std)
#     plot!(true_protein[:,1],true_protein[:,2], label="True parameters", color=TolVibrantMagenta, lw=2)

#     plot!(
#         xlabel="Time (minutes)",
#         ylabel="Protein molecule number",
#         title="40 Cells"
#     )
# end

# posterior_predictive_plot(chain_1)