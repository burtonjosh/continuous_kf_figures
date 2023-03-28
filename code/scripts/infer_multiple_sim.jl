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

p = [3407.99, 5.17, log(2)/30, log(2)/90, 15.86, 1.27, 30.];
tspan=(0.,1720.);

function make_protein(measurement_std)
    prob = SDDEProblem(hes_model_drift, hes_model_noise, [30.,500.], h, tspan, p; saveat=10);
    sol = solve(prob,RKMilCommute());

    unobserved_data = Array(sol)[:,100:end];
    # measurement_std = 0.1*mean(unobserved_data[2,:])

    protein = unobserved_data[2,:] + measurement_std*randn(length(unobserved_data[2,:]));
    times = 0:10:730

    return hcat(times,protein)
end

# inference 

@model function kf_multiple(datasets, repression_mean, measurement_variance)
    P₀ ~ truncated(Normal(repression_mean, 500^2); lower=1000., upper=repression_mean*2)
    h ~ truncated(Normal(4, 2); lower=2., upper=6.)
    τ ~ truncated(Normal(18, 10); lower=5., upper=40.)# τ ~ Categorical(40)
    
    n_steps = ceil(Int, τ/10.0) + 1

    for data in datasets
        _, distributions = kalman_filter(
            data,
            [P₀, h, log(2)/30, log(2)/90, 15.86, 1.27, τ],
            measurement_variance;
            alg=RK4(),
            off_diagonal_steps = n_steps,
            euler_dt=10.0
        )

        Turing.@addlogprob! logpdf(MvNormal(distributions[:,1], diagm(distributions[:,2])), data[:,2])
    end
end

measurement_std = 600.0
datasets = [make_protein(measurement_std) for _ in 1:40]
repression_mean = mean(last.(mean.(datasets, dims=1)))

multiple_model = kf_multiple(
    datasets,
    repression_mean,
    measurement_std^2
)
# rand(multiple_model)

@time result_single = pathfinder(multiple_model; ndraws=2_000, optimizer=LBFGS(m=6), init_scale=10)
plot(result_single.draws_transformed)

chn = result_single.draws_transformed

density(chn)
density!(chain_10)

h5open(string(saving_path_data, "40_cells_three_params_single.h5"), "w") do f
    write(f, chn)
end

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


density(chain_1[:h], label="1 cell", lw=2)
density!(chain_10[:h], label="10 cells", lw=2)
density!(chain_20[:h], label="20 cells", lw=2)
density!(chain_30[:h], label="30 cells", lw=2)
density!(chain_40[:h], label="40 cells", lw=2)
# vline!([30.0], color=:black, lw=2, line=:dash, label="Ground truth", legend=:topleft)
# vline!([5.17], color=:black, lw=2, line=:dash, label="Ground truth", legend=:topright)
vline!([5.17], color=:black, lw=2, line=:dash, label="Ground truth", legend=:topleft)
plot!(xlabel="Sample value", ylabel="Density",title="h")
savefig(string(saving_path_figure,"compare_multiple_cells_three_params_hill.pdf"))

corner(chain_40)
savefig(string(saving_path_figure,"corner_40.pdf"))