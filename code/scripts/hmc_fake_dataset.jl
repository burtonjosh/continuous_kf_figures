loading_path = string(@__DIR__, "/../../data/")
saving_path_figure = string(@__DIR__, "/../../output/figures/")
saving_path_data = string(@__DIR__, "/../../output/data/")

using StatsPlots, Random, DelimitedFiles, DelayedKalmanFilter
using Turing, LinearAlgebra, BenchmarkTools, Optim, Pathfinder
using DifferentialEquations
Random.seed!(25)

begin
    black = RGBA{Float64}(colorant"#000000")
    TolVibrantOrange = RGBA{Float64}(colorant"#EE7733")
    TolVibrantBlue = RGBA{Float64}(colorant"#0077BB")
    TolVibrantMagenta = RGBA{Float64}(colorant"#EE3377")
end;

# load data
protein_observations = readdlm(string(saving_path_data,"fake_protein_observations.csv"))
measurement_std = readdlm(string(saving_path_data,"fake_measurement_std.csv"))[1]
p = readdlm(string(saving_path_data,"true_params.csv"))

@model function kf_repression(data, times, repression_mean, measurement_variance)
    P₀ ~ truncated(Normal(repression_mean, 500^2); lower=1000., upper=repression_mean*2)
    # h ~ truncated(Normal(4, 2); lower=2., upper=6.)
    # τ ~ Categorical(40)

    _, distributions = kalman_filter(
        hcat(times, data),
        [P₀, 5.17, log(2)/30, log(2)/90, 15.86, 1.27, 30.],
        measurement_variance;
        alg=RK4(),
        off_diagonal_steps = 3,
        euler_dt=10.0
    )
    data ~ MvNormal(distributions[:,1], diagm(distributions[:,2]))
end

@model function kf_test(data, times, repression_mean, measurement_variance)
    P₀ ~ truncated(Normal(repression_mean, 500^2); lower=1000., upper=repression_mean*2)
    h ~ truncated(Normal(4, 2); lower=2., upper=6.)
    τ ~ truncated(Normal(18, 10); lower=1., upper=40.)# τ ~ Categorical(40)
    
    n_steps = ceil(Int, τ/10.0) + 1
    _, distributions = kalman_filter(
        hcat(times, data),
        [P₀, h, log(2)/30, log(2)/90, 15.86, 1.27, τ],
        measurement_variance;
        alg=RK4(),
        off_diagonal_steps = n_steps,
        euler_dt=10.0
    )

    Turing.@addlogprob! logpdf(MvNormal(distributions[:,1], diagm(distributions[:,2])), data)
end

model = kf_repression(
    protein_observations[:,2],
    protein_observations[:,1],
    mean(protein_observations[:,2]),
    measurement_std^2
)

test_model = kf_test(
    protein_observations[:,2],
    protein_observations[:,1],
    mean(protein_observations[:,2]),
    measurement_std^2
)
sample(test_model, NUTS(), 200)

result_single = pathfinder(test_model; ndraws=8_000, optimizer=LBFGS(m=6), init_scale=10)
plot(result_single.draws_transformed)

result_multi = multipathfinder(test_model, 8_000; nruns=20, init_scale=10, optimizer=LBFGS(m=6))
plot(result_multi.draws_transformed)

mle_estimate = optimize(model, MLE())
writedlm(string(saving_path_data,"repression_mle.csv"), mle_estimate.values.array)
# mle_estimate = readdlm(string(saving_path_data,"repression_mle.csv"))

chn = sample(model, NUTS(0.8), MCMCThreads(), 300, 4)#, init_params = repeat(mle_estimate.values.array,4))
plot(chn)

summarystats(chn)
# save the chain with all internals and info on run time
write(string(saving_path_data,"repression_chain_rk4.jls"), chn)

# code for loading chains
chn2 = read(string(saving_path_data, "repression_chain.jls"), Chains)

## another option for saving and loading chains
## (this loses some information e.g. runtime)
# using MCMCChainsStorage, HDF5
# h5open(string(saving_path_data,"repression_chain.h5"), "w") do f
#     write(f, chn)
# end

# chn_test = h5open(string(saving_path_data,"repression_chain.h5")) do f
#     read(f, Chains)
# end

# make figures
plot(chn, seriestype = :traceplot)
savefig(string(saving_path_figure, "repression_inference_traceplot.pdf"))

density(chn, lw=2)#, color=TolVibrantBlue, lw=2, label=false)
vline!([3407.99], color=TolVibrantBlue, lw=2, line=:dash, label="Ground truth", legend=:topright)
savefig(string(saving_path_figure, "repression_inference_density.pdf"))

vline!([mle_estimate.values.array], color=TolVibrantOrange, lw=2, line=:dash, label="MLE")
savefig(string(saving_path_figure, "repression_inference_density_mle.pdf"))

autocorplot(chn, lw=2, legend=:topright)
savefig(string(saving_path_figure, "repression_inference_autocor.pdf"))

# all params
@model function kf_all_params(data, times, repression_mean, measurement_variance)
    P₀ ~ truncated(Normal(repression_mean, 500^2); lower=1000., upper=repression_mean*2)
    h ~ truncated(Normal(4, 2); lower=2., upper=6.)
    αₘ ~ LogUniform(0.01, 60.0)
    αₚ ~ LogUniform(1.0, 40.0)
    τ ~ truncated(Normal(18,10); lower=5., upper=40.)

    n_steps = ceil(Int, τ/10) + 1

    try
        _, distributions = kalman_filter(
            hcat(times, data),
            [P₀, h, log(2)/30, log(2)/90, αₘ, αₚ, τ],
            measurement_variance;
            alg=RK4(),
            off_diagonal_steps=n_steps,
            euler_dt=10.0,
        )

        data ~ MvNormal(distributions[:,1], diagm(distributions[:,2]))
    catch e
        Turing.@addlogprob! -Inf
    end

end

model = kf_all_params(
    protein_observations[:,2],
    protein_observations[:,1],
    mean(protein_observations[:,2]),
    measurement_std^2
)

rand(model)

# mle 
mle_estimate = optimize(model, MLE())
writedlm(string(saving_path_data,"multiple_params_mle.csv"), mle_estimate.values.array)
readdlm(string(saving_path_data,"multiple_params_mle.csv"))

chn = sample(model, NUTS(0.8), MCMCThreads(), 50, 4)#, init_params = fill(mle_estimate.values.array, 4))
plot(chn)
write(string(saving_path_data,"all_5_params_chain.jls"), chn)

chn = read(string(saving_path_data, "all_params_chain.jls"), Chains)

plot(chn, seriestype = :traceplot, leftmargin = 5Plots.mm)
savefig(string(saving_path_figure, "all_params_inference_traceplot.pdf"))

chain_label = ["Chain 1" "Chain 2" "Chain 3" "Chain 4"]

# P₀
density(chn[:P₀], lw=2, label=chain_label)
vline!([3407.99], color=TolVibrantBlue, lw=3, line=:dash, label="Ground truth", legend=:topleft, alpha=0.7)
plot!(title="P₀", xlabel="Parameter values", ylabel = "Density")
savefig(string(saving_path_figure, "all_params_repression_inference_density.pdf"))

vline!([mle_estimate.values.array[1]], color=TolVibrantOrange, lw=2, label="MLE", ls=:dashdot, alpha=0.7)
savefig(string(saving_path_figure, "all_params_repression_inference_density_mle.pdf"))

# h
density(chn[:h], lw=2, label=chain_label)
vline!([5.17], color=TolVibrantBlue, lw=3, line=:dash, label="Ground truth", legend=:topleft, alpha=0.7)
plot!(title="h", xlabel="Parameter values", ylabel = "Density")
savefig(string(saving_path_figure, "all_params_hill_inference_density.pdf"))

vline!([mle_estimate.values.array[2]], color=TolVibrantOrange, lw=2, label="MLE", ls=:dashdot, alpha=0.7)
savefig(string(saving_path_figure, "all_params_hill_inference_density_mle.pdf"))

# τ
density(chn[:τ], lw=2, label=chain_label)
vline!([30.], color=TolVibrantBlue, lw=3, line=:dash, label="Ground truth", legend=:topleft, alpha=0.7)
plot!(title="τ", xlabel="Parameter values", ylabel = "Density")
savefig(string(saving_path_figure, "all_params_delay_inference_density.pdf"))

vline!([mle_estimate.values.array[3]], color=TolVibrantOrange, lw=2, label="MLE", ls=:dashdot, alpha=0.7)
savefig(string(saving_path_figure, "all_params_delay_inference_density_mle.pdf"))

# autocor
autocorplot(chn, lw=2, legend=true, leftmargin=5*Plots.mm)
savefig(string(saving_path_figure, "all_params_inference_autocor.pdf"))

## pathfinder
result_single = pathfinder(model; ndraws=8_000, optimizer=LBFGS(m=6), init_scale=10)
write(string(saving_path_data,"single_pathfinder_all_params.jls"), result_single.draws_transformed)
# plot(result_single.draws_transformed)

# plot vs mcmc
density(chn, color=TolVibrantBlue, lw=2, label=["MCMC" nothing nothing nothing], margin=5Plots.mm)
density!(result_single.draws_transformed, color=TolVibrantMagenta, lw=2, label="Pathfinder", legend=:topleft)
savefig(string(saving_path_figure, "pathfinder_vs_mcmc.pdf"))

# multipathfinder
result_multi = multipathfinder(model, 2_000; nruns=8, optimizer=LBFGS(m=6), init_scale=10)
result_multi.draws_transformed
plot(result_multi.draws_transformed)
chn2 = read(string(saving_path_data, "single_pathfinder.jls"), Chains)

arr = Array(chn)
arr2 = Array(chn2)

Chains(vcat(arr2,arr))
plot(Chains(arr))
plot!(Chains(arr2),alpha=0.3)
 