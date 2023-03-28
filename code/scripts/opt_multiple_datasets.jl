loading_path = string(@__DIR__, "/../../data/selected_data_for_inference/")
saving_path_figure = string(@__DIR__, "/../../output/figures/")
saving_path_data = string(@__DIR__, "/../../output/data/")

using StatsPlots, Random, DelimitedFiles, DelayedKalmanFilter
using Turing, LinearAlgebra, BenchmarkTools, Optim, DifferentialEquations
# Random.seed!(25)

begin
    black = RGBA{Float64}(colorant"#000000")
    TolVibrantOrange = RGBA{Float64}(colorant"#EE7733")
    TolVibrantBlue = RGBA{Float64}(colorant"#0077BB")
    TolVibrantMagenta = RGBA{Float64}(colorant"#EE3377")
end;

# load data
dataset_strings = [i for i in readdir(loading_path) if !contains(i, "measurement_variance")]
measurement_variance_strings = [i for i in readdir(loading_path) if contains(i, "measurement_variance")]
experiment_strings = ["040417", "280317p1", "280317p6"]


@model function kf_repression_multiple(data, times, repression_mean, measurement_std)
    P₀ ~ truncated(Normal(repression_mean, 500^2); lower=100, upper=repression_mean*2)
    h ~ truncated(Normal(4, 2); lower=2, upper=6)
    αₘ ~ LogUniform(0.01, 60)
    αₚ ~ LogUniform(1, 40)
    τ ~ truncated(Normal(18, 10); lower=5., upper=40.)

    n_steps = ceil(Int, τ/5.0) + 1

    try
        _, distributions = kalman_filter(
            hcat(times, data),
            [P₀, h, log(2)/30, log(2)/90, αₘ, αₚ, τ],
            measurement_std^2;
            alg=RK4(),
            off_diagonal_steps=n_steps,
            euler_dt=5.0
        )
        # println(logpdf(MvNormal(distributions[:,1], diagm(distributions[:,2])), data))
        data ~ MvNormal(distributions[:,1], diagm(distributions[:,2]))
    catch e
        # println(e)
        Turing.@addlogprob! -Inf
    end
end

# @model function kf_repression_multiple(datasets, repression_mean, measurement_std)
#     P₀ ~ truncated(Normal(repression_mean, 500^2); lower=1000., upper=repression_mean*2)
#     h ~ truncated(Normal(4, 2); lower=2., upper=6.)
#     αₘ ~ LogUniform(0.01, 60.0)
#     αₚ ~ LogUniform(1.0, 40.0)
#     τ ~ truncated(Normal(18, 10); lower=5., upper=40.)

#     n_steps = ceil(Int, τ/15.0) + 1

#     for data in datasets
#         try
#             Turing.@addlogprob! calculate_log_likelihood_at_parameter_point(
#                 data,
#                 [P₀, h, log(2)/30, log(2)/90, αₘ, αₚ, τ],
#                 measurement_std^2;
#                 alg=RK4(),
#                 off_diagonal_steps=n_steps,
#                 euler_dt=15.0
#             )
#         catch e
#             Turing.@addlogprob! -Inf
#             return nothing
#         end
#     end
#     return nothing
# end

datasets = [readdlm(string(loading_path,i)) for i in dataset_strings if contains(i, "protein_observations_040417") && contains(i,"cluster_1")]
repression_mean = mean(last.(mean.([datasets[1]], dims=1)))
measurement_std = readdlm(string(loading_path,"040417_measurement_variance_detrended.csv"))[1]

# clean up
for dataset in datasets
    dataset[:,1] .-= dataset[1,1]
end
times = datasets[1][:,1]
model = kf_repression_multiple(datasets[1][:,2], times, repression_mean, measurement_std)
rand(model)
mle_estimate = optimize(model, MLE())

# writedlm(string(saving_path_data,"repression_multiple_mle.csv"), mle_estimate.values.array)
test_chn = sample(model, NUTS(0.9), 50, init_params=mle_estimate.values.array)
using Pathfinder

single_result = pathfinder(model; ndraws=4_000, optimizer=LBFGS(m=6), init_scale=10)
plot(single_result.draws_transformed)
# chn = single_result.draws_transformed
# chn2 = single_result.draws_transformed
# plot(chn)
# write(string(saving_path_data,"multiple_repression_chain_singlepath.jls"), chn)

multi_result = multipathfinder(model, 4_000; nruns=4, optimizer=LBFGS(m=6), init_scale=10)
# plot(multi_result.draws_transformed)
# chn_multi = mutiple_repression_single_result.draws_transformed
# write(string(saving_path_data,"multiple_repression_chain_multipath.jls"), chn_multi)

# # make figures
# p1 = density(chn, xlim=(3200, 3700))
# p2 = density(chn_multi, xlim=(3200, 3700))
# plot(p1, p2, layout=(2,1), size=(400,450))

# plot(chn, color=TolVibrantBlue, linewidth=2, seriestype = :traceplot)
# savefig(string(saving_path_figure, "repression_inference_traceplot.pdf"))

# density(chn, color=TolVibrantBlue, lw=2, label=false)
# vline!([3407.99], color=TolVibrantMagenta, lw=2, label="Ground truth", legend=:topright)
# savefig(string(saving_path_figure, "repression_inference_density.pdf"))

# vline!([mle_estimate.values.array], color=TolVibrantOrange, lw=2, label="MLE")
# savefig(string(saving_path_figure, "repression_inference_density_mle.pdf"))

# autocorplot(chn, color=TolVibrantBlue, lw=2)
# savefig(string(saving_path_figure, "repression_inference_autocor.pdf"))

# # all params
# @model function kf_all_params(data, times, repression_mean, measurement_variance)
#     P₀ ~ truncated(Normal(repression_mean, 500^2); lower=1000., upper=repression_mean*2)
#     h ~ truncated(Normal(4, 2); lower=2., upper=6.)
#     τ ~ truncated(Normal(18,10); lower=1., upper=40.)

#     _, distributions = kalman_filter(
#         hcat(times, data),
#         [P₀, h, log(2)/30, log(2)/90, 15.86, 1.27, τ],
#         measurement_variance
#     )
#     data ~ MvNormal(distributions[:,1], diagm(distributions[:,2]))
# end

# model = kf_all_params(
#     protein_observations[:,2],
#     protein_observations[:,1],
#     mean(protein_observations[:,2]),
#     measurement_std^2
# )

# mle_estimate = optimize(model, MLE())
# writedlm(string(saving_path_data,"multiple_params_mle.csv"), mle_estimate.values.array)

# chn = sample(model, NUTS(0.8), 150, init_params = mle_estimate.values.array)
# write(string(saving_path_data,"all_params_chain.jls"), chn)

# plot(chn, color=TolVibrantBlue, linewidth=2, seriestype = :traceplot)
# savefig(string(saving_path_figure, "all_params_inference_traceplot.pdf"))

# # P₀
# density(chn[:P₀], color=TolVibrantBlue, lw=2, label=false)
# vline!([3407.99], color=TolVibrantMagenta, lw=2, label="Ground truth", legend=:topright)
# plot!(title="P₀", xlabel="Parameter values", ylabel = "Density")
# savefig(string(saving_path_figure, "all_params_repression_inference_density.pdf"))

# vline!([mle_estimate.values.array[1]], color=TolVibrantOrange, lw=2, label="MLE", ls=:dash)
# savefig(string(saving_path_figure, "all_params_repression_inference_density_mle.pdf"))

# # h
# density(chn[:h], color=TolVibrantBlue, lw=2, label=false)
# vline!([5.17], color=TolVibrantMagenta, lw=2, label="Ground truth", legend=:topright)
# plot!(title="h", xlabel="Parameter values", ylabel = "Density")
# savefig(string(saving_path_figure, "all_params_hill_inference_density.pdf"))

# vline!([mle_estimate.values.array[2]], color=TolVibrantOrange, lw=2, label="MLE", ls=:dash)
# savefig(string(saving_path_figure, "all_params_hill_inference_density_mle.pdf"))

# # τ
# density(chn[:τ], color=TolVibrantBlue, lw=2, label=false)
# vline!([30.], color=TolVibrantMagenta, lw=2, label="Ground truth", legend=:topright)
# plot!(title="τ", xlabel="Parameter values", ylabel = "Density")
# savefig(string(saving_path_figure, "all_params_delay_inference_density.pdf"))

# vline!([mle_estimate.values.array[3]], color=TolVibrantOrange, lw=2, label="MLE", ls=:dash)
# savefig(string(saving_path_figure, "all_params_delay_inference_density_mle.pdf"))


# autocorplot(chn, color=TolVibrantBlue, lw=2)
# savefig(string(saving_path_figure, "all_params_inference_autocor.pdf"))

# using Pathfinder

# result_single = pathfinder(model; ndraws=2_000, optimizer=LBFGS(m=6))