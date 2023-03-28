loading_path = string(@__DIR__, "/../../data/")
saving_path_figure = string(@__DIR__, "/../../output/figures/")
saving_path_data = string(@__DIR__, "/../../output/data/")

using Plots, Random, DelimitedFiles, DelayedKalmanFilter
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

timesteps = collect(1.0:10.0)
ll_values = [
    calculate_log_likelihood_at_parameter_point(
        protein_observations,
        p,
        measurement_std^2;
        off_diagonal_timestep = i
    ) for i in timesteps
]

plot(
    timesteps,
    ll_values,
    linewidth=2,
    color=TolVibrantBlue,
    label=false
)

plot!(
    xlabel="Off-diagonal timestep (minutes)",
    ylabel="log-likelihood"
)

savefig(string(saving_path_figure, "off_diagonal_timestep.pdf"))

plot(
    timesteps,
    (first(ll_values) .- ll_values),
    linewidth=2,
    color=TolVibrantBlue,
    label=false
)

plot!(
    xlabel="Off-diagonal timestep (minutes)",
    ylabel="Error"
)

savefig(string(saving_path_figure, "off_diagonal_timestep_error.pdf"))

plot(
    timesteps,
    (first(ll_values) .- ll_values) ./ abs(first(ll_values)),
    linewidth=2,
    color=TolVibrantBlue,
    label=false
)

plot!(
    xlabel="Off-diagonal timestep (minutes)",
    ylabel="Relative error"
)

savefig(string(saving_path_figure, "off_diagonal_timestep_relative_error.pdf"))


using BenchmarkTools

median_benchmark = zeros(length(timesteps))
for (index, time_step) in enumerate(timesteps)
    b = @benchmark calculate_log_likelihood_at_parameter_point(
        protein_observations,
        p,
        measurement_std^2;
        off_diagonal_timestep = $time_step
    )
    median_benchmark[index] = median(b).time / 10e8
end

median_benchmark

plot(
    timesteps,
    median_benchmark,
    linewidth=2,
    color=TolVibrantBlue,
    label=false
)

plot!(
    xlabel="Off-diagonal timestep (minutes)",
    ylabel="Median benchmark (seconds)"
)

savefig(string(saving_path_figure, "off_diagonal_timestep_benchmark.pdf"))