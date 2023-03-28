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

euler_timesteps = [0.01, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0 , 10.0]
off_diagonal_steps = ceil.(Int, (p[end]-1) ./ [0.1, 0.2, 0.5, 1.0, 2.0, 5.0 , 10.0])

if isfile(string(saving_path_data,"euler_timestep_multiple_diags_ll.csv"))
    ll_values = readdlm(string(saving_path_data,"euler_timestep_multiple_diags_ll.csv"))
else
    ll_values = zeros(
        length(euler_timesteps),
        length(off_diagonal_steps)
    )

    for (i, euler_timestep) in enumerate(euler_timesteps)
        for (j, diagonal_steps) in enumerate(off_diagonal_steps)
            ll_values[i, j] = calculate_log_likelihood_at_parameter_point(
                protein_observations,
                p,
                measurement_std^2;
                off_diagonal_steps=diagonal_steps,
                euler_dt = euler_timestep
            )
        end
    end
    writedlm(string(saving_path_data,"euler_timestep_multiple_diags_ll.csv"),ll_values)
end

# ground_truth = ll_values[1,1]
ground_truth = readdlm(string(saving_path_data,"rk4_timestep_multiple_diags_ll.csv"))[1,1]
off_diag_label = reshape([
    "\\Deltas = 0.1";
    "\\Deltas = 0.2";
    "\\Deltas = 0.5";
    "\\Deltas = 1.0";
    "\\Deltas = 2.0";
    "\\Deltas = 5.0";
    "\\Deltas = 10.0";
], 1, 7)

plot(
    euler_timesteps,
    ll_values;
    linewidth=2,
    # color=TolVibrantBlue,
    label=off_diag_label,
    xaxis=:log,
    xflip=true,
    xticks=(euler_timesteps, string.(euler_timesteps)),
    legend=:topright
)

plot!(
    xlabel="Forward Euler timestep, \\Deltat (minutes)",
    ylabel="log-likelihood"
)

savefig(string(saving_path_figure, "forward_euler_timestep_multiple_diags.pdf"))

plot(
    euler_timesteps,
    abs.(ground_truth .- ll_values),
    linewidth=2,
    # color=TolVibrantBlue,
    label=off_diag_label,
    xaxis=:log10,
    xflip=true,
    xticks=(euler_timesteps, string.(euler_timesteps)),
    legend=:topright
)

plot!(
    xlabel="Forward Euler timestep (minutes)",
    ylabel="Error"
)

savefig(string(saving_path_figure, "forward_euler_timestep_error_multiple_diags.pdf"))

test_ll_values = copy(ll_values)
# test_ll_values[1,1] = NaN
plot(
    euler_timesteps,
    abs.(ground_truth .- test_ll_values) ./ abs(ground_truth),
    linewidth=3,
    # color=TolVibrantBlue,
    label=off_diag_label,
    xticks=(euler_timesteps,string.(euler_timesteps)),
    yticks=([1e-3,1e-4,1e-5,1e-6, 1e-7]),
    xaxis=:log10,
    yaxis=:log10,
    xflip=true,
    legend=:bottomleft,
    ylim=(4e-8,2e-3)
)

plot!(
    xlabel="Forward Euler timestep (minutes)",
    ylabel="Relative error"
)

savefig(string(saving_path_figure, "forward_euler_timestep_relative_error_multiple_diags.pdf"))


using BenchmarkTools

if isfile(string(saving_path_data,"euler_timestep_multiple_diags_benchmark.csv"))
    median_benchmark = readdlm(string(saving_path_data,"euler_timestep_multiple_diags_benchmark.csv"))
else
    median_benchmark = zeros(
        length(euler_timesteps),
        length(off_diagonal_steps)
    )

    for (i, euler_timestep) in enumerate(euler_timesteps)
        for (j, diagonal_steps) in enumerate(off_diagonal_steps)

            b = @benchmark calculate_log_likelihood_at_parameter_point(
                protein_observations,
                p,
                measurement_std^2;
                off_diagonal_steps=$diagonal_steps,
                euler_dt = $euler_timestep
            )
            median_benchmark[i, j] = median(b).time / 10e8
        end
    end
    writedlm(string(saving_path_data,"euler_timestep_multiple_diags_benchmark.csv"),median_benchmark)
end


plot(
    euler_timesteps,
    median_benchmark,
    linewidth=3,
    # color=TolVibrantBlue,
    label=off_diag_label,
    xticks=(euler_timesteps,string.(euler_timesteps)),
    yticks=[0.1, 1.0, 10.,10^2],
    xaxis=:log10,
    yaxis=:log10,
    xflip=true,
    legend=:topleft
)

plot!(
    xlabel="Forward Euler timestep (minutes)",
    ylabel="Median benchmark (seconds)"
)

savefig(string(saving_path_figure, "forward_euler_timestep_benchmark_multiple_diags.pdf"))