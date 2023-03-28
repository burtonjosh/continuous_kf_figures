loading_path = string(@__DIR__, "/../../data/")
saving_path_figure = string(@__DIR__, "/../../output/figures/")
saving_path_data = string(@__DIR__, "/../../output/data/")

using Plots, Random, DelimitedFiles

begin
    black = RGBA{Float64}(colorant"#000000")
    TolVibrantOrange = RGBA{Float64}(colorant"#EE7733")
    TolVibrantBlue = RGBA{Float64}(colorant"#0077BB")
    TolVibrantMagenta = RGBA{Float64}(colorant"#EE3377")
end;

# load data
euler_ll = readdlm(string(saving_path_data,"euler_timestep_multiple_diags_ll.csv"))
euler_benchmark = readdlm(string(saving_path_data,"euler_timestep_multiple_diags_benchmark.csv"))
rk4_ll = readdlm(string(saving_path_data,"rk4_timestep_multiple_diags_ll.csv"))
rk4_benchmark = readdlm(string(saving_path_data,"rk4_timestep_multiple_diags_benchmark.csv"))

# what to choose as ground truth?
ground_truth = rk4_ll[1,1]
rk4_ll[1,1] = NaN
rk4_benchmark[1,1] = NaN

scatter(
    [abs(euler_ll[1,1] - ground_truth) / abs(ground_truth)],
    [euler_benchmark[1,1]],
    color=:black,
    marker=:circle,
    label="Euler"
)

scatter!(
    [abs(rk4_ll[1,1] - ground_truth) / abs(ground_truth)],
    [rk4_benchmark[1,1]],
    color=:black,
    marker=:diamond,
    label="RK4"
)

scatter!(
    abs.(euler_ll .- ground_truth) ./ abs(ground_truth),
    euler_benchmark,
    marker=:circle,
    # color=TolVibrantBlue,
    label=false
)
scatter!(
    abs.(rk4_ll .- ground_truth) ./ abs(ground_truth),
    rk4_benchmark,
    marker=:diamond,
    # color=TolVibrantMagenta,
    yaxis=:log,
    xaxis=:log,
    yticks=[0.1,1.0,10.0,100.0],
    xticks=([1e-3,1e-4,1e-5,1e-6, 1e-7]),
    xlim=(4e-8,2e-3),
    label=false
)

plot!(
    xlabel="Relative Error",
    ylabel="Compute Time (s)"
)

savefig(string(saving_path_figure, "relative_error_v_time.pdf"))