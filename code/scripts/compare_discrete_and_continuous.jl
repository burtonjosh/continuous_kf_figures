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

# define values of τ, use round to simulate discrete τ
time_delays = LinRange(25.,40.,40)
discrete_delays_1 = round.(time_delays)
discrete_delays_2 = round.(time_delays*2) ./2


many_p_continuous = [[p[1:6]...,i] for i in time_delays]
many_p_discrete_1 = [[p[1:6]...,i] for i in discrete_delays_1]
many_p_discrete_2 = [[p[1:6]...,i] for i in discrete_delays_2]

if isfile(string(saving_path_data,"ll_delays_continuous_1.csv"))
    ll_delays_continuous_1 = readdlm(string(saving_path_data,"ll_delays_continuous_1.csv"))
    ll_delays_continuous_2 = readdlm(string(saving_path_data,"ll_delays_continuous_2.csv"))
    ll_delays_discrete_1 = readdlm(string(saving_path_data,"ll_delays_discrete_1.csv"))
    ll_delays_discrete_2 = readdlm(string(saving_path_data,"ll_delays_discrete_2.csv"))
else
    ll_delays_continuous_1 = [calculate_log_likelihood_at_parameter_point(protein_observations, p, measurement_std^2; off_diagonal_steps=ceil(Int, p[end])) for p in many_p_continuous]
    ll_delays_continuous_2 = [calculate_log_likelihood_at_parameter_point(protein_observations, p, measurement_std^2; off_diagonal_steps=ceil(Int, p[end]*2), euler_dt=0.5) for p in many_p_continuous]
    ll_delays_discrete_1 = [calculate_log_likelihood_at_parameter_point(protein_observations, p, measurement_std^2; off_diagonal_steps=ceil(Int, p[end])) for p in many_p_discrete_1]
    ll_delays_discrete_2 = [calculate_log_likelihood_at_parameter_point(protein_observations, p, measurement_std^2; off_diagonal_steps=ceil(Int, p[end]*2), euler_dt=0.5) for p in many_p_discrete_2]

    writedlm(string(saving_path_data, "ll_delays_continuous_1.csv"), ll_delays_continuous_1)
    writedlm(string(saving_path_data, "ll_delays_continuous_2.csv"), ll_delays_continuous_2)
    writedlm(string(saving_path_data, "ll_delays_discrete_1.csv"), ll_delays_discrete_1)
    writedlm(string(saving_path_data, "ll_delays_discrete_2.csv"), ll_delays_discrete_2)
end



plot(
    time_delays,
    ll_delays_continuous_1,
    label="Continuous τ (\\Deltat = 1.0)",
    linewidth=2,
    color=TolVibrantBlue
)

plot!(
    time_delays,
    ll_delays_discrete_1,
    label="Discrete τ (\\Deltat = 1.0)",
    color=TolVibrantBlue,
    markershape=:circle,
    line=:steppost
)

plot!(
    xlabel="Time delay, τ (minutes)",
    ylabel="log-likelihood"
)

savefig(string(saving_path_figure,"discrete_vs_continuous_10.pdf"))

plot(
    time_delays,
    ll_delays_continuous_2,
    label="Continuous τ (\\Deltat = 0.5)",
    linewidth=2,
    color=TolVibrantMagenta
)

plot!(
    time_delays,
    ll_delays_discrete_2,
    label="Discrete τ (\\Deltat = 0.5)",
    color=TolVibrantMagenta,
    alpha=0.75,
    markershape=:circle,
    line=:steppost
)

plot!(
    xlabel="Time delay, τ (minutes)",
    ylabel="log-likelihood"
)

savefig(string(saving_path_figure,"discrete_vs_continuous_05.pdf"))

plot(
    time_delays,
    ll_delays_discrete_1,
    label="Discrete τ (\\Deltat = 1.0)",
    color=TolVibrantBlue,
    alpha=0.75,
    markershape=:circle,
    line=:steppost
)

plot!(
    time_delays,
    ll_delays_discrete_2,
    label="Discrete τ (\\Deltat = 0.5)",
    color=TolVibrantMagenta,
    alpha=0.75,
    markershape=:circle,
    line=:steppost
)

plot!(
    xlabel="Time delay, τ (minutes)",
    ylabel="log-likelihood"
)

savefig(string(saving_path_figure,"discrete_vs_discrete.pdf"))