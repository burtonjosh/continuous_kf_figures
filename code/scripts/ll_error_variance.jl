loading_path = string(@__DIR__, "/../../data/")
saving_path_figure = string(@__DIR__, "/../../output/figures/")
saving_path_data = string(@__DIR__, "/../../output/data/")

using StatsPlots, Random, DelimitedFiles, DelayedKalmanFilter, DifferentialEquations, Distributions
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

repression_mean = mean(protein_observations[:,2])

priors = [
    truncated(Normal(repression_mean, 500^2); lower=1000., upper=repression_mean*1.5),
    truncated(Normal(4, 2); lower=2., upper=6.),
    LogUniform(0.01, 10.0),
    LogUniform(1.0, 10.0),
    truncated(Normal(18,10); lower=5., upper=40.)
]

N_pars = 150
all_params = [rand.(priors) for _ in 1:N_pars]
# writedlm(string(saving_path_data,"all_params_error_variance.csv"),all_params)

if isfile(string(saving_path_data,"ll_error_variance_05.csv"))
    ll_values = readdlm(string(saving_path_data,"ll_error_variance_05.csv"))
else
    ll_values = zeros(
        first(size(all_params)),
        2
    )
    for (i, p) in enumerate(all_params)
        println(p)
        # fast solve
        ll_values[i, 1] = calculate_log_likelihood_at_parameter_point(
            protein_observations,
            [p[1], p[2], log(2)/30, log(2)/90, p[3], p[4], p[5]],
            measurement_std^2;
            alg=RK4(),
            off_diagonal_steps=ceil(Int, (p[end]-1) / 10.0)+1,
            euler_dt = 10.0
        )
        # ground truth
        ll_values[i, 2] = calculate_log_likelihood_at_parameter_point(
            protein_observations,
            [p[1], p[2], log(2)/30, log(2)/90, p[3], p[4], p[5]],
            measurement_std^2;
            alg=RK4(),
            off_diagonal_steps=ceil(Int, (p[end]-1) / 1.0),
            euler_dt = 1.0
        )
    end
    writedlm(string(saving_path_data,"ll_error_variance_10.csv"),ll_values)
end

fast = ll_values[:,1]
truth = ll_values[:,2]

scatter(
    abs.(ll_values[:,2]),
    abs.(fast .- truth) ./ abs.(truth),
    xaxis=:log,
    yaxis=:log,
    xflip=true,
    xticks=([590, 650, 1000, 2000],["-590", "-650", "-1000", "-2000"]),
    label=false
)

plot!(
    xlabel="Log-likelihood",
    ylabel="Relative error"
)

savefig(string(saving_path_figure, "ll_error_variance_05.pdf"))