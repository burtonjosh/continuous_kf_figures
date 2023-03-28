loading_path = string(@__DIR__, "/../../data/")
saving_path_figure = string(@__DIR__, "/../../output/figures/")
saving_path_data = string(@__DIR__, "/../../output/data/")

using StatsPlots, Random, DelimitedFiles, DelayedKalmanFilter, ForwardDiff, BenchmarkTools
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

# define central finite difference method
function central_fdm(protein_observations, params, measurement_std, h)
    gradients = zeros(length(params))

    for index in 1:length(params)
        diff_params = copy(params)
        diff_params[index] += h / 2
        forward = calculate_log_likelihood_at_parameter_point(
            protein_observations,
            diff_params,
            measurement_std^2;
            off_diagonal_steps=29
        )

        diff_params[index] -= h
        backward = calculate_log_likelihood_at_parameter_point(
            protein_observations,
            diff_params,
            measurement_std^2;
            off_diagonal_steps=29
        )
        gradients[index] = (forward - backward) / h
    end
    gradients
end

f = x -> calculate_log_likelihood_at_parameter_point(
    protein_observations,
    x,
    measurement_std^2;
    off_diagonal_steps = ceil(Int, x[end])
)

g = x -> ForwardDiff.gradient(f, x)

auto_diff_grad = g(p)
central_gradients = [central_fdm(protein_observations, p, measurement_std, h) for h in 10.0 .^(-7:-2)]

labels = ["h = 1e-7", "h = 1e-6", "h = 1e-5", "h = 1e-4", "h = 1e-3", "h = 1e-2"]
ctg = repeat(labels, inner = 7)
nam = repeat(["P₀", "h", "μₘ", "μₚ", "αₘ", "αₚ", "τ"], outer = length(labels))

collected_gradients = hcat(
    central_gradients...
)

errors = abs.(collected_gradients .- auto_diff_grad)
groupedbar(
    nam,
    errors,
    group = ctg,
    xlabel = "Parameters",
    ylabel = "Derivative error",
    bar_width = 0.67,
    lw = 0,
    framestyle = :box,
    yaxis=:log,
    legend=:topleft,
    # color = [TolVibrantBlue TolVibrantMagenta TolVibrantOrange]
)
savefig(string(saving_path_figure, "auto_diff_error_new.pdf"))



auto_diff_b = @benchmark g(p)
finite_diff_2_b = @benchmark grad(central_fdm(2,1), f, p)
finite_diff_3_b = @benchmark grad(central_fdm(3,1), f, p)
finite_diff_4_b = @benchmark grad(central_fdm(4,1), f, p)


auto_diff_median = median(auto_diff_b).time / 10e8
finite_diff_2_median = median(finite_diff_2_b).time / 10e8
finite_diff_3_median = median(finite_diff_3_b).time / 10e8
finite_diff_4_median = median(finite_diff_4_b).time / 10e8

medians = [
    auto_diff_median,
    finite_diff_2_median,
    finite_diff_3_median,
    finite_diff_4_median
    ]
names = [
    "Forward mode AD",
    "CFD (2nd order)",
    "CFD (3rd order)",
    "CFD (4th order)"
]

bar(
    names,
    medians,
    label = false,
    color = TolVibrantBlue,
    yaxis=:log
    )

plot!(
    xlabel="Differentiation method",
    ylabel= "Median benchmark (seconds)"
)

savefig(string(saving_path_figure, "auto_diff_benchmark.pdf"))