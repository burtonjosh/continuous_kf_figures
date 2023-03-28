loading_path = string(@__DIR__, "/../../data/")
saving_path_figure = string(@__DIR__, "/../../output/figures/")
saving_path_data = string(@__DIR__, "/../../output/data/")

using Plots, Random, DelimitedFiles, DelayedKalmanFilter, DifferentialEquations
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

ode_solvers = [BS3(), RK4(), Tsit5(), Vern9(), Feagin14()]
tolerances = [1/10^i for i in 1:7]

labels = ["BS3", "RK4", "Tsit5", "Vern9", "Feagin14"]
markers = [:circle, :diamond, :dtriangle, :ltriangle, :hexagon]

ll_values = zeros(length(ode_solvers), length(tolerances))

for (solver_index, solver) in enumerate(ode_solvers)
    for (tolerance_index, tolerance) in enumerate(tolerances)
        ll_values[solver_index, tolerance_index] = calculate_log_likelihood_at_parameter_point(
            protein_observations,
            p,
            measurement_std^2;
            alg = solver,
            absolute_tolerance = tolerance,
            relative_tolerance = tolerance
        )

    end
end

plot(legend=:topright)

for i in 1:length(ode_solvers)
    plot!(
    tolerances,
    ll_values[i, :],
    linewidth=2,
    # color=TolVibrantBlue,
    label=labels[i],
    xaxis=:log,
    # yaxis=:log,
    marker=markers[i],
    xticks = tolerances,
    xflip=true
)
end

plot!(
    xlabel="Relative and Absolute Tolerance",
    ylabel="log-likelihood"
)

savefig(string(saving_path_figure, "solver_comparison_all_diagonals.pdf"))

ground_truth = ll_values[end, end]

plot(legend=:topright)

for i in 1:length(ode_solvers)
    if i == length(ode_solvers)
        xvalues = tolerances[1:end-1]
        yvalues = abs.(ground_truth .- ll_values[i, 1:end-1])
    else
        xvalues = tolerances
        yvalues = abs.(ground_truth .- ll_values[i, :])
    end
    
    plot!(
        xvalues,
        yvalues,
        linewidth=2,
        # color=TolVibrantBlue,
        label=labels[i],
        xaxis=:log,
        yaxis=:log,
        marker=markers[i],
        xticks = tolerances,
        xflip=true
    )
end

plot!(
    xlabel="Relative and Absolute Tolerance",
    ylabel="Error"
)

savefig(string(saving_path_figure, "solver_comparison_all_diagonals_error.pdf"))

plot(legend=:topright)

for i in 1:length(ode_solvers)
    if i == length(ode_solvers)
        xvalues = tolerances[1:end-1]
        yvalues = abs.(ground_truth .- ll_values[i, 1:end-1]) ./ abs(ground_truth)
    else
        xvalues = tolerances
        yvalues = abs.(ground_truth .- ll_values[i, :]) ./ abs(ground_truth)
    end

    plot!(
        xvalues,
        yvalues,
        linewidth=2,
        # color=TolVibrantBlue,
        label=labels[i],
        xaxis=:log,
        yaxis=:log,
        marker=markers[i],
        xticks = tolerances,
        xflip=true
    )
end

plot!(
    xlabel="Relative and Absolute Tolerance",
    ylabel="Relative Error"
)

savefig(string(saving_path_figure, "solver_comparison_all_diagonals_relative_error.pdf"))

using BenchmarkTools

median_benchmark = zeros(size(ll_values))

for (solver_index, solver) in enumerate(ode_solvers)
    for (tolerance_index, tolerance) in enumerate(tolerances)
        b = @benchmark calculate_log_likelihood_at_parameter_point(
            protein_observations,
            p,
            measurement_std^2;
            alg=$solver,
            absolute_tolerance=$tolerance,
            relative_tolerance=$tolerance
        )
        median_benchmark[solver_index, tolerance_index] = median(b).time / 10e8
    end
end

plot(legend=:topleft)

for i in 1:length(ode_solvers)
    plot!(
    tolerances,
    median_benchmark[i, :],
    linewidth=2,
    # color=TolVibrantBlue,
    label=labels[i],
    xaxis=:log,
    marker=markers[i],
    xticks = tolerances,
    xflip=true
)
end

plot!(
    xlabel="Relative and Absolute Tolerance",
    ylabel="Median benchmark (seconds)"
)

savefig(string(saving_path_figure, "solver_comparison_all_diagonals_benchmark.pdf"))