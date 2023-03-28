loading_path = string(@__DIR__, "/../../data/")
saving_path_figure = string(@__DIR__, "/../../output/figures/")
saving_path_data = string(@__DIR__, "/../../output/data/")

using Plots, Random, DelimitedFiles, DelayedKalmanFilter, StochasticDelayDiffEq, Statistics
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

prob = SDDEProblem(hes_model_drift, hes_model_noise, [30.,500.], h, tspan, p; saveat=10);
sol = solve(prob,RKMilCommute());

unobserved_data = Array(sol)[:,100:end];
measurement_std = 0.1*mean(unobserved_data[2,:])

protein = unobserved_data[2,:] +
    measurement_std*randn(length(unobserved_data[2,:]));

times = 0:10:730
protein_observations = hcat(times,protein)

system_state, distributions = kalman_filter(protein_observations, p, measurement_std^2);
means = [get_mean_at_time(i, system_state)[2] for i in times];
stds = [sqrt(get_variance_at_time(i, system_state)[2,2]+measurement_std^2) for i in times];

plot(
    times,
    unobserved_data[2,:],
    label="Unobserved",
    linewidth=2,
    color=TolVibrantBlue
)

scatter!(times, protein_observations[:,2], label="Observations")

plot!(
    times,
    means,
    ribbon=stds,
    fillalpha=.1,
    label="Kalman filter (with 1SD and 2SD)",
    linewidth=2,
    color=TolVibrantMagenta
)

plot!(
    times,
    means,
    ribbon=2*stds,
    fillalpha=.1,
    label=false,
    linewidth=2,
    color=TolVibrantMagenta
)

plot!(xlabel="Time (minutes)", ylabel="Protein molecule number")

savefig(string(saving_path_figure, "kalman_filter_plot.pdf"))
writedlm(string(saving_path_data, "fake_protein_observations.csv"), protein_observations)
writedlm(string(saving_path_data, "fake_measurement_std.csv"), measurement_std)
writedlm(string(saving_path_data, "true_protein.csv"), unobserved_data)
writedlm(string(saving_path_data, "true_params.csv"), p)

function make_protein(measurement_std)
    prob = SDDEProblem(hes_model_drift, hes_model_noise, [30.,500.], h, tspan, p; saveat=10);
    sol = solve(prob,RKMilCommute());

    unobserved_data = Array(sol)[:,100:end];
    # measurement_std = 0.1*mean(unobserved_data[2,:])

    protein = unobserved_data[2,:] + measurement_std*randn(length(unobserved_data[2,:]));
    times = 0:10:730

    return hcat(times,protein)
end

protein = make_protein(600.0)