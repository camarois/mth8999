using Optim
using Distributions
using ForwardDiff
using CSV
using DataFrames, NLSolversBase

include("utils/utils.jl")

ROUND_DIGITS = 4

struct OptimizedResult
    estimates::Array{Float64,1}
    var_cov_matrix::Array{Float64,2}
    std::Array{Float64,1}
    intervals::Array
end

function parameter_estimation(func, initial_parameters, to_round=false)
    opt = optimize(func, initial_parameters)

    println("Minimum value is: ", opt.minimum)

    parameters = Optim.minimizer(opt)
    var_cov_matrix = variance_covariance_matrix(func, parameters)
    std = standard_errors(var_cov_matrix)

    intervals = []
    for i in eachindex(parameters)
        parameter_95 = confidence_intervals_0_95(parameters[i], std[i])

        if to_round
            parameter_95 = round.(parameter_95; digits=ROUND_DIGITS)
        end

        append!(intervals, [parameter_95])
    end

    if to_round
        parameters = round.(parameters; digits=ROUND_DIGITS)
        var_cov_matrix = round.(var_cov_matrix; digits=ROUND_DIGITS)
        std = round.(std; digits=ROUND_DIGITS)
    end
    return OptimizedResult(parameters, var_cov_matrix, std, intervals)
end

function mean_failure_time(corrosion_level, parameters, var_cov_matrix, to_round=false)
    lambda = parameters[1]^-1 * corrosion_level^(-parameters[2])
    delta = [(-parameters[1]^(-2)) * corrosion_level^(-parameters[2]); (-parameters[1]^(-1)) * corrosion_level^(-parameters[2]) * log(corrosion_level)]
    deltaT = transpose(delta)
    var = deltaT * var_cov_matrix * delta

    mean_failure_time = confidence_intervals_0_95(lambda, sqrt(var))
    if to_round
        mean_failure_time = round.(mean_failure_time; digits=ROUND_DIGITS)
    end

    return mean_failure_time
end

function deviance(func, parameters_m1, func_m0, parameters_m0)
    opt_1 = optimize(func, parameters_m1).minimum
    opt_0 = optimize(func_m0, parameters_m0).minimum

    return round(2 * (opt_0 - opt_1); digits=2)
end

# Script
engine_df = CSV.read(joinpath(@__DIR__, "../data/engine.csv"), DataFrame)

n = length(engine_df[!, :CorrosionLevel])
function log_likelihood(betas, W, T)
    llike = n * log(betas[1]) + betas[2] * sum(log.(W)) - betas[1] * sum(W .^ betas[2] .* T)
    llike = -llike
end

function log_likelihood_m0(betas, T)
    llike = n * log(betas[1]) - betas[1] * sum(T)
    llike = -llike
end

W = reshape(engine_df[!, :CorrosionLevel], :, 1)
T = reshape(engine_df[!, :Lifetime], :, 1)
corrosion_level = 3

nvar = 2
func = TwiceDifferentiable(vars -> log_likelihood(vars[1:nvar], W, T),
    ones(nvar); autodiff=:forward)
func_m0 = TwiceDifferentiable(vars -> log_likelihood_m0(vars[1:1], T),
    ones(1); autodiff=:forward)

result = parameter_estimation(func, [1.0, 1.0], true)
println("Parameters estimations : ", result.estimates)
println("variance-covariance matrix: ", result.var_cov_matrix)
println("Standard-deviation : ", result.std)
println("Confidence intervals : ", result.intervals)

result = parameter_estimation(func, [1.0, 1.0])
println("Mean failure time : ", mean_failure_time(corrosion_level, result.estimates, result.var_cov_matrix, true))
println("Deviance : ", deviance(func, [1.0, 1.0], func_m0, [1.0]))