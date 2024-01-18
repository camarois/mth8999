using Optim
using Distributions
using ForwardDiff

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

function deviance(func, parameters_m1, func_0, parameters_m0)
    opt_1 = optimize(func, parameters_m1).minimum
    opt_0 = optimize(func_0, parameters_m0).minimum

    return round(2 * (opt_0 - opt_1); digits=2)
end