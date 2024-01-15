using Optim
using Distributions
using ForwardDiff

include("utils/utils.jl")

struct OptimizedResult
    estimates::Array{Float64,1}
    var_cov_matrix::Array{Float64,2}
    std::Array{Float64,1}
    intervals::Array
end

function confidence_intervals(func, initial_parameters)
    opt = optimize(func, initial_parameters)

    println("Minimum value is: ", opt.minimum)

    parameters = Optim.minimizer(opt)
    var_cov_matrix = variance_covariance_matrix(func, parameters)
    std = standard_errors(var_cov_matrix)

    intervals = []
    for i in eachindex(parameters)
        parameter_95 = confidence_intervals_0_95(parameters[i], std[i])
        append!(intervals, [round.(parameter_95;digits=4)])
    end 

    return OptimizedResult(round.(parameters;digits=4), round.(var_cov_matrix;digits=4), round.(std;digits=4), intervals)
end