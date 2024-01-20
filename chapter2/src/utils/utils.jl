using LinearAlgebra: diag

"""
    variance_covariance_matrix(func::TwiceDifferentiable, parameters::AbstractVector{<:Real})
    Returns the variance covariance matric of a twice differentiable log-likelihood function.
"""
function variance_covariance_matrix(func::TwiceDifferentiable, parameters::AbstractVector{<:Real})
    hessian = hessian!(func, parameters)
    return inv(hessian)
end

"""
    standard_errors(var_cov_matrix::AbstractMatrix{<:Real})
    Returns the standard errors based on a model's variance covariance matrix.
"""
function standard_errors(var_cov_matrix::AbstractMatrix{<:Real})
    temp = diag(var_cov_matrix)
    return sqrt.(temp)
end

"""
    confidence_intervals_0_95(parameter::Float64, standard_deviation::Float64)
    Returns a 95% confidence interval for an estimated parameter.
"""
function confidence_intervals_0_95(parameter::Float64, standard_deviation::Float64)
    return [parameter - 1.96 * standard_deviation, parameter + 1.96 * standard_deviation]
end