using LinearAlgebra: diag

function variance_covariance_matrix(func, parameters)
    hessian = hessian!(func, parameters)
    return inv(hessian)
end

function standard_errors(var_cov_matrix)
    temp = diag(var_cov_matrix)
    return sqrt.(temp)
end

function confidence_intervals_0_95(parameter, standard_deviation)
    return [parameter - 1.96 * standard_deviation, parameter + 1.96 * standard_deviation]
end