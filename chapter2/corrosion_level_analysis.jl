using CSV
using DataFrames, Optim, NLSolversBase
using Distributions
using ForwardDiff
using LinearAlgebra: diag

engine_df = CSV.read(joinpath(@__DIR__, "data/engine.csv"), DataFrame)

n = length(engine_df[!,:CorrosionLevel])
function log_likelihood(betas, W, T)
    llike = n*log(betas[1]) + betas[2] * sum(log.(W)) - betas[1] * sum(W.^betas[2] .* T)
    llike = -llike
end

W = reshape(engine_df[!,:CorrosionLevel], :, 1)
T = reshape(engine_df[!,:Lifetime], :, 1)

nvar=2
func = TwiceDifferentiable(vars -> log_likelihood(vars[1:nvar], W, T),
                           ones(nvar); autodiff=:forward);


opt = optimize(func, [1.0, 1.0])
print(opt)
print(opt.minimum)
print(opt.minimizer)

parameters = Optim.minimizer(opt)
numerical_hessian = hessian!(func,parameters)
var_cov_matrix = inv(numerical_hessian)

print(var_cov_matrix)

temp = diag(var_cov_matrix)
std = sqrt.(temp)

print(std)

a_95 = [parameters[1] - 1.96 * std[1], parameters[1] + 1.96 * std[1]]
b_95 = [parameters[2] - 1.96 * std[1], parameters[2] + 1.96 * std[2]]