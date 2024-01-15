using CSV
using DataFrames, NLSolversBase

@testset "corrosion_levels.jl" begin
   # Inspired from optim documentation https://julianlsolvers.github.io/Optim.jl/stable/#examples/generated/maxlikenlm/
    engine_df = CSV.read(joinpath(@__DIR__, "../data/engine.csv"), DataFrame)

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


    @testset "confidence_intervals()" begin
        @testset "valid_values()" begin

            result = confidence_intervals(func, [1.0, 1.0])

            @test result.estimates == [1.133, 0.4792]
            @test result.var_cov_matrix == [0.0468 -0.0144; -0.0144 0.0311]
            @test result.std == [0.2164, 0.1762]
            @test result.intervals == [[0.7089, 1.5572], [0.1338, 0.8246]]
        end
    end
end