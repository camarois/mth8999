include(joinpath(dirname(@__DIR__), "src/utils/utils.jl"))
include(joinpath(dirname(@__DIR__), "src/corrosion_levels.jl"))
using Test

@testset "corrosion_levels.jl" begin
    include("test_corrosion_levels.jl")
    include("utils/test_utils.jl")
end