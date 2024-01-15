@testset "utils.jl" begin
    @testset "standard_errors()" begin
        @testset "valid_values()" begin

            x = [0.5 1 1 0.5]
            @test standard_errors(x) == [sqrt(0.5)]
        end
    end

    @testset "confidence_intervals_0_95()" begin
        @testset "valid_values()" begin

            x = 1.0
            std = 0.1
            @test confidence_intervals_0_95(x, std) == [x - 1.96*std, x + 1.96*std]
        end

        @testset "nothing_values()" begin

            @test_throws MethodError confidence_intervals_0_95(nothing, nothing)
        end
    end
end