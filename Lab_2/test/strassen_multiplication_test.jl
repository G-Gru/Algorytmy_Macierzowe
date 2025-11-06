using Test
using Random

include(joinpath(@__DIR__, "..", "lib", "strassen_multiplication.jl"))
using .strassen_multiplication

@testset "Strassen multiplication" begin

    @testset "1x1" begin
        A = [2.0]
        B = [3.0]
        C, adds, muls = strassen(A, B)
        @test C == A .* B
        @test muls == 1
        @test adds == 0
    end

    @testset "2x2 exact" begin
        A = [1.0 2.0; 3.0 4.0]
        B = [5.0 6.0; 7.0 8.0]
        C, adds, muls = strassen(A, B)
        @test C == A * B
        @test isa(adds, Integer) && isa(muls, Integer)
        @test adds >= 0 && muls >= 1
    end

    @testset "4x4 random" begin
        Random.seed!(1234)
        A = rand(4,4)
        B = rand(4,4)
        C, adds, muls = strassen(A, B)
        @test isapprox(C, A * B; atol=1e-12, rtol=1e-12)
        @test adds >= 0 && muls >= 1
    end

end