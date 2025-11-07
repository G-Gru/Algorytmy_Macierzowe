using Test
using Random

include(joinpath(@__DIR__, "..", "lib", "strassen_split.jl"))
using .strassen_split

rand01_open(r::Integer, c::Integer) = 1e-8 .+ (1.0 - 1e-8) * rand(r, c)
rand01_open() = 1e-8 .+ (1.0 - 1e-8) * rand()

# Beware: Tests will fail, strassen_spl is not finished yet!

@testset "Strassen split" begin

    @testset "1x1" begin
        A = [rand01_open()]
        B = [rand01_open()]
        C, adds, muls = strassen_spl(A, B)
        @test C == A .* B
        @test muls == 1
        @test adds == 0
    end

    @testset "2x2 exact" begin
        A = [0.2 0.4; 0.3 0.7]
        B = [0.5 0.6; 0.7 0.8]
        C, adds, muls = strassen_spl(A, B)
        @test isapprox(C, A * B, atol=1e-12, rtol=1e-12)
        @test isa(adds, Integer) && isa(muls, Integer)
        @test adds >= 0 && muls >= 1
    end

    @testset "4x4 random" begin
        Random.seed!(1234)
        A = rand01_open(4,4)
        B = rand01_open(4,4)
        C, adds, muls = strassen_spl(A, B)
        @test isapprox(C, A * B; atol=1e-12, rtol=1e-12)
        @test adds >= 0 && muls >= 1
    end

    @testset "3x3 padded" begin
        Random.seed!(5678)
        A = rand01_open(3,3)
        B = rand01_open(3,3)
        C, adds, muls = strassen_spl(A, B)
        @test isapprox(C[1:3, 1:3], A * B; atol=1e-12, rtol=1e-12)
        @test adds >= 0 && muls >= 1
    end

    @testset "5x5 padded" begin
        Random.seed!(91011)
        A = rand01_open(5,5)
        B = rand01_open(5,5)
        C, adds, muls = strassen_spl(A, B)
        @test isapprox(C[1:5, 1:5], A * B; atol=1e-12, rtol=1e-12)
        @test adds >= 0 && muls >= 1
    end

    @testset "21x21 padded" begin
        Random.seed!(121314)
        A = rand01_open(21,21)
        B = rand01_open(21,21)
        C, adds, muls = strassen_spl(A, B)
        @test isapprox(C[1:21, 1:21], A * B; atol=1e-12, rtol=1e-12)
        @test adds >= 0 && muls >= 1
    end

        @testset "rectangular: 5x4 * 4x5" begin
        Random.seed!(2025)
        A = rand01_open(5,4)
        B = rand01_open(4,5)
        C, adds, muls = strassen_spl(A, B)
        @test size(C) == (5,5)
        @test isapprox(C, A * B; atol=1e-12, rtol=1e-12)
        @test adds >= 0 && muls >= 1
    end

    @testset "rectangular: 4x5 * 5x4" begin
        Random.seed!(2026)
        A = rand01_open(4,5)
        B = rand01_open(5,4)
        C, adds, muls = strassen_spl(A, B)
        @test size(C) == (4,4)
        @test isapprox(C, A * B; atol=1e-12, rtol=1e-12)
        @test adds >= 0 && muls >= 1
    end

    @testset "rectangular: 6x7 * 7x6" begin
        Random.seed!(2027)
        A = rand01_open(6,7)
        B = rand01_open(7,6)
        C, adds, muls = strassen_spl(A, B)
        @test size(C) == (6,6)
        @test isapprox(C, A * B; atol=1e-12, rtol=1e-12)
        @test adds >= 0 && muls >= 1
    end

    @testset "rectangular: 3x5 * 5x2" begin
        Random.seed!(2028)
        A = rand01_open(3,5)
        B = rand01_open(5,2)
        C, adds, muls = strassen_spl(A, B)
        @test size(C) == (3,2)
        @test isapprox(C, A * B; atol=1e-12, rtol=1e-12)
        @test adds >= 0 && muls >= 1
    end
end
