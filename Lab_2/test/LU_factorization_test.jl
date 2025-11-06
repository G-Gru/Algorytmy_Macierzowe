include(joinpath(@__DIR__, "..", "src", "Lab_2.jl"))

using Test
using LinearAlgebra
using Random
using .Lab_2.LU_factorization

Random.seed!(123)

rand01_open(n...) = 1e-8 .+ (1.0 - 1e-8) .* rand(n...)

@testset "LU_factorization_basic" begin
    # 1x1 matrix
    A1 = [0.5]
    L1, U1 = LU_factor(A1)
    @test L1 .* U1 == A1
    @test L1[1,1] == 1.0
    @test U1[1,1] == 0.5

    # 2x2 matrix
    A2 = [0.6 0.2;
          0.1 0.7]
    L2, U2 = LU_factor(A2)
    @test isapprox(L2 * U2, A2; atol=1e-10)
    @test all(abs.(diag(L2) .- 1.0) .< 1e-12)
    @test maximum(abs.(triu(L2, 1))) < 1e-12
    @test maximum(abs.(tril(U2, -1))) < 1e-12

    # 3x3 example constructed from L*U
    L3_true = [1.0 0.0 0.0;
               0.2 1.0 0.0;
               0.3 0.1 1.0]
    U3_true = [0.5 0.2 0.1;
               0.0 0.6 0.2;
               0.0 0.0 0.4]
    A3 = L3_true * U3_true
    @assert all(A3 .< 1.0) && all(A3 .> 1e-8)
    L3, U3 = LU_factor(A3)
    @test isapprox(L3 * U3, A3; atol=1e-10)
    @test all(abs.(diag(L3) .- 1.0) .< 1e-12)
    @test maximum(abs.(triu(L3, 1))) < 1e-12
    @test maximum(abs.(tril(U3, -1))) < 1e-12

    # 4x4 example from L*U
    L4_true = [1.0 0.0 0.0 0.0;
               0.2 1.0 0.0 0.0;
               0.1 0.15 1.0 0.0;
               0.05 0.05 0.07 1.0]
    U4_true = [0.6 0.2 0.1 0.05;
               0.0 0.65 0.15 0.04;
               0.0 0.0 0.5 0.1;
               0.0 0.0 0.0 0.55]
    A4 = L4_true * U4_true
    @assert all(A4 .< 1.0) && all(A4 .> 1e-8)
    L4, U4 = LU_factor(A4)
    @test isapprox(L4 * U4, A4; atol=1e-10)
    @test all(abs.(diag(L4) .- 1.0) .< 1e-12)
    @test maximum(abs.(triu(L4, 1))) < 1e-12
    @test maximum(abs.(tril(U4, -1))) < 1e-12

    # 5x5 example from L*U
    L5_true = [1.0 0.0 0.0 0.0 0.0;
               0.12 1.0 0.0 0.0 0.0;
               0.08 0.07 1.0 0.0 0.0;
               0.03 0.04 0.05 1.0 0.0;
               0.02 0.01 0.03 0.06 1.0]
    U5_true = [0.7 0.12 0.08 0.05 0.03;
               0.0 0.65 0.1 0.04 0.02;
               0.0 0.0 0.6 0.09 0.01;
               0.0 0.0 0.0 0.55 0.05;
               0.0 0.0 0.0 0.0 0.5]
    A5 = L5_true * U5_true
    @assert all(A5 .< 1.0) && all(A5 .> 1e-8)
    L5, U5 = LU_factor(A5)
    @test isapprox(L5 * U5, A5; atol=1e-10)
    @test all(abs.(diag(L5) .- 1.0) .< 1e-12)
    @test maximum(abs.(triu(L5, 1))) < 1e-12
    @test maximum(abs.(tril(U5, -1))) < 1e-12
end