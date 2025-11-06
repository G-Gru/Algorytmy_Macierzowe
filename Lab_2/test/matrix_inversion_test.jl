using Test
using LinearAlgebra
include("../src/matrix_inversion.jl")
using .matrix_inversion
using Random

@testset "matrix_inversion_tests" begin
    # scalar
    @test inverse(2) == 1/2

    # 1x1 matrix
    A1 = [3.0]
    @test inverse(A1) == [1/3.0]

    # 2x2 matrix
    A2 = [4.0 7.0; 2.0 6.0]
    invA2 = inverse(A2)
    I2 = Matrix{Float64}(I, 2, 2)
    @test isapprox(invA2 * A2, I2; atol=1e-10)

    # 4x4 positive-definite matrix (to ensure invertible)
    Random.seed!(123)
    M = randn(4,4)
    A4 = M' * M + 1.0 * Matrix{Float64}(I,4,4)
    invA4 = inverse(A4)
    I4 = Matrix{Float64}(I, 4, 4)
    @test isapprox(A4 * invA4, I4; atol=1e-8)

    # custom multiplication argument (should behave like default)
    mul(X,Y) = X * Y
    @test isapprox(inverse(A2, mul), invA2; atol=1e-10)
end