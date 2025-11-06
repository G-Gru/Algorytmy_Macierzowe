include(joinpath(@__DIR__, "..", "src", "Lab_2.jl"))
include(joinpath(@__DIR__, "..", "lib", "strassen_padded.jl"))


using .Lab_2.matrix_inversion
using .strassen_padded
using Test
using LinearAlgebra
using Random

rand01_open() = 1e-8 .+ (1.0 - 1e-8) * rand()
default_mul(A, B, add_count::Ref{Int}=Ref(0), mul_count::Ref{Int}=Ref(0)) = (A * B, add_count[], mul_count[])


@testset "matrix_inversion_tests" begin
    # scalar
    s = 0.5
    invs, adds, muls = inverse(s)
    @test invs == 1 / s

    # 1x1 matrix
    A1 = [0.25]
    invA1, add1, mul1 = inverse(A1)
    @test invA1 == [1 / A1[1]]

    # 2x2 matrix    
    A2 = [0.4 0.7; 0.2 0.6]
    invA2, add2, mul2 = inverse(A2)
    I2 = Matrix{Float64}(I, 2, 2)
    @test isapprox(invA2 * A2, I2; atol=1e-10)
    @test add2 + mul2 >= 0

    # 3x3 matrix    
    A3 = [0.1 0.2 0.3;
          0.01 0.11 0.04;
          0.5 0.6 0.01]
    invA3, add3, mul3 = inverse(A3)
    I3 = Matrix{Float64}(I, 3, 3)
    @test isapprox(A3 * invA3, I3; atol=1e-10)
    @test add3 + mul3 >= 0

    # 4x4 positive-definite-like
    A4 = zeros(4,4)
    for i in 1:4
        A4[i,i] = 0.8
        if i < 4
            A4[i,i+1] = 0.05
            A4[i+1,i] = 0.05
        end
    end
    invA4, add4, mul4 = inverse(A4)
    I4 = Matrix{Float64}(I, 4, 4)
    @test isapprox(A4 * invA4, I4; atol=1e-8)
    @test add4 + mul4 >= 0

    # 5x5 tridiagonal    
    A5 = zeros(5,5)
    for i in 1:5
        A5[i,i] = 0.85
        if i < 5
            A5[i,i+1] = 0.05
            A5[i+1,i] = 0.05
        end
    end
    invA5, add5, mul5 = inverse(A5)
    I5 = Matrix{Float64}(I, 5, 5)
    @test isapprox(A5 * invA5, I5; atol=1e-10)
    @test add5 + mul5 >= 0

    # 21x21 tridiagonal-like    
    n21 = 21
    A21 = zeros(n21, n21)
    for i in 1:n21
        A21[i,i] = 0.9
        if i < n21
            A21[i,i+1] = 0.05
            A21[i+1,i] = 0.05
        end
    end
    invA21, add21, mul21 = inverse(A21)
    I21 = Matrix{Float64}(I, n21, n21)
    @test isapprox(A21 * invA21, I21; atol=1e-6)
    @test add21 + mul21 >= 0

    # lower-triangular inversion (tri = :L)
    nL = 5
    L = zeros(nL,nL)
    for i in 1:nL
        L[i,i] = 0.8                      # diagonal in (1e-8,1.0)
        for j in 1:i-1
            L[i,j] = 0.04                 # lower part nonzero
        end
    end
    invL, addL, mulL = inverse(L, default_mul, 'L')
    I_L = Matrix{Float64}(I, nL, nL)
    @test isapprox(L * invL, I_L; atol=1e-10)
    # check inverse is lower-triangular (upper strictly triangular entries ~ 0)
    @test maximum(abs.(triu(invL, 1))) < 1e-10
    @test addL + mulL >= 0

    # upper-triangular inversion (tri = :U)
    nU = 6
    U = zeros(nU,nU)
    for i in 1:nU
        U[i,i] = 0.85
        for j in i+1:nU
            U[i,j] = 0.03
        end
    end
    invU, addU, mulU = inverse(U, default_mul, 'U')
    I_U = Matrix{Float64}(I, nU, nU)
    @test isapprox(U * invU, I_U; atol=1e-10)
    # check inverse is upper-triangular (lower strictly triangular entries ~ 0)
    @test maximum(abs.(tril(invU, -1))) < 1e-10
    @test addU + mulU >= 0


    # custom multiplication argument (must accept count refs and return same shape tuple)
    mul(X,Y, add_count::Ref{Int}=Ref(0), mul_count::Ref{Int}=Ref(0)) = (X * Y, add_count[], mul_count[])
    inv_custom, addc, mulc = inverse(A2, mul)
    @test isapprox(inv_custom, invA2; atol=1e-10)
    @test addc + mulc >= 0
end


@testset "matrix_inversion_tests_strassen" begin
    # scalar
    s = 0.5
    invs, adds, muls = inverse(s, strassen_pad)
    @test invs == 1 / s

    # 1x1 matrix
    A1 = [0.25]
    invA1, add1, mul1 = inverse(A1, strassen_pad)
    @test invA1 == [1 / A1[1]]

    # 2x2 matrix    
    A2 = [0.4 0.7; 0.2 0.6]
    invA2, add2, mul2 = inverse(A2, strassen_pad)
    I2 = Matrix{Float64}(I, 2, 2)
    @test isapprox(invA2 * A2, I2; atol=1e-10)
    @test add2 + mul2 > 0

    # 3x3 matrix    
    A3 = [0.1 0.2 0.3;
          0.01 0.11 0.04;
          0.5 0.6 0.01]
    invA3, add3, mul3 = inverse(A3, strassen_pad)
    I3 = Matrix{Float64}(I, 3, 3)
    @test isapprox(A3 * invA3, I3; atol=1e-10)
    @test add3 + mul3 > 0

    # 4x4 positive-definite-like
    A4 = zeros(4,4)
    for i in 1:4
        A4[i,i] = 0.8
        if i < 4
            A4[i,i+1] = 0.05
            A4[i+1,i] = 0.05
        end
    end
    invA4, add4, mul4 = inverse(A4, strassen_pad)
    I4 = Matrix{Float64}(I, 4, 4)
    @test isapprox(A4 * invA4, I4; atol=1e-8)
    @test add4 + mul4 > 0

    # 5x5 tridiagonal    
    A5 = zeros(5,5)
    for i in 1:5
        A5[i,i] = 0.85
        if i < 5
            A5[i,i+1] = 0.05
            A5[i+1,i] = 0.05
        end
    end
    invA5, add5, mul5 = inverse(A5, strassen_pad)
    I5 = Matrix{Float64}(I, 5, 5)
    @test isapprox(A5 * invA5, I5; atol=1e-10)
    @test add5 + mul5 > 0

    # 21x21 tridiagonal-like    
    n21 = 21
    A21 = zeros(n21, n21)
    for i in 1:n21
        A21[i,i] = 0.9
        if i < n21
            A21[i,i+1] = 0.05
            A21[i+1,i] = 0.05
        end
    end
    invA21, add21, mul21 = inverse(A21, strassen_pad)
    I21 = Matrix{Float64}(I, n21, n21)
    @test isapprox(A21 * invA21, I21; atol=1e-6)
    @test add21 + mul21 > 0

    # lower-triangular inversion (tri = 'L')
    nL = 5
    L = zeros(nL,nL)
    for i in 1:nL
        L[i,i] = 0.8                      # diagonal in (1e-8,1.0)
        for j in 1:i-1
            L[i,j] = 0.04                 # lower part nonzero
        end
    end
    invL, addL, mulL = inverse(L, strassen_pad, 'L')
    I_L = Matrix{Float64}(I, nL, nL)
    @test isapprox(L * invL, I_L; atol=1e-10)
    # check inverse is lower-triangular (upper strictly triangular entries ~ 0)
    @test maximum(abs.(triu(invL, 1))) < 1e-10
    @test addL + mulL > 0

    # upper-triangular inversion (tri = 'U')
    nU = 6
    U = zeros(nU,nU)
    for i in 1:nU
        U[i,i] = 0.85
        for j in i+1:nU
            U[i,j] = 0.03
        end
    end
    invU, addU, mulU = inverse(U, strassen_pad, 'U')
    I_U = Matrix{Float64}(I, nU, nU)
    @test isapprox(U * invU, I_U; atol=1e-10)
    # check inverse is upper-triangular (lower strictly triangular entries ~ 0)
    @test maximum(abs.(tril(invU, -1))) < 1e-10
    @test addU + mulU > 0

    # custom multiplication argument (use strassen implementation as custom mul)
    mul(X,Y, add_count::Ref{Int}=Ref(0), mul_count::Ref{Int}=Ref(0)) = strassen_pad(X, Y, add_count, mul_count)
    inv_custom, addc, mulc = inverse(A2, mul)
    @test isapprox(inv_custom, invA2; atol=1e-10)
    @test addc + mulc > 0
end
