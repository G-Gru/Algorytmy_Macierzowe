module LU_factorization

include(joinpath(@__DIR__, "..", "src", "matrix_inversion.jl"))

using LinearAlgebra
using .matrix_inversion

export LU_factor

function LU_factor(A::Number, mul = (X,Y)->X*Y)
    L = one(eltype(A))
    U = A
    return L, U
end

function LU_factor(A::AbstractVector, mul = (X,Y)->X*Y)
    n = length(A)
    if n == 1
        L = similar(A, eltype(A), 1)
        U = similar(A, eltype(A), 1)
        L[1] = one(eltype(A))
        U[1] = A[1]
        return L, U
    else
        throw(ArgumentError("LU factorization not implemented for vectors of length > 1"))
    end
    
end

function LU_factor(A::AbstractMatrix, mul = (X,Y)->X*Y)
    n = size(A,1)
    if n != size(A,2)
        throw(ArgumentError("matrix must be square"))
    end
    if n == 1
        L = similar(A, eltype(A), 1, 1)
        U = similar(A, eltype(A), 1, 1)
        L[1,1] = one(eltype(A))
        U[1,1] = A[1,1]
        return L, U
    end

    n1 = fld(n, 2)
    idx2 = n1 + 1

    A11 = Matrix(view(A, 1:n1, 1:n1))
    A12 = Matrix(view(A, 1:n1, idx2:n))
    A21 = Matrix(view(A, idx2:n, 1:n1))
    A22 = Matrix(view(A, idx2:n, idx2:n))

    L11, U11 = LU_factor(A11, mul)

    U11_inv = inverse(U11, mul, :U)
    L21 = mul(A21, U11_inv)

    L11_inv = inverse(L11, mul, :L)
    U12 = mul(L11_inv, A12)

    S = A22 .- mul(L21, U12)

    Ls, Us = LU_factor(S, mul)

    L = similar(A)
    U = similar(A)

    L[1:n1, 1:n1] = L11
    U[1:n1, 1:n1] = U11

    L[1:n1, idx2:n] .= zero(eltype(A))
    U[1:n1, idx2:n] = U12

    L[idx2:n, 1:n1] = L21
    U[idx2:n, 1:n1] .= zero(eltype(A))

    L[idx2:n, idx2:n] = Ls
    U[idx2:n, idx2:n] = Us

    return L, U
end

end # module LU_factorization