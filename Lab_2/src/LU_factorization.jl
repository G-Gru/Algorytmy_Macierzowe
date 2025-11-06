module LU_factorization

include(joinpath(@__DIR__, "..", "src", "matrix_inversion.jl"))

using LinearAlgebra
using .matrix_inversion

export LU_factor

function LU_factor(A::Number, mul = matrix_inversion.default_mul, add_count::Ref{Int}=Ref(0), mul_count::Ref{Int}=Ref(0))
    L = one(eltype(A))
    U = A
    return L, U, add_count[], mul_count[]
end

function LU_factor(A::AbstractVector, mul = matrix_inversion.default_mul, add_count::Ref{Int}=Ref(0), mul_count::Ref{Int}=Ref(0))
    n = length(A)
    if n == 1
        L = similar(A, eltype(A), 1)
        U = similar(A, eltype(A), 1)
        L[1] = one(eltype(A))
        U[1] = A[1]
        return L, U, add_count[], mul_count[]
    else
        throw(ArgumentError("LU factorization not implemented for vectors of length > 1"))
    end
    
end

function LU_factor(A::AbstractMatrix, mul = matrix_inversion.default_mul, add_count::Ref{Int}=Ref(0), mul_count::Ref{Int}=Ref(0))
    n = size(A,1)
    if n != size(A,2)
        throw(ArgumentError("matrix must be square"))
    end
    if n == 1
        L = similar(A, eltype(A), 1, 1)
        U = similar(A, eltype(A), 1, 1)
        L[1,1] = one(eltype(A))
        U[1,1] = A[1,1]
        return L, U, add_count[], mul_count[]
    end

    n1 = fld(n, 2)
    idx2 = n1 + 1

    A11 = Matrix(view(A, 1:n1, 1:n1))
    A12 = Matrix(view(A, 1:n1, idx2:n))
    A21 = Matrix(view(A, idx2:n, 1:n1))
    A22 = Matrix(view(A, idx2:n, idx2:n))

    L11, U11, _, _ = LU_factor(A11, mul, add_count, mul_count)

    U11_inv, _, _ = inverse(U11, mul, 'U', add_count, mul_count)
    L21, _, _ = mul(A21, U11_inv, add_count, mul_count)

    L11_inv, _, _ = inverse(L11, mul, 'L', add_count, mul_count)
    U12, _, _ = mul(L11_inv, A12, add_count, mul_count)

    tmp, _, _ = mul(L21, U12, add_count, mul_count)
    S = matrix_inversion._matsub!(A22, tmp, add_count)

    Ls, Us, _, _ = LU_factor(S, mul, add_count, mul_count)

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

    return L, U, add_count[], mul_count[]
end

end # module LU_factorization