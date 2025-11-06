module gaussian_elimination

export solve

using LinearAlgebra

using ..LU_factorization
using ..matrix_inversion

function triangulate(A::AbstractMatrix, b::AbstractVector, mul=*)
    n = size(A, 1)
    if !(n == size(A, 2) == size(b, 1))
        throw(ArgumentError("A must be of size n×n and b must be of size n"))
    end
    n == 1 && return A, b

    nhalf = n ÷ 2

    A11 = @view A[1:nhalf, 1:nhalf]
    A12 = @view A[1:nhalf, nhalf+1:end]
    A21 = @view A[nhalf+1:end, 1:nhalf]
    A22 = @view A[nhalf+1:end, nhalf+1:end]
    b1  = @view b[1:nhalf]
    b2  = @view b[nhalf+1:end]

    L11, U11 = LU_factor(A11)
    L11inv, U11inv = inverse(L11)[1], inverse(U11)[1]

    S = A22 - reduce(mul, [A21, U11inv, L11inv, A12])

    LS, US = LU_factor(S)
    LSinv, _, _ = inverse(LS)

    C11 = U11
    C12 = mul(L11, A12)
    C21 = zeros(size(A21))
    C22 = US

    RHS1 = mul(L11inv, b1)
    RHS2 = mul(LSinv, b2) - reduce(mul, [LSinv, A21, U11inv, L11inv, b1])

    Ares = [C11 C12
            C21 C22]
    bres = [RHS1; RHS2]

    Ares, bres
end

end # module gaussian_elimination
