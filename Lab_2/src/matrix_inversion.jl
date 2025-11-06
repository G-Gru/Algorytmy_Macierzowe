module matrix_inversion

using LinearAlgebra

export inverse

# default multiplication wrapper
default_mul(A, B, add_count::Ref{Int}=Ref(0), mul_count::Ref{Int}=Ref(0)) = (A * B, add_count[], mul_count[])

# helpers for counting operations 
function _matadd!(X, Y, add_count::Ref{Int})
    add_count[] += prod(size(X))
    return X .+ Y
end
function _matsub!(X, Y, add_count::Ref{Int})
    add_count[] += prod(size(X))
    return X .- Y
end

# Inverse function for scalars
function inverse(A::Number, mul = default_mul, tri::Char = 'N', add_count::Ref{Int}=Ref(0), mul_count::Ref{Int}=Ref(0))
    return 1 / A, add_count[], mul_count[]
end

# Inverse function for vectors 
function inverse(A::AbstractVector, mul = default_mul, tri::Char = 'N', add_count::Ref{Int}=Ref(0), mul_count::Ref{Int}=Ref(0))
    n = length(A)
    if n == 1
        return (similar(A, eltype(A), 1) .= 1 / A[1]), add_count[], mul_count[]
    else
        throw(ArgumentError("vector inversion not implemented for length > 1"))
    end
end

# Inverse function for matrices
function inverse(A::AbstractMatrix, mul = default_mul, tri::Char = 'N', add_count::Ref{Int}=Ref(0), mul_count::Ref{Int}=Ref(0))
    n = size(A,1)
    if n != size(A,2)
        throw(ArgumentError("matrix must be square"))
    end
    if n == 1
        return (similar(A, eltype(A), 1, 1) .= 1 / A[1,1]), add_count[], mul_count[]
    end

    tri_i = tri == 'N' ? 0 : tri == 'L' ? 1 : tri == 'U' ? 2 : throw(ArgumentError("tri must be 'N', 'L' or 'U'"))

    n1 = fld(n, 2)
    n2 = n1 + 1

    A11 = Matrix(view(A, 1:n1, 1:n1))
    if tri_i != 1
        A12 = Matrix(view(A, 1:n1, n2:n))
    end
    if tri_i != 2
        A21 = Matrix(view(A, n2:n, 1:n1))
    end
    A22 = Matrix(view(A, n2:n, n2:n))

    A11_inv, _, _ = inverse(A11, mul, tri, add_count, mul_count)

    if tri_i == 0
        tmp1, _, _ = mul(A11_inv, A12, add_count, mul_count)
        tmp2, _, _ = mul(A21, tmp1, add_count, mul_count)
        S22 = _matsub!(A22, tmp2, add_count)
    else
        S22 = A22
    end

    S22_inv, _, _ = inverse(S22, mul, tri, add_count, mul_count)

    if tri_i == 0
        tmp3, _, _ = mul(A21, A11_inv, add_count, mul_count)
        tmp4, _, _ = mul(S22_inv, tmp3, add_count, mul_count)
        T, _, _ = mul(A12, tmp4, add_count, mul_count)
    end

    I_n1 = Matrix{eltype(A)}(I, n1, n1)

    if tri_i == 0
        B11_tmp = _matadd!(I_n1, T, add_count)
        B11, _, _ = mul(A11_inv, B11_tmp, add_count, mul_count)

        tmp5, _, _ = mul(A12, S22_inv, add_count, mul_count)
        B12, _, _ = mul(A11_inv, tmp5, add_count, mul_count)
        B12 .= -B12

        tmp6, _, _ = mul(A21, A11_inv, add_count, mul_count)
        B21, _, _ = mul(S22_inv, tmp6, add_count, mul_count)
        B21 .= -B21

        B22 = S22_inv
    elseif tri_i == 1
        B11 = A11_inv
        B12 = zeros(eltype(A), n1, n - n1)
        tmp6, _, _ = mul(A21, A11_inv, add_count, mul_count)
        B21, _, _ = mul(S22_inv, tmp6, add_count, mul_count)
        B21 .= -B21
        B22 = S22_inv
    elseif tri_i == 2
        B11 = A11_inv
        tmp5, _, _ = mul(A12, S22_inv, add_count, mul_count)
        B12, _, _ = mul(A11_inv, tmp5, add_count, mul_count)
        B12 .= -B12
        B21 = zeros(eltype(A), n - n1, n1)
        B22 = S22_inv
    end

    invA = similar(A)
    invA[1:n1, 1:n1] = B11
    invA[1:n1, n1+1:n] = B12
    invA[n1+1:n, 1:n1] = B21
    invA[n1+1:n, n1+1:n] = B22

    return invA, add_count[], mul_count[]
end

end # module matrix_inversion