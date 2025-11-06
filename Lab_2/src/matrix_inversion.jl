module matrix_inversion

using LinearAlgebra

export inverse

# Inverse function for scalars
function inverse(A::Number, mul = (X,Y)->X*Y, tri::Symbol = :none)
    return 1 / A
end

# Inverse function for vectors
function inverse(A::AbstractVector, mul = (X,Y)->X*Y, tri::Symbol = :none)
    n = length(A)
    if n == 1
        return A .^ -1
    else
        throw(ArgumentError("vector inversion not implemented for length > 1"))
    end
end

# Inverse function for matrices
function inverse(A::AbstractMatrix, mul = (X,Y)->X*Y, tri::Symbol = :none)
    n = size(A,1)
    if n != size(A,2)
        throw(ArgumentError("matrix must be square"))
    end
    if n == 1
        return A .^ -1
    end

    n1 = fld(n, 2)
    n2 = n1 + 1

    A11 = Matrix(view(A, 1:n1, 1:n1))
    if tri != :L
        A12 = Matrix(view(A, 1:n1, n2:n))
    end
    if tri != :U
        A21 = Matrix(view(A, n2:n, 1:n1))
    end
    A22 = Matrix(view(A, n2:n, n2:n))

    A11_inv = inverse(A11, mul)
    if tri == :none
        S22 = A22 .- mul(A21, mul(A11_inv, A12))
    else
        S22 = A22
    end
    S22_inv = inverse(S22, mul)

    if tri == :none
        T = mul(A12, mul(S22_inv, mul(A21, A11_inv)))
    end
    I_n1 = Matrix{eltype(A)}(I, n1, n1)

    if tri == :none
        B11 = mul(A11_inv, I_n1 + T)
        B12 = -mul(A11_inv, mul(A12, S22_inv))
        B21 = -mul(S22_inv, mul(A21, A11_inv))
        B22 = S22_inv
    elseif tri == :L
        B11 = A11_inv
        B12 = zeros(eltype(A), n1, n - n1)
        B21 = -mul(S22_inv, mul(A21, A11_inv))
        B22 = S22_inv
    elseif tri == :U
        B11 = A11_inv
        B12 = -mul(A11_inv, mul(A12, S22_inv))
        B21 = zeros(eltype(A), n - n1, n1)
        B22 = S22_inv
    end


    invA = similar(A)
    invA[1:n1, 1:n1] = B11
    invA[1:n1, n1+1:n] = B12
    invA[n1+1:n, 1:n1] = B21
    invA[n1+1:n, n1+1:n] = B22

    return invA
end
end # module matrix_inversion
