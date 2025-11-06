module strassen_multiplication

export strassen

function strassen(A::Number, B::Number, add_count::Ref{Int}=Ref(0), mul_count::Ref{Int}=Ref(0))
    mul_count[] += 1
    return A * B, add_count[], mul_count[]
    
end

function strassen(A::AbstractVector, B::AbstractVector, add_count::Ref{Int}=Ref(0), mul_count::Ref{Int}=Ref(0))
    n = length(A)
    @assert n == length(B) "vectors must be of same length"

    if n == 1
        mul_count[] += 1
        return A .* B, add_count[], mul_count[]
    else
        throw(ArgumentError("Strassen multiplication not implemented for vectors of length > 1"))
    end
    
end

function strassen(A::AbstractMatrix, B::AbstractMatrix, add_count::Ref{Int}=Ref(0), mul_count::Ref{Int}=Ref(0))
    n = size(A, 1)
    @assert n == size(A,2) == size(B,1) == size(B,2) "square matrices of same size required"

    # helpers that update addition counters (counts scalar add/sub per element)
    matadd(X, Y) = (add_count[] += prod(size(X)); X .+ Y)
    matsub(X, Y) = (add_count[] += prod(size(X)); X .- Y)

    if n == 1
        mul_count[] += 1
        return A * B, add_count[], mul_count[]
    end

    mid = n ÷ 2
    A11 = @view A[1:mid, 1:mid];  A12 = @view A[1:mid, mid+1:n]
    A21 = @view A[mid+1:n, 1:mid]; A22 = @view A[mid+1:n, mid+1:n]

    B11 = @view B[1:mid, 1:mid];  B12 = @view B[1:mid, mid+1:n]
    B21 = @view B[mid+1:n, 1:mid]; B22 = @view B[mid+1:n, mid+1:n]

    P1, _, _ = strassen(matadd(Array(A11), Array(A22)), matadd(Array(B11), Array(B22)), add_count, mul_count)
    P2, _, _ = strassen(matadd(Array(A21), Array(A22)), Array(B11), add_count, mul_count)
    P3, _, _ = strassen(Array(A11), matsub(Array(B12), Array(B22)), add_count, mul_count)
    P4, _, _ = strassen(Array(A22), matsub(Array(B21), Array(B11)), add_count, mul_count)
    P5, _, _ = strassen(matadd(Array(A11), Array(A12)), Array(B22), add_count, mul_count)
    P6, _, _ = strassen(matsub(Array(A21), Array(A11)), matadd(Array(B11), Array(B12)), add_count, mul_count)
    P7, _, _ = strassen(matsub(Array(A12), Array(A22)), matadd(Array(B21), Array(B22)), add_count, mul_count)

    C11 = matadd(matadd(P1, P4), matsub(P7, P5))
    C12 = matadd(P3, P5)
    C21 = matadd(P2, P4)
    C22 = matadd(matadd(P1, P3), matsub(P6, P2))

    C = vcat(hcat(C11, C12), hcat(C21, C22))
    return C, add_count[], mul_count[]
end
end # module