module strassen_padded

export strassen_pad

# helper to check if integer is power of 2
ispow2(n::Integer) = n > 0 && (n & (n - 1)) == 0

# Strassen multiplication for numbers
function strassen_pad(A::Number, B::Number, add_count::Ref{Int}=Ref(0), mul_count::Ref{Int}=Ref(0), orig_rows::Int=typemax(Int), orig_cols::Int=typemax(Int))
    mul_count[] += 1
    return A * B, add_count[], mul_count[]
    
end

# Strassen multiplication for vectors
function strassen_pad(A::AbstractVector, B::AbstractVector, add_count::Ref{Int}=Ref(0), mul_count::Ref{Int}=Ref(0), orig_rows::Int=typemax(Int), orig_cols::Int=typemax(Int))
    n = length(A)
    @assert n == length(B) "vectors must be of same length"

    if n == 1
        mul_count[] += 1
        return A .* B, add_count[], mul_count[]
    else
        throw(ArgumentError("Strassen multiplication not implemented for vectors of length > 1"))
    end
    
end

# Strassen multiplication for matrices
function strassen_pad(A::AbstractMatrix, B::AbstractMatrix, add_count::Ref{Int}=Ref(0), mul_count::Ref{Int}=Ref(0), orig_rows::Int=typemax(Int), orig_cols::Int=typemax(Int))
    rA, cA = size(A)
    rB, cB = size(B)
    @assert cA == rB "matrices must have compatible dimensions"

    if orig_rows == typemax(Int) && orig_cols == typemax(Int)
        orig_rows, orig_cols = rA, cB
    end

    n = max(rA, cA, cB)
    if !ispow2(n)
        m = 1
        while m < n
            m <<= 1
        end
    else
        m = n
    end

    if m != rA || m != cA || m != rB || m != cB
        Ap = zeros(eltype(A), m, m)
        Bp = zeros(eltype(B), m, m)
        Ap[1:rA, 1:cA] .= A
        Bp[1:rB, 1:cB] .= B
        A = Ap
        B = Bp
        n = m
    else
        n = m
    end

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

    P1, _, _ = strassen_pad(matadd(Array(A11), Array(A22)), matadd(Array(B11), Array(B22)), add_count, mul_count, orig_rows, orig_cols)
    P2, _, _ = strassen_pad(matadd(Array(A21), Array(A22)), Array(B11), add_count, mul_count, orig_rows, orig_cols)
    P3, _, _ = strassen_pad(Array(A11), matsub(Array(B12), Array(B22)), add_count, mul_count, orig_rows, orig_cols)
    P4, _, _ = strassen_pad(Array(A22), matsub(Array(B21), Array(B11)), add_count, mul_count, orig_rows, orig_cols)
    P5, _, _ = strassen_pad(matadd(Array(A11), Array(A12)), Array(B22), add_count, mul_count, orig_rows, orig_cols)
    P6, _, _ = strassen_pad(matsub(Array(A21), Array(A11)), matadd(Array(B11), Array(B12)), add_count, mul_count, orig_rows, orig_cols)
    P7, _, _ = strassen_pad(matsub(Array(A12), Array(A22)), matadd(Array(B21), Array(B22)), add_count, mul_count, orig_rows, orig_cols)

    C11 = matadd(matadd(P1, P4), matsub(P7, P5))
    C12 = matadd(P3, P5)
    C21 = matadd(P2, P4)
    C22 = matadd(matadd(P1, P3), matsub(P6, P2))

    C = vcat(hcat(C11, C12), hcat(C21, C22))

    # trim to original rows/cols
    if size(C, 1) > orig_rows || size(C, 2) > orig_cols
        C = C[1:orig_rows, 1:orig_cols]
    end

    return C, add_count[], mul_count[]
end
end # module