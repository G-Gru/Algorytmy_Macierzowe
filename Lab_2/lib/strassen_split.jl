module strassen_split

export strassen_spl

function matadd(A, B, add_count)
    @assert size(A) == size(B)
    add_count[] += prod(size(A))
    return A .+ B
end

function matsub(A, B, add_count)
    @assert size(A) == size(B)
    add_count[] += prod(size(A))
    return A .- B
end

# Beware: This implementation is not complete and does not work correctly

function strassen_spl(A::Number, B::Number,
                  add_count::Ref{Int}=Ref(0),
                  mul_count::Ref{Int}=Ref(0))
    mul_count[] += 1
    return [A * B], add_count[], mul_count[]
end

function strassen_spl(A::AbstractVector, B::AbstractVector,
                  add_count::Ref{Int}=Ref(0),
                  mul_count::Ref{Int}=Ref(0))
    @assert length(A) == length(B)
    s = zero(eltype(A))
    if length(A) == 1
        mul_count[] += 1
        return [A[1] * B[1]], add_count[], mul_count[]
    end
    for i in eachindex(A, B)
        mul_count[] += 1
        add_count[] += 1
        s += A[i] * B[i]
    end
    return [s], add_count[], mul_count[]
end

function strassen_spl(A::AbstractMatrix, B::AbstractMatrix,
                  add_count::Ref{Int}=Ref(0),
                  mul_count::Ref{Int}=Ref(0))

    arows, acols = size(A)
    brows, bcols = size(B)
    @assert acols == brows "incompatible matrix dimensions"

    if arows == 1 && acols == 1 && bcols == 1
        mul_count[] += 1
        return [A[1,1] * B[1,1]], add_count[], mul_count[]
    end

    m = arows ÷ 2
    k = acols ÷ 2
    n = bcols ÷ 2

    has_bottom = arows % 2 != 0
    has_rightA = acols % 2 != 0
    has_rightB = bcols % 2 != 0

    A11 = @view A[1:m, 1:k];        A12 = @view A[1:m, k+1:end]
    A21 = @view A[m+1:end, 1:k];    A22 = @view A[m+1:end, k+1:end]

    B11 = @view B[1:k, 1:n];        B12 = @view B[1:k, n+1:end]
    B21 = @view B[k+1:end, 1:n];    B22 = @view B[k+1:end, n+1:end]

    P1, _, _ = strassen_spl(matadd(Array(A11), Array(A22), add_count), matadd(Array(B11), Array(B22), add_count), add_count, mul_count)
    P2, _, _ = strassen_spl(matadd(Array(A21), Array(A22), add_count), Array(B11), add_count, mul_count)
    P3, _, _ = strassen_spl(Array(A11), matsub(Array(B12), Array(B22), add_count), add_count, mul_count)
    P4, _, _ = strassen_spl(Array(A22), matsub(Array(B21), Array(B11), add_count), add_count, mul_count)
    P5, _, _ = strassen_spl(matadd(Array(A11), Array(A12), add_count), Array(B22), add_count, mul_count)
    P6, _, _ = strassen_spl(matsub(Array(A21), Array(A11), add_count), matadd(Array(B11), Array(B12), add_count), add_count, mul_count)
    P7, _, _ = strassen_spl(matsub(Array(A12), Array(A22), add_count), matadd(Array(B21), Array(B22), add_count), add_count, mul_count)

    C11 = matadd(matadd(P1, P4, add_count), matsub(P7, P5, add_count), add_count)
    C12 = matadd(P3, P5, add_count)
    C21 = matadd(P2, P4, add_count)
    C22 = matadd(matadd(P1, P3, add_count), matsub(P6, P2, add_count), add_count)

    C = vcat(hcat(C11, C12), hcat(C21, C22))

    # handle remainders and add them to C
    full_rows = size(C, 1)
    full_cols = size(C, 2)

    if has_rightA || has_rightB || has_bottom
        C_ext = zeros(eltype(C), arows, bcols)
        C_ext[1:full_rows, 1:full_cols] .= C

        if has_rightB
            B_extra = @view B[:, bcols:end]
            C_ext[:, end] .= (A * B_extra)[:, 1]
        end

        if has_bottom
            A_extra = @view A[arows:end, :]
            C_ext[end, :] .= (A_extra * B)[1, :]
        end

        if has_bottom && has_rightB
            C_ext[end, end] = (A[arows, :] * B[:, bcols])[1]
        end

        C_with_remainder = C_ext
    else
        C_with_remainder = C
    end

    return C_with_remainder, add_count[], mul_count[]
end

end