module Operations

export matrix_vector_mult

function matrix_vector_mult(v, X)
    rows = size(X, 1)

    if isempty(v.sons)
        if v.rank == 0
            return zeros(rows)
        end
        return v.U * (v.V * X)
    end

    mid = rows ÷ 2
    X1, X2 = X[1:mid,:], X[mid+1:end,:]
    @assert length(v.sons) == 4

    Y1_1 = matrix_vector_mult(v.sons[1], X1)
    Y1_2 = matrix_vector_mult(v.sons[2], X2)
    Y2_1 = matrix_vector_mult(v.sons[3], X1)
    Y2_2 = matrix_vector_mult(v.sons[4], X2)

    [Y1_1 + Y1_2; Y2_1 + Y2_2]
end

end
