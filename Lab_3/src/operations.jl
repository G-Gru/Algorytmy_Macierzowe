module Operations

import ..create_tree
import ..tree_node

export matrix_vector_mult, matrix_matrix_add, matrix_matrix_mult

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

function new_mat(u, v, sons)
    tree_node.TreeNode{eltype(u)}(
        size(u, 2),
        (1, size(u, 1), 1, size(v, 2)),
        nothing,
        u,
        v,
        sons
    )
end

function sons_of(m)
    if isempty(m.sons)
        umid = size(m.U, 1) ÷ 2
        vmid = size(m.V, 2) ÷ 2

        u1, u2 = m.U[1:umid,:], m.U[umid+1:end,:]
        v1, v2 = m.V[:,1:vmid], m.V[:,vmid+1:end]

        lu = new_mat(u1, v1, [])
        ru = new_mat(u1, v2, [])
        ll = new_mat(u2, v1, [])
        rl = new_mat(u2, v2, [])

        [lu, ru, ll, rl]
    else
        m.sons
    end
end


function matrix_matrix_add(v, w, r=1, m=5.0)  # TODO are these default values alright?
    T = eltype(v.U)

    # both matrices are zeroed leaves
    if isempty(v.sons) && isempty(w.sons) && v.rank == 0 && w.rank == 0
        # Return a zeroed matrix of proper dimensions - TODO is this correct?
        return new_mat(zero(v.U), zero(v.V), [])
    end

    # both matrices are nonzero leaves
    if isempty(v.sons) && isempty(w.sons) && v.rank != 0 && w.rank != 0
        U = [v.U w.U]
        V = [v.V; w.V]

        # reconstructs a matrix from the values, then compresses it (costly)
        ans = U * V
        rows, cols = size(ans, 1), size(ans, 2)

        return create_tree.compression_tree(ans, 1, rows, 1, cols, r, m)
    end

    sons = [matrix_matrix_add(a, b) for (a, b) in zip(sons_of(v), sons_of(w))]
    return new_mat(zeros(T, 0, 0), zeros(T, 0, 0), sons)
end

function matrix_matrix_mult(v, w)
    T = eltype(v.U)

    if isempty(v.sons) && isempty(w.sons)
        if v.rank == 0 && w.rank == 0
            return new_mat(zero(v.U), zero(v.V), [])
        elseif v.rank != 0 && w.rank != 0
            u = v.U * (v.V * w.U)
            return new_mat(u, w.V, [])
        end
    end

    a = sons_of(v)
    b = sons_of(w)
    sons = [
        matrix_matrix_add(matrix_matrix_mult(a[1], b[1]), matrix_matrix_mult(a[2], b[3])),
        matrix_matrix_add(matrix_matrix_mult(a[1], b[2]), matrix_matrix_mult(a[2], b[4])),
        matrix_matrix_add(matrix_matrix_mult(a[3], b[1]), matrix_matrix_mult(a[4], b[3])),
        matrix_matrix_add(matrix_matrix_mult(a[3], b[2]), matrix_matrix_mult(a[4], b[4])),
    ]

    new_mat(zeros(T, 0, 0), zeros(T, 0, 0), sons)
end

end
