module Operations

import ..create_tree
import ..tree_node

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

function new_mat(u, v, sons)
    tree_node.TreeNode{T}(
        size(u, 2),
        (1, )
        # suggested TODO - simple constructor for compressed matrices
    )
end

function matrix_matrix_add(v, w, r=1, m=5.0)  # TODO are these default values alright?
    r_min, c_min = 1, 1
    r_max, c_max = size(v.U, 2), size(v.V, 1)
    T = eltype(v.U)

    # both matrices are zeroed leaves
    if isempty(v.sons) && isempty(w.sons) && v.rank == 0 && w.rank == 0
        # Return a zeroed matrix of proper dimensions - TODO is this correct?
        return tree_node.TreeNode{T}(0, (r_min, r_max, c_min, c_max), T[],
                                    zero(v.U), zero(v.V), tree_node.TreeNode{T}[])
    end

    # both matrices are nonzero leaves
    if isempty(v.sons) && isempty(w.sons) && v.rank =! 0 && w.rank =! 0
        U = [v.U w.U]
        V = [v.V; w.V]

        # reconstructs a matrix from the values, then compresses it (costly)
        m = U * V
        rows, cols = size(m, 1), size(m, 2)

        return create_tree.compression_tree(m, 1, rows, 1, cols, r, m)
    end

    sons_of = function(m)
        if isempty(m.sons)
            umid = size(m.U, 1) ÷ 2
            vmid = size(m.V, 2) ÷ 2

            u1, u2 = u[1:umid,:], u[umid+1:end,:]
            v1, v2 = v[:,1:vmid], v[:,vmid+1:end]

            #TODO
        else
            m.sons
        end
    end

    # none of the matrices are leaves, recurse them
    if !isempty(v.sons) and !isempty(w.sons)
        sons = [matrix_matrix_add(a, b) for (a, b) in zip(v.sons, w.sons)]
        return tree_node.TreeNode{T}() # TODO construct
    end

    # only one of the matrices is a leaf, split it
end

end
