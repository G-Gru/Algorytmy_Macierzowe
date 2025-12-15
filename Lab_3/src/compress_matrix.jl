module compress_matrix

using LinearAlgebra
using ..tree_node

export compress

function compress(r_min::Int, r_max::Int, c_min::Int, c_max::Int,
                  U::AbstractMatrix, D::AbstractMatrix, V::AbstractMatrix,
                  r::Int)

    T = eltype(U)

    if r_min == 0 && r_max == 0 && c_min == 0 && c_max == 0
        return tree_node.TreeNode{T}(0, (r_min, r_max, c_min, c_max), T[],
                                    zeros(T, 0, 0), zeros(T, 0, 0), tree_node.TreeNode{T}[])
    end

    d = diag(D)
    rank = min(r, length(d))

    singularvalues = d[1:rank]
    Uc = U[:, 1:rank]
    Vc = D[1:rank, 1:rank] * V[1:rank, :]

    return tree_node.TreeNode{T}(rank, (r_min, r_max, c_min, c_max),
                                singularvalues, Uc, Vc, tree_node.TreeNode{T}[])
end

end # module compress_matrix