module create_tree

using LinearAlgebra
using ..tree_node
using ..compress_matrix

export compression_tree

function truncated_svd(A::AbstractMatrix, k::Int)
    F = svd(A; full=false)
    k = min(k, length(F.S))
    U = F.U[:, 1:k]
    D = Diagonal(F.S[1:k])
    V = F.Vt[1:k, :]   # V = Vᵀ
    return U, D, V
end

function compression_tree(A::AbstractMatrix,
                          r_min::Int, r_max::Int, c_min::Int, c_max::Int,
                          r::Int, m::Float64, relative::Bool=false)

    T = eltype(A)
    sub = @view A[r_min:r_max, c_min:c_max]

    if r_min == r_max || c_min == c_max
        k = min(r + 1, minimum(size(sub)))
        U, D, V = truncated_svd(Matrix(sub), k)
        return compress(r_min, r_max, c_min, c_max, U, D, V, min(r, k))
    end

    k = min(r + 1, minimum(size(sub)))
    if k <= r
        U, D, V = truncated_svd(Matrix(sub), k)
        return compress(r_min, r_max, c_min, c_max, U, D, V, min(r, k))
    end

    U, D, V = truncated_svd(Matrix(sub), r + 1)
    s = diag(D)

    comparator = relative ? (s[r + 1] / s[1]) : s[r + 1]

    if comparator <= m
        return compress(r_min, r_max, c_min, c_max, U, D, V, r)
    end

    mid_r = (r_min + r_max) ÷ 2
    mid_c = (c_min + c_max) ÷ 2

    v = tree_node.TreeNode{T}(-1, (r_min, r_max, c_min, c_max), T[],
                              zeros(T, 0, 0), zeros(T, 0, 0), tree_node.TreeNode{T}[])

    push!(v.sons, compression_tree(A, r_min,     mid_r,   c_min,     mid_c,   r, m, relative))
    push!(v.sons, compression_tree(A, r_min,     mid_r,   mid_c + 1, c_max,   r, m, relative))
    push!(v.sons, compression_tree(A, mid_r + 1, r_max,   c_min,     mid_c,   r, m, relative))
    push!(v.sons, compression_tree(A, mid_r + 1, r_max,   mid_c + 1, c_max,   r, m, relative))

    return v
end

end # module create_tree