module tree_node

export TreeNode

mutable struct TreeNode{T}
    rank::Int
    size::NTuple{4,Int}                 # (r_min, r_max, c_min, c_max)
    singularvalues::Vector{T} | nothing
    U::Matrix{T}
    V::Matrix{T}                        # stores (D*Vᵀ) truncated
    sons::Vector{TreeNode{T}}
end

end # module tree_node
