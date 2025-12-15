module show_image


include(joinpath(@__DIR__, "tree_node.jl"))
using .tree_node

using Colors
using FileIO

export show_grey, show_color

function _tree_to_matrix(root::tree_node.TreeNode)
    rmin, rmax, cmin, cmax = root.size
    nrows = rmax - rmin + 1
    ncols = cmax - cmin + 1

    if isempty(root.sons)
        if root.rank == 0
            return zeros(Float64, nrows, ncols)
        end
        return Float64.(root.U * root.V)   # leaf stores U and (D*Vᵀ)
    end

    X = zeros(Float64, nrows, ncols)
    for child in root.sons
        crmin, crmax, ccmin, ccmax = child.size
        roff = crmin - rmin + 1
        coff = ccmin - cmin + 1
        sub = _tree_to_matrix(child)
        X[roff:roff + size(sub, 1) - 1, coff:coff + size(sub, 2) - 1] .= sub
    end

    return X
end

function _tree_to_matrix(root)
    rmin, rmax, cmin, cmax = root.size
    nrows = rmax - rmin + 1
    ncols = cmax - cmin + 1

    if isempty(root.sons)
        if root.rank == 0
            return zeros(Float64, nrows, ncols)
        end
        return Float64.(root.U * root.V)   # leaf stores U and (D*Vᵀ)
    end

    X = zeros(Float64, nrows, ncols)
    for child in root.sons
        crmin, crmax, ccmin, ccmax = child.size
        roff = crmin - rmin + 1
        coff = ccmin - cmin + 1
        sub = _tree_to_matrix(child)
        X[roff:roff + size(sub,1) - 1, coff:coff + size(sub,2) - 1] .= sub
    end
    return X
end

# map any numeric matrix to [0,1] (simple, notebook-friendly)
function _norm01(A::AbstractMatrix)
    amin = minimum(A)
    amax = maximum(A)
    if amax == amin
        return fill(0.0, size(A))
    end
    return (A .- amin) ./ (amax - amin)
end

# --- GREY ---

function show_grey(root; savepath::Union{Nothing,String}=nothing)
    return show_grey(_tree_to_matrix(root); savepath=savepath)  # calls matrix version
end

function show_grey(root::tree_node.TreeNode; savepath::Union{Nothing,String}=nothing)
    A = _tree_to_matrix(root)
    return show_grey(A; savepath=savepath)   # calls the existing matrix method
end

function show_grey(A::AbstractMatrix; savepath::Union{Nothing,String}=nothing)
    X = _norm01(Float64.(A))
    img = Gray.(X)              # matrix of Gray pixels (shows in .ipynb)

    if savepath !== nothing
        save(savepath, img)
    end

    return img
end

# --- COLOR ---

function show_color(t1, t2, t3; space::Symbol=:rgb, savepath::Union{Nothing,String}=nothing)
    return show_color(_tree_to_matrix(t1), _tree_to_matrix(t2), _tree_to_matrix(t3);
                      space=space, savepath=savepath)
end

function show_color(t1::TreeNode, t2::TreeNode, t3::TreeNode;
                    space::Symbol=:rgb,
                    savepath::Union{Nothing,String}=nothing)
    A1 = _tree_to_matrix(t1)
    A2 = _tree_to_matrix(t2)
    A3 = _tree_to_matrix(t3)
    return show_color(A1, A2, A3; space=space, savepath=savepath)  # existing matrix method
end

function show_color(A1::AbstractMatrix, A2::AbstractMatrix, A3::AbstractMatrix;
                    space::Symbol=:rgb,
                    savepath::Union{Nothing,String}=nothing)

    @assert size(A1) == size(A2) == size(A3) "All 3 matrices must have the same size"

    if space == :rgb
        R = _norm01(Float64.(A1))
        G = _norm01(Float64.(A2))
        B = _norm01(Float64.(A3))
        img = RGB.(R, G, B)

    elseif space == :hsv
        H = Float64.(A1); S = Float64.(A2); V = Float64.(A3)

        # heuristic: if H looks like degrees, scale to [0,1]
        if maximum(H) > 1.0
            H = H ./ 360.0
        end
        H = clamp.(H, 0.0, 1.0)
        S = clamp.(_norm01(S), 0.0, 1.0)
        V = clamp.(_norm01(V), 0.0, 1.0)

        img = RGB.(HSV.(H, S, V))

    elseif space == :lab
        L = Float64.(A1); a = Float64.(A2); b = Float64.(A3)

        # heuristic: if L is normalized, scale to [0,100]
        if maximum(L) <= 1.0
            L = 100.0 .* L
        end

        # heuristic: if a/b are normalized, map [0,1] -> [-128,127]
        if minimum(a) >= 0.0 && maximum(a) <= 1.0
            a = (a .- 0.5) .* 255.0
        end
        if minimum(b) >= 0.0 && maximum(b) <= 1.0
            b = (b .- 0.5) .* 255.0
        end

        img = RGB.(Lab.(L, a, b))

    else
        throw(ArgumentError("space must be :rgb, :hsv, or :lab"))
    end

    if savepath !== nothing
        save(savepath, img)
    end

    return img
end

end # module show_image