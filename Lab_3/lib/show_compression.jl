module show_compression

include(joinpath(@__DIR__, "show_image.jl"))

using .show_image

export show_comp, rank_map

function _leaves(v)
    if isempty(v.sons)
        return [v]
    end
    out = []
    for s in v.sons
        append!(out, _leaves(s))
    end
    return out
end

function _draw_border!(X, r1::Int, r2::Int, c1::Int, c2::Int, val::Float64)
    X[r1, c1:c2] .= val
    X[r2, c1:c2] .= val
    X[r1:r2, c1] .= val
    X[r1:r2, c2] .= val
    return X
end

function _border_map(root; pad_square::Bool=true)
    rmin, rmax, cmin, cmax = root.size
    nrows = rmax - rmin + 1
    ncols = cmax - cmin + 1
    N = pad_square ? max(nrows, ncols) : nrows
    M = pad_square ? max(nrows, ncols) : ncols

    X = fill(0.0, N, M)

    roff = 1 - rmin
    coff = 1 - cmin

    for leaf in _leaves(root)
        a, b, c, d = leaf.size
        r1 = a + roff; r2 = b + roff
        c1 = c + coff; c2 = d + coff
        _draw_border!(X, r1, r2, c1, c2, 1.0)
    end

    return X
end

function show_comp(root; pad_square::Bool=true, savepath::Union{Nothing,String}=nothing)
    X = _border_map(root; pad_square=pad_square)
    return show_grey(X; savepath=savepath)
end

end # module show_compression