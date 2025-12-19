module show_compression

include(joinpath(@__DIR__, "show_image.jl"))

using .show_image

export show_comp, rank_map, show_comp_thick, show_comp_thick_colored

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

function _draw_border_thick!(
    X,
    r1::Int, r2::Int, c1::Int, c2::Int,
    val::Float64,
    thickness::Int
)
    t = max(1, thickness)
    for i in 0:(t - 1)
        rr1 = r1 + i; rr2 = r2 - i
        cc1 = c1 + i; cc2 = c2 - i
        (rr1 > rr2 || cc1 > cc2) && break

        X[rr1, cc1:cc2] .= val
        X[rr2, cc1:cc2] .= val
        X[rr1:rr2, cc1] .= val
        X[rr1:rr2, cc2] .= val
    end
    return X
end

function _border_map(root; pad_square::Bool=true, base_thickness::Int=1)
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

        thickness = base_thickness * max(1, leaf.rank)
        _draw_border_thick!(X, r1, r2, c1, c2, 1.0, thickness)
    end

    return X
end

function rank_map(root; pad_square::Bool=true)
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
        X[r1:r2, c1:c2] .= Float64(leaf.rank)
    end

    return X
end


function show_comp_thick(
    root;
    pad_square::Bool=true,
    base_thickness::Int=1,
    savepath::Union{Nothing,String}=nothing
)
    X = _border_map(root; pad_square=pad_square, base_thickness=base_thickness)
    return show_grey(X; savepath=savepath)
end

function show_comp_thick_colored(
    root,
    pad_square::Bool=true,
    base_thickness::Int=1,
    savepath::Union{Nothing,String}=nothing
)
    rmin, rmax, cmin, cmax = root.size
    nrows = rmax - rmin + 1
    ncols = cmax - cmin + 1
    N = pad_square ? max(nrows, ncols) : nrows
    M = pad_square ? max(nrows, ncols) : ncols

    R = fill(0.0, N, M)
    G = fill(0.0, N, M)
    B = fill(0.0, N, M)

    roff = 1 - rmin
    coff = 1 - cmin

    leaves = _leaves(root)
    means  = [isempty(l.singularvalues) ? 0.0 : sum(l.singularvalues) / length(l.singularvalues) for l in leaves]

    maxs = maximum(means)

    for leaf in _leaves(root)
        a, b, c, d = leaf.size
        r1 = a + roff; r2 = b + roff
        c1 = c + coff; c2 = d + coff

        thickness = base_thickness * max(1, leaf.rank)

        _draw_border_thick!(R, r1, r2, c1, c2, 1.0, thickness)
        _draw_border_thick!(G, r1, r2, c1, c2, 1.0, thickness)
        _draw_border_thick!(B, r1, r2, c1, c2, 1.0, thickness)

        ri1 = r1 + thickness
        ri2 = r2 - thickness
        ci1 = c1 + thickness
        ci2 = c2 - thickness

        if ri1 <= ri2 && ci1 <= ci2
           mean_s = isempty(leaf.singularvalues) ? 0.0 : Float64(sum(leaf.singularvalues)) / length(leaf.singularvalues)
            intensity = clamp(mean_s / maxs, 0.0, 1.0)
            R[ri1:ri2, ci1:ci2] .= intensity
        end
    end

    return show_color(R, G, B; space=:rgb, savepath=savepath)
end

end # module show_compression