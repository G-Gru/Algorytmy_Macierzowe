module Lab_4
import Random

function network_vertex_idx(x, y, z, k)
    (x-1) * 2^2k + (y-1) * 2^k + (z-1) + 1
end

function gen_network_matrix(k, rng=Random.Xoshiro(2137))
    m = zeros(2^3k, 2^3k)

    for v1 ∈ 1:2^3k
        for sign ∈ [-1, 1]
            for side ∈ 0:2
                v2 = v1 + sign * 2 ^ (side * k)
                if 1 <= v2 <= 2^3k
                    while abs(m[v1, v2]) < 10e-8
                        m[v1, v2] = 100 * rand(rng)
                    end
                else
                    break  # don't check further indices if already out of bounds
                end
            end
        end
    end

    m
end

include("operations.jl")
end # module Lab_4