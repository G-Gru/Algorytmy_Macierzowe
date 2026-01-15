using Lab_2.gaussian_elimination

using LinearAlgebra
using Random

function triangulates_correctly(A::Matrix{Float64}, b::Vector{Float64})
    Ares, bres = gaussian_elimination.triangulate(A, b)
    @test all(iszero, tril(Ares, -1))
    @test isapprox(Ares \ bres, A \ b, atol=1e-8)
end

@testset "Predefined matrix elimination" begin
    # 2×2
    A = [0.1 0.05
         0.2 0.3]
    b = [0.1, 0.3]
    Ares, bres = gaussian_elimination.triangulate(A, b)
    @test isapprox(Ares,
        [0.1 0.05
         0.0 0.2]
    )
    @test isapprox(bres, [0.1, 0.1])

    # 3×3
    A = [ 2.0  1.0 -1.0
         -3.0 -1.0  2.0
         -2.0  1.0  2.0]
    b = [8.0, -11.0, -3.0]
    Ares, bres = gaussian_elimination.triangulate(A, b)
    @test isapprox(Ares,
        [2.0 1.0 -1.0
         0.0 0.5  0.5
         0.0 0.0 -1.0]
    )
    @test isapprox(bres, [8.0, 1.0, 1.0])

    # 4×4
    A = [1.0 2.0 0.0 1.0
         2.0 1.0 3.0 0.0
         0.0 1.0 1.0 2.0
         1.0 0.0 2.0 1.0]
    b = [1.0, 5.0, 2.0, 3.0]
    triangulates_correctly(A, b)
end

# @testset "Random matrix elimination" begin
#     rng = MersenneTwister(2137)
#     for i ∈ 1:5
#         local A, b, c
#         while true
#             A = rand(rng, i, i)
#             b = rand(rng, i)
#             c = cond(A)
#             c < 1 && break
#         end
#         println("dla $i jest cond $c")
#         triangulates_correctly(A, b)
#     end
# end
