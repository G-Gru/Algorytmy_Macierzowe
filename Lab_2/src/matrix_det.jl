module matrix_det

include(joinpath(@__DIR__, "..", "src", "LU_factorization.jl"))

using .LU_factorization
export lu_det

default_mul(A, B, add_count::Ref{Int}=Ref(0), mul_count::Ref{Int}=Ref(0)) = (A * B, add_count[], mul_count[])

function lu_det(A::AbstractMatrix)
    _, U,_,_ = LU_factor(A, default_mul)
    determinant = 1
    for i in 1:size(U, 1)
        determinant *= U[i, i]
    end
    return determinant
end

end # module matrix_det