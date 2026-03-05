begin

############## Matrix elements calculations ####################

function calculate_matrix_elements( Ops::MultipleFockOperator, basis::Vector{AbstractFockState})
    s = MutableFockVector(MutableFockState.(basis))

    buf1 = MutableFockState(basis[1])
    buf2 = MutableFockState(basis[1])

    D = length(basis)
    if D > 5000
        # Use sparse
        matrix = spzeros(ComplexF64, D, D)
    else
        # Use dense
        matrix = zeros(ComplexF64, D, D)
    end
    for v in s.vector 
        for o_term in Ops.terms 
            op_str = o_term.product
            L = length(op_str)
            if op_str[L][2]
                ad_j!(buf1, v, op_str[L][1])
            else
                a_j!(buf1, v, op_str[L][1])
            end
            for i in (L-1):-1:1
                if op_str[i][2]
                    ad_j!(buf2, buf1, op_str[i][1])
                else
                    a_j!(buf2, buf1, op_str[i][1])
                end
                
                buf1.occupations .= buf2.occupations                
                buf1.coefficient = buf2.coefficient
                buf1.iszero = buf2.iszero

                if buf1.iszero
                    break
                end
            end
            if buf1.iszero
                continue
            else
                idcolumn = s.basis[key_from_occup(buf2.occupations, v.space.cutoff)] 
                idrow = s.basis[key_from_occup(v.occupations, v.space.cutoff)]
                matrix[idrow, idcolumn] += o_term.coefficient * buf2.coefficient
            end
        end
    end
    return matrix
end

function tuple_vector_equal(t::NTuple{N,Int}, v::Vector{Int}) where N
    @inbounds for i in 1:N
        if t[i] != v[i]
            return false
        end
    end
    return true
end
function tuple_vector_equal(t::NTuple{N,Int}, v::Vector{UInt8}) where N
    @inbounds for i in 1:N
        if t[i] != v[i]
            return false
        end
    end
    return true
end



function sparseness(M::Matrix{ComplexF64})
    s = 0
    t = 0
    for e in M 
        t+=1
        if e !=0.
            s+=1
        end
    end
    return s/t
end

########## Diagonalisation function for operator matrices ############

function diagonalise_KR(M::Matrix{ComplexF64}; states=5)
    N = size(M)[1]
    sp = sparseness(M)
    @assert N>1_000 "Matrix is small enough to do LinearAlgebra.eigen()"

    if sp > .15
        @warn "The matrix sparseness is smaller than .15 %, no sparsematrix parsing is performed"
    else 
        M = sparse(M)
    end
    x₀ = rand(ComplexF64, size(M)[1])
    x₀ ./= norm(x₀) 
    return eigsolve(M, x₀, states, :SR, ComplexF64 ; ishermitian=true)
end


############################################################
# Convert a MultipleFockState to a full many-body coefficient tensor
############################################################
"""
    MB_tensor(MBstate::MultipleFockState) -> Array{ComplexF64,N}

Given a `MultipleFockState` representing a many-body quantum state, 
returns the coefficient tensor `C` such that:

    |ψ⟩ = ∑ C[n,m,l,...] |n,m,l,...⟩

The dimensions of `C` are determined by the number of modes and the
cutoff in each mode.
"""
function MB_tensor(MBstate::MultipleFockState)
    s = MBstate.states[1]
    V = s.space
    modes = prod(V.geometry)
    dims = ntuple(i -> (V.cutoff + 1), modes)
    C = zeros(ComplexF64, dims)
    
    for state in MBstate.states
        index = collect(state.occupations) .+ 1
        C[index...] = state.coefficient
    end
    
    return C
end

############################################################
# Entanglement entropy via Schmidt decomposition
############################################################
"""
    Entanglement_Entropy(C::Array{ComplexF64,N}, cut::Int64) -> (S_ent, S)

Computes the von Neumann entanglement entropy for a bipartition
of the system after reshaping the coefficient tensor `C`.

- `cut`: number of modes in subsystem A.
- Returns:
    - `S_ent`: entanglement entropy
    - `S`: singular values (Schmidt coefficients)
"""
function Entanglement_Entropy(C::Array{ComplexF64,N}, cut::Int64) where N
    dims = size(C)
    d = dims[1]
    C_matrix = zeros(ComplexF64, d^cut, d^(N - cut))

    # Map N-dimensional tensor indices to 2D matrix
    for i in CartesianIndices(C)
        row = 0
        for t in 0:(cut-1)
            ind = cut - t
            row += (i[ind]-1) * d^t
        end

        column = 0
        for t in 0:(N - cut - 1)
            ind = N - t
            column += (i[ind]-1) * d^t
        end

        C_matrix[row + 1, column + 1] = C[i]
    end

    # Compute singular values and probabilities
    _, S, _ = svd(C_matrix)
    p = S.^2 ./ sum(S.^2)  # Schmidt probabilities

    # Von Neumann entropy
    S_ent = -sum(p[p .> 0] .* log.(p[p .> 0]))
    return S_ent, S
end

end;

