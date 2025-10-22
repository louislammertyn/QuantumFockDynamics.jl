begin

############## 1. generating states ####################

function all_states_U1(V::U1FockSpace) 
    
    N= V.particle_number
    L = prod(V.geometry)
    U1occs = bounded_compositions(N, L, V.cutoff)
    states = Vector{AbstractFockState}()
    for occ in U1occs
        push!(states, fock_state(V, occ))
    end
    return states
end

function all_states_U1( V::UnrestrictedFockSpace)
    states = []
    ranges = ntuple(_->0:V.cutoff, prod(V.geometry))
    U1occs = [collect(t) for t in Iterators.product(ranges...)]
    println(U1occs)
    for occ in U1occs
        push!(states, fock_state(V, occ))
    end
    return states
end

function all_states_U1_O(V::U1FockSpace) 
    geometry = V.geometry
    L = prod(geometry)
    N = V.particle_number

    # initial Fock state
    v_i = zeros(Int, L)
    v_i[1] = N
    fs_i = fock_state(V, v_i)

    # nearest-neighbor hopping
    T = ZeroFockOperator()
    for i in 1:(L-1)
        T += FockOperator(((i,true),(i+1,false)), 1., V)
    end
    T += dagger_FO(T)

    # iterative BFS-like exploration
    queue = [fs_i]
    visited = Set([fs_i.occupations])

    while !isempty(queue)
        s = pop!(queue)
        ns = T * s
        if typeof(ns)==FockState
            ns = MultipleFockState([ns])
        end
        for s_new in ns.states
            if !(s_new.occupations in visited)
                push!(visited, s_new.occupations)
                push!(queue, s_new)
            end
        end
    end
    result::Vector{AbstractFockState} = [fock_state(V, collect(occs)) for occs in visited]
    return result
end


function bounded_compositions(N::Int, L::Int, cutoff::Int; thread_threshold::Int=10_000)
    cutoff += 1
    max_i = cutoff^L  
    
    if max_i < thread_threshold || Threads.nthreads() == 1
        # ---------------- Single-threaded version ----------------
        results = Vector{Vector{Int}}()
        for i in 0:max_i-1
            n = digits(i, base=cutoff)
            if sum(n) == N && length(n) <= L
                push!(results, reverse(n))
            end
        end
    else
        # ---------------- Multithreaded version ----------------
        thread_results = [Vector{Vector{Int}}() for _ in 1:Threads.nthreads()]
        Threads.@threads for i in 0:max_i-1
            n = digits(i, base=cutoff)
            if sum(n) == N && length(n) <= L
                push!(thread_results[Threads.threadid()], reverse(n))
            end
        end
        results = reduce(vcat, thread_results)
    end

    # Padding & sorting (shared by both paths)
    padded = results .|> x -> vcat(zeros(Int, L - length(x)), x)
    sorted = sort(padded, by=x -> evalpoly(cutoff, x), rev=true)
    return sorted
end


function basisFS(space::U1FockSpace; nodata=true, savedata=false)
    dirpath = "./src/assets/states"
    savename = "basis_u1_geom=$(join(space.geometry, 'x'))_cutoff=$(space.cutoff)_N=$(space.particle_number).jld2"
    savepath = joinpath(dirpath, savename)

    (nodata & !savedata) && return all_states_U1_O(space)
    
    # Create directory if it doesn't exist
    if !isdir(dirpath)
        mkpath(dirpath)
    end

    # Case 1: File exists and we want to use it
    if isfile(savepath) && !nodata
        data = load(savepath)
        return data["states"]

    # Case 2: File doesn't exist, but we want to save new data
    elseif !isfile(savepath) && savedata
        states = all_states_U1(space)
        save(savepath, Dict("states" => states))
        return states

    # Case 3: We don’t want to use or save data (pure computation)
    else
        return all_states_U1_O(space)
    end
end


function calculate_matrix_elements(states::Vector{AbstractFockState}, Ops::MultipleFockOperator)
    Op_matrix = zeros(ComplexF64, length(states), length(states))
    tmp = MutableFockState(states[1])
    
    for (i,bra) in enumerate(states), (j,ket) in enumerate(states)  
        total_ij = 0.0 + 0im
        for Op in Ops.terms
            reset2!(tmp, ket.occupations, ket.coefficient)                 
            apply!(Op, tmp)   
            if tuple_vector_equal(bra.occupations, tmp.occupations)
                total_ij += bra.coefficient' * tmp.coefficient
            end
        end
        if total_ij != 0. + 0im
            Op_matrix[i,j] = total_ij
        end
    
    end
    return Op_matrix
end


function tuple_vector_equal(t::NTuple{N, Int}, v::Vector{Int}) where N
    @inbounds for i in 1:N
        if t[i] != v[i]
            return false
        end
    end
    return true
end

function calculate_matrix_elements_parallel(states::Vector{AbstractFockState}, Ops::MultipleFockOperator)
    n = length(states)
    Op_matrix = zeros(ComplexF64, n, n)
    println("Calculating matrix elements using $(Threads.nthreads()) threads.")
    Threads.@threads for idx in 1:(n*n)
        i = div(idx - 1, n) + 1
        j = mod(idx - 1, n) + 1
        bra = states[i]
        ket = states[j]
        total_ij = 0.0 + 0im
        tmp = MutableFockState(ket)
        
        for Op in Ops.terms
            reset2!(tmp, ket.occupations, ket.coefficient)
            apply!(Op, tmp)

            tmp.iszero && continue

            tuple_vector_equal(bra.occupations, tmp.occupations) && (
                total_ij += bra.coefficient' * tmp.coefficient
            )
            
        end

        total_ij != 0. + 0im && (Op_matrix[i, j] = total_ij)

    end
    return Op_matrix
end



function calculate_matrix_elements_naive(states::Vector{AbstractFockState}, Op::AbstractFockOperator)
    Op_matrix = zeros(ComplexF64, length(states), length(states))
    for (i,bra) in enumerate(states), (j,ket) in enumerate(states)  
        Op_matrix[i,j] = bra * (Op * ket)   
    end
    return Op_matrix
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

