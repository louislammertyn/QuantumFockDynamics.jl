############################################################
# Fock Operator Utilities for Many-Body Quantum Systems
# 
# This file contains implementations of common Fock operators
# and utility functions to compute properties of many-body
# quantum states, such as:
#   - One-body and two-body density matrices
#   - On-site densities and fluctuations
#   - Mapping operators to momentum space
#   - Construction of common Hamiltonians (Bose-Hubbard)
############################################################

begin

a(V::AbstractFockSpace, i::Int) = FockOperator(((i, false),), 1, V)
adag(V::AbstractFockSpace, i::Int) = FockOperator(((i, true),), 1, V)
ni(V::AbstractFockSpace, i::Int) = FockOperator(((i, true), (i, false)), 1, V)


############################################################
# On-site density operator
############################################################
"""
    density_onsite(state::AbstractFockState, sites::Dict, geometry::NTuple{D, Int}) -> Array{ComplexF64,D}

Computes the expectation value ⟨n_i⟩ on each lattice site for a given Fock state.
"""
function density_onsite(state::AbstractFockState, sites::Dict, geometry::NTuple{D, Int64}) where D
    matrix = zeros(ComplexF64, geometry)
    V = typeof(state) == MultipleFockState ? state.states[1].space : state.space
    
    for s in keys(sites)
        n = ni(V, sites[s])
        matrix[s...] = state * (n * state)
    end
    return matrix
end

function center_of_mass(densities::AbstractArray)
    geometry = size(densities)
    CoM = zeros(length(geometry))
    for (i,L) in enumerate(geometry)
        shape = ntuple(d -> d==i ? L : 1, ndims(densities))
        w = reshape(collect(1:L), shape)
        CoM[i] = sum(densities .* w) / sum(densities)
    end
    return CoM
end

############################################################
# On-site density fluctuations
############################################################
"""
    density_flucs(state::AbstractFockState, sites::Dict, geometry::NTuple{D, Int}) -> Array{ComplexF64,D}

Computes the variance ⟨n_i^2⟩ - ⟨n_i⟩^2 for each site.
"""
function density_flucs(state::AbstractFockState, lattice::Lattice)
    geometry = state.space.geometry
    matrix = zeros(ComplexF64, geometry)
    sites = lattice.sites
    for s in keys(sites)
        n = FockOperator(((sites[s], true), (sites[s], false),
                          (sites[s], true), (sites[s], false)), 1. + 0im, state.space)
        matrix[s...] = state * (n * state)
    end
    return matrix - density_onsite(state, sites, geometry).^2
end

############################################################
# One-body density matrix
############################################################
"""
    one_body_ρ(state::AbstractFockState, sites::Dict, geometry::NTuple{D, Int}) -> Array{ComplexF64,2D}

Computes the one-body density matrix ρ_{ij} = ⟨a_i^† a_j⟩.
"""
function one_body_ρ(state::AbstractFockState, lattice::Lattice) 
    V = typeof(state) == MultipleFockState ? state.states[1].space : state.space

    geometry = V.geometry
    size_m = vcat(collect(geometry), collect(geometry)) |> Tuple
    ρ = zeros(ComplexF64, size_m)
    sites = lattice.sites

    for s1 in keys(sites), s2 in keys(sites)
        ind = vcat(collect(s1), collect(s2))
        Op = FockOperator(((sites[s1], true), (sites[s2], false)), 1. + 0im, V)
        ρ[ind...] = state * (Op * state)
    end

    return ρ
end


############################################################
# Hamiltonians: Bose-Hubbard
############################################################
"""
    Bose_Hubbard_H(V::U1FockSpace, lattice::Lattice, J::Number=1., U::Number=1.) -> (Kin, Int)

Constructs the kinetic and interaction parts of the Bose-Hubbard Hamiltonian:

- `Kin`: hopping term H_J
- `Int`: on-site interaction term H_U
"""
function Bose_Hubbard_H(V::U1FockSpace, lattice::Lattice, J::Number=1., U::Number=1.)
    t_K = ManyBodyTensor_init(ComplexF64, V, 1, 1)
    t_Int = ManyBodyTensor_init(ComplexF64, V, 2, 2)

    NN = lattice.NN

    # Filling conditions
    neighbour(sites_tuple) = (sites_tuple[1] ∈ NN[sites_tuple[2]]) ? J : zero(J)
    function onsite(sites_tuple::Tuple)
        @assert length(sites_tuple)==4  
        s1, s2, s3, s4 = sites_tuple 
        return (s1 == s2) & (s2 == s3) & (s3 == s4) ? U : zero(U)
    end
    

    t_K = fill_nbody_tensor(t_K, lattice, (neighbour,))
    t_Int = fill_nbody_tensor(t_Int, lattice, (onsite,))

    K = nbody_Op(V, lattice, t_K)
    I = nbody_Op(V, lattice, t_Int)

    return K, I
end



############################################################
# Map Fock operator to momentum space
############################################################
"""
    momentum_space_Op(Op::MultipleFockOperator, lattice::Lattice, dimensions::Tuple) -> MultipleFockOperator

Transforms a `MultipleFockOperator` to momentum space using FFTs.

Arguments:
- `Op`: MultipleFockOperator to transform
- `lattice`: Lattice object
- `dimensions`: tuple of FFT dimensions for each spatial direction

Returns:
- `Op_momentum`: MultipleFockOperator in momentum space
"""
function momentum_space_Op(Op::MultipleFockOperator, lattice::Lattice, dimensions::Tuple)
    tensors = extract_nbody_tensors(Op, lattice)
    
    @assert (length(tensors)==2)
    for t in tensors
        s = t.domain + t.codomain
        @assert ( (s != 2) || (s!=4)) "Momentumspace functionality only defined for 1 body and 2 body operators"
        if s == 2
            real_tensor_2body = t 
        elseif s== 4
            real_tensor_4body = t 
        end
    end

    V = Op.terms[1].space
    geometry = V.geometry
    D = length(geometry)
    dimensions_bra = dimensions
    dimensions_ket = Tuple(collect(dimensions) .+ D)

    # --- 2-body transformation ---
    #real_tensor_2body = get_tensor_2body(Op, lattice)
    if iszero(real_tensor_2body)
        tensor_2body_m = zeros(ComplexF64, nbody_geometry(geometry, 2))
    else
        tensor_2body_m = fft(real_tensor_2body, dimensions_bra)
        tensor_2body_m = ifft(tensor_2body_m, dimensions_ket)
    end

    # --- 4-body transformation ---
    #real_tensor_4body = get_tensor_4body(Op, lattice)
    if iszero(real_tensor_4body)
        tensor_4body_m = zeros(ComplexF64, nbody_geometry(geometry, 4))
    else
        bra_dims_4body  = collect(dimensions)
        bra_dims_4body2 = collect(dimensions) .+ D
        ket_dims_4body  = collect(dimensions) .+ 2*D
        ket_dims_4body2 = collect(dimensions) .+ 3*D

        tensor_4body_m = fft(real_tensor_4body, bra_dims_4body)
        tensor_4body_m = fft(tensor_4body_m, bra_dims_4body2)
        tensor_4body_m = ifft(tensor_4body_m, ket_dims_4body)
        tensor_4body_m = ifft(tensor_4body_m, ket_dims_4body2)
        tensor_4body_m .= tensor_4body_m
    end

    # Construct momentum-space operator
    return two_body_Op(V, lattice, tensor_2body_m) + four_body_Op(V, lattice, tensor_4body_m)
end


"""
This functionality implements operator transformations under either:

1. Projections onto a subset of the total single particle Hilbert space, or
2. Full unitary basis transformations on the many-body Fock operators.

The transformations are of the form:

    d†_α = Σ_i ϕ_i^α c†_i

where ϕ_i^α=⟨i|ϕ^α⟩ are either:

- The eigenstates |ϕ^α⟩ defining the subspace onto which one projects where one 
  then assumes the projection on the subspace as c†_i ≈ Σ_α (ϕ_i^α)*d†_α, or
- If they form an orthonormal set, the basis functions into which the Fock operators are transformed.

The matrix encoding the projection or transformation is denoted as:

    M_αi = φ_i^α

Please note the the i index labels the vectorised modes of the system and α labels the eigenstates |ϕ^α>
"""

function transform(O::MultipleFockOperator, lattice::Lattice, modes::Matrix{ComplexF64})
    if size(modes,1) == size(modes,2)
        @assert isapprox(modes * modes', I, atol=1e-12)
    end

    V = O.terms[1].space
    new_geometry = (size(modes,1),)
    new_lattice = Lattice(new_geometry)
    new_V = V isa UnrestrictedFockSpace ? UnrestrictedFockSpace(new_geometry, V.cutoff) :
            V isa U1FockSpace         ? U1FockSpace(new_geometry, V.cutoff, V.particle_number) :
            error("Unsupported Fock space type: $(typeof(V))")

    tnsrs = extract_nbody_tensors(O, lattice)
    new_tnsrs = Vector{ManyBodyTensor}()

    for t_ in tnsrs
        t_v = vectorize_tensor(t_, lattice).tensor

        dom = t_.domain
        codom = t_.codomain
        N = dom + codom

        # build index strings
        old_tensor_indices = 1:N
        new_tensor_indices = -1 .* (1:N)

        tnsrs_prod = Vector{SparseArray}()
        indices = Vector{Vector{Int64}}()
        modes_sp = SparseArray(modes)
        modes_adj_sp = SparseArray(conj.(modes))

        for i in 1:codom
            push!(tnsrs_prod, modes_adj_sp)
            push!(indices, [new_tensor_indices[i], old_tensor_indices[i]])
        end
        for i in codom+1:N
            push!(tnsrs_prod, modes_sp)
            push!(indices, [new_tensor_indices[i], old_tensor_indices[i]])
        end
        push!(tnsrs_prod, t_v)
        push!(indices, collect(old_tensor_indices))
        

        t_new_v = ncon(Tuple(tnsrs_prod), Tuple(indices))
        t_new_v = ManyBodyTensor(t_new_v, new_V, dom, codom)

        push!(new_tnsrs, t_new_v)

    end

    return construct_Multiple_Operator(new_V, new_lattice, new_tnsrs)
end

"""
If given a list of modes then it is assumed that the mode matrices correspond to transformations along the different labels e.g.

 d†_α = Σ_i ϕ_i^α c†_i

the ϕ are now a product of functions with each factor corresponding to the transformation function of it's respective basis transformation
"""

function transform(O::MultipleFockOperator, lattice::Lattice, modes::Tuple{Matrix{ComplexF64}})
    for mode in modes
        if size(mode,1) == size(mode,2)
            @assert isapprox(mode * mode', I, atol=1e-12)
        end     
    end

    V = O.terms[1].space
    new_geometry = Tuple([size(mode,1) for mode in modes])
    new_lattice = Lattice(new_geometry)
    new_V = V isa UnrestrictedFockSpace ? UnrestrictedFockSpace(new_geometry, V.cutoff) :
            V isa U1FockSpace         ? U1FockSpace(new_geometry, V.cutoff, V.particle_number) :
            error("Unsupported Fock space type: $(typeof(V))")

    tnsrs = extract_nbody_tensors(O, lattice)
    new_tnsrs = Vector{ManyBodyTensor}()

    modes_vectorised = zeros(ComplexF64, prod(new_geometry), prod(lattice.geometry))

    for (s_old, s_old_v) in lattice.sites, (s_new, s_new_v) in new_lattice.sites
        modes_vectorised[s_new_v, s_old_v] = prod([modes[i][s_new[i], s_old[i]] for i in eachindex(modes)] )
    end

    for t_ in tnsrs
        t_v = vectorize_tensor(t_, lattice).tensor

        dom = t_.domain
        codom = t_.codomain
        N = dom + codom

        # build index strings
        old_tensor_indices = 1:N
        new_tensor_indices = -1 .* (1:N)

        tnsrs_prod = Vector{SparseArray}()
        indices = Vector{Vector{Int64}}()
        modes_sp = SparseArray(modes_vectorised)

        for i in 1:dom 
            push!(tnsrs_prod, modes_sp)
            push!(indices, [new_tensor_indices[i], old_tensor_indices[i]])
        end
        for i in dom+1:N
            push!(tnsrs_prod, conj.(modes_sp))
            push!(indices, [new_tensor_indices[i], old_tensor_indices[i]])
        end
        push!(tnsrs_prod, t_v)
        push!(indices, old_tensor_indices)
        

        t_new_v = ncon(Tuple(tnsrs_prod), Tuple(indices))
        t_new_v = ManyBodyTensor(t_new_v, new_V, dom, codom)

        push!(new_tnsrs, devectorize_tensor(t_new_v, new_lattice))

    end

    return construct_Multiple_Operator(new_V, new_lattice, new_tnsrs)
end


function reduce_terms(Op::MultipleFockOperator, term_condition::Function)
    reduced_Op = ZeroFockOperator()
    for o in Op.terms
        if term_condition(o)
            reduced_Op += o 
        end
    end
    return reduced_Op
end
############################################################
# End of Fock operator utilities
############################################################

end;
