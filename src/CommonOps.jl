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
    t_K = ManyBodyTensor(ComplexF64, V, 1, 1)
    t_Int = ManyBodyTensor(ComplexF64, V, 2, 2)

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

    K = n_body_Op(V, lattice, t_K)
    I = n_body_Op(V, lattice, t_Int)

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
    tensors = extract_n_body_tensors(Op, lattice)
    
    @assert (length(tensors)==2)
    for t in tensors
        s = length(size(t))
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


############################################################
# End of Fock operator utilities
############################################################

end;
