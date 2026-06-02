module FoSpDynamics

using Reexport
@reexport using FoSpCore
using LinearAlgebra
using OrdinaryDiffEq
using TensorOperations
using SparseArrayKit
using SparseArrays


include("./utils.jl")
include("./ED.jl")
include("./TimeEv.jl")
include("./CommonOps.jl")
include("./MeanField.jl")
include("./Bogoliubov.jl")
include("./Thermal.jl")


#####################################################################################################
#####################################################################################################

export combinations, Identity, fold, folded_k_from_modes

#####################################################################################################
#####################################################################################################

export Time_Evolution_ed,
       Time_Evolution,
       schrodinger!,
       Time_Evolution_TD,
       Time_Evolution_VN,
       Time_Evolution_TD_VN,
       Time_Evolution_TDM_VN,
       schrodinger_TD!,
       Heisenberg_eom,
       von_neumann!,
       Von_Neumann_TD!,
       Von_Neumann_TDM!,
       Unitary_Ev,
       Unitary_Ev_TD,
       Unitary_Ev_Op,
       Unitary_Ev_Op_TD

#####################################################################################################
#####################################################################################################

export a, adag, ni
export single_particle_matrix, single_particle_sector, single_particle_operator
export density_onsite, center_of_mass, one_body_ρ, density_flucs, momentum_density
export Hopping_Ham, Bose_Hubbard_H, delta, momentum_space_Op, bloch_matrix

#####################################################################################################
#####################################################################################################

export all_states_U1, all_states_U1_O, bounded_compositions, basisFS
export calculate_matrix_elements
export tuple_vector_equal, sparseness, diagonalise_KR, MB_tensor, Entanglement_Entropy
export transform, reduce_terms

#####################################################################################################
#####################################################################################################

export MFMonomial, MF, MFeq
export construct_MF, construct_eoms
export eval_MF
export GPE!
export make_normalisation_cb, make_convergence_cb
export get_mf_groundstate
export mf_time_evolution
export get_μ

#####################################################################################################
#####################################################################################################

export BogoliubovRep, construct_BogoliubovRep, eval_Bog, Bogoliubov_spectrum, sort_bg, Bogoliubov_gs, plot_Bogoliubov_spectrum
export J, depletion

#####################################################################################################
#####################################################################################################

export thermal_ρ_matrix,
       thermal_exp,
       Liouvillian_Super,
       Time_Evolution_thermal_ρ_Liouv,
       Time_Evolution_thermal_ρ_TD_Liouv,
       Time_Evolution_thermal_ρ_TD_VN,
       Unitary_Ev_ρ_TD,
       Unitary_Ev_ρ
end
