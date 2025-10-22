using Revise
using QuantumFockCore
using QuantumFockDynamics
using LinearAlgebra
using Plots
using ProgressMeter
using SparseArrays
using KrylovKit




################### Phase Diagram ####################
juend=10
L = 5
Nrange = 1:10
jurange = 1:juend
pd_gap = zeros(Nrange[end],jurange[end])
pd = zeros(Nrange[end],jurange[end])
 
@showprogress for n in Nrange, ju in jurange
U = 1
J = (ju*(1/juend)) * U
N = n+1
E_scale = (J+ U*(N/L))/2

geometry = (L,)
D=length(geometry)

V = U1FockSpace(geometry,N,N)
states = basisFS(V)

latt = Lattice(geometry)

Kin, Int = Bose_Hubbard_H(V, latt, J, U)

M = calculate_matrix_elements_parallel(states,Kin + Int)
@assert M == M'
M_s = sparse(Hermitian(M))
x₀ = rand(ComplexF64, size(M)[1])
es, vs, _ =  eigsolve(M_s, x₀, 10, :SR; ishermitian=true);

gs_coeff = vs[1]
pd_gap[(n),ju] =  (es[2]- es[1]) / E_scale
gs = create_MFS( gs_coeff, states)

ρ = zeros(L,L)

for i in 1:L, j in 1:L
    ρ_ij = FockOperator(((i, true), (j,false)), 1. +0im, V)
    ρ[i,j] = real(gs * (ρ_ij*gs))
end
es_rho, vs_rho = eigen(ρ)
pd[(n),ju] = es_rho[end] / N
end;

heatmap(collect(jurange ) .* (1/juend), collect(Nrange) .+ 1, pd, xlabel="J/U", ylabel="N", color=:viridis, title="Largest eigenvalue ρ")
heatmap(collect(jurange ) .* (1/juend), collect(Nrange) .+ 1, pd_gap , xlabel="J/U", ylabel="N", color=:magma, title="Many body gap Δ")




##### 2D with Krylov methods


U = 1
J = .1* U


N = 18
geometry = (5,)
D= length(geometry)

V = U1FockSpace(geometry,N,N)
states = basisFS(V)

latt = Lattice(geometry)

Kin, Int = Bose_Hubbard_H(V, latt, J, U)
H = Kin + Int

@time M = calculate_matrix_elements_parallel(states,H)
sparseness(M)
M_s = sparse(M)
x₀ = rand(ComplexF64, size(M)[1])
@time es, _, _ = eigsolve(M, x₀, 3, :SR; ishermitian=true);
@time es, _, _ =  eigsolve(M_s, x₀, 5, :SR; ishermitian=true);
es
es
es, vs = eigen(Hermitian(M))
es
gs = create_MFS(vs[:,1], states)

density_onsite(gs, sites, geometry)
es, _ = eigen(one_body_ρ(gs, sites, geometry))
density_flucs(gs, sites, geometry)
rho = one_body_ρ(gs, sites, geometry)