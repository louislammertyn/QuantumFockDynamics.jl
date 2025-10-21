#include("C:/Users/loulamme/OneDrive - UGent/Documenten/Assistent ELIS/Code/synthdim/FockED/src/FockSpace.jl")
using Revise
using FockSpace
using Test
using LinearAlgebra
using Plots
using ProgressMeter

spectrum = []

N = 4
geometry = (5,5)
D=length(geometry)

V = U1FockSpace(prod(geometry),N,N)
states = all_states_U1(geometry, V)

ind_v = lattice_vectorisation_map(geometry)
NN = Lattice_NN(geometry; periodic=(false, false))


phirange = 0:0.01:1
@showprogress for p in phirange

hoppings = zeros(ComplexF64, ntuple(i->geometry[mod(i,D)+1] , 2*D))
ϕ= p * 2 * π
for site in keys(NN), n in NN[site]
    index = (site..., n...)
    if n[2] - site[2] == 1 || n[2] - site[2] == geometry[2] -1
        hoppings[index...] = 1. * exp(site[1]* ϕ * 1im)
    elseif n[2] - site[2] == -1 || n[2] - site[2] == -(geometry[2] -1)
        hoppings[index...] = 1. * exp(site[1]* ϕ * -1im)
    else
        hoppings[index...] = 1. 
    end
end 


H = ZeroFockOperator()

for site in keys(NN)
    for n in NN[site]
        index = (site..., n...)
        H += FockOperator(((ind_v[site], true), (ind_v[n], false)), hoppings[index...])
    end
end

M = calculate_matrix_elements_parallel(states, H)
@assert isapprox(M, M')

es, vs = eigen(Hermitian(M))
push!(spectrum, es)

end;

pl = scatter(legend=false);
for (i,es) in enumerate(spectrum)
    scatter!(ones(length(es))* collect(phirange)[i], real.(es), markersize=0.7, color=:black, alpha=0.5)
end;
display(pl)