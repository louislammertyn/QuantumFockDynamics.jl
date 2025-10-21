using Revise
using FockSpace
using Test
using LinearAlgebra
using Plots
using ProgressMeter

u = 0.01
t = 1.


N = 1
geometry = (10,)
D=length(geometry)

V = U1FockSpace(prod(geometry),N,N)
states = all_states_U1(geometry, V)

ind_v = lattice_vectorisation_map(geometry)
NN = Lattice_NN(geometry; periodic=(false,))