# In this worked example we will show signatures of fractional quantum hall states
# in the groundstate of the 2D Bose-Hubbard-Hofstadter model

using Revise
using FockSpace
using LinearAlgebra
using Plots
using ProgressMeter
using KrylovKit
using SparseArrays


J = 1
U = 10

N = 5
Lx = 6
Ly = 3

geometry = (Lx,Ly)
lattice = Lattice(geometry; periodic=(false,false));
