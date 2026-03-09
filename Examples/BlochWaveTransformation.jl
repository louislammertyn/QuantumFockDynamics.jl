using Revise
using LinearAlgebra, Plots
using FoSpDynamics

# ─── parameters ──────────────────────────────────────────────────────────────

Lx = 10
Ly = 20
tx  = 3.0
ty = -2.0

# ─── build the lattice and Fock space ────────────────────────────────────────

geometry = (Lx, Ly)
lattice  = Lattice(geometry; periodic=(true,true))
V        = U1FockSpace(geometry, 2, 2)   # 2 particles 

# ─── build the real-space Hamiltonian ────────────────────────────────────────
# 1D nearest-neighbour hopping with PBC
function hopping_1dx(sites_tuple)
    s1, s2 = sites_tuple
    if s1 ∈ lattice.NN[s2] && s1[2] == s2[2]
        return -tx 
    end
    return zero(ComplexF64)
end

function hopping_1dy(sites_tuple)
    s1, s2 = sites_tuple
    if s1 ∈ lattice.NN[s2] && s1[1] == s2[1]
        return -ty
    end
    return zero(ComplexF64)
end

h = ManyBodyTensor_init(ComplexF64, V, 1, 1)
h = fill_nbody_tensor(h, lattice, (hopping_1dx, hopping_1dy))

H = nbody_Op(V, lattice, h)   # adapt to your constructor
H - dagger_FO(H)
# ─── build the Bloch mode matrices ───────────────────────────────────────────
# M[k+1, r+1] = (1/√L) exp(i*2π*k*r/L)   →   unitary DFT matrix

function bloch_matrix(L::Int)
    M = Matrix{ComplexF64}(undef, L, L)
    for k in 0:L-1, r in 0:L-1
        M[k+1, r+1] = exp(im * 2π * k * r / L) / sqrt(L)
    end
    return M
end

Mx = bloch_matrix(Lx)
My = bloch_matrix(Ly)

println("Mx unitary check: ", isapprox(Mx * Mx', I(Lx), atol=1e-12))
println("My unitary check: ", isapprox(My * My', I(Ly), atol=1e-12))

# ─── apply the Bloch transform ───────────────────────────────────────────────
H_bloch = transform(H, lattice, (Mx, My))
show_lattice(H, lattice)

show_lattice( H_bloch, lattice)
H_bloch - dagger_FO(H_bloch)
# ─── inspect the result ──────────────────────────────────────────────────────

new_lattice = Lattice(geometry)
tnsr      = extract_nbody_tensors(H_bloch, lattice)[1]

begin
# extract numerical dispersions on the full (kx, ky) grid
kx_vals = [mod((i-1) * (2π/Lx) + π, 2π) - π for i in 1:Lx]
ky_vals = [mod((j-1) * (2π/Ly) + π, 2π) - π for j in 1:Ly]

ε_numerical  = zeros(Float64, Lx, Ly)
ε_analytical = zeros(Float64, Lx, Ly)

for i in 1:Lx, j in 1:Ly
    ε_numerical[i,j]  = real(tnsr[i,j,i,j])
    ε_analytical[i,j] = -2tx * cos(kx_vals[i]) - 2ty * cos(ky_vals[j])
end

# ── sort both grids by (kx, ky) for clean surface rendering ──────────────────
ix = sortperm(kx_vals)
iy = sortperm(ky_vals)
kx_sorted = kx_vals[ix]
ky_sorted = ky_vals[iy]
ε_num_sorted  = ε_numerical[ix, iy]
ε_ana_sorted  = ε_analytical[ix, iy]

# ── 3D comparison plot ───────────────────────────────────────────────────────
pl = plot(
    layout    = (1, 2),
    size      = (1200, 500),
    xlabel    = "kx",
    ylabel    = "ky",
    zlabel    = "ε(k)",
)

surface!(pl[1],
    kx_sorted, ky_sorted, ε_ana_sorted',
    title  = "Analytical: -2tx·cos(kx) - 2ty·cos(ky)",
    color  = :blues,
    alpha  = 0.8,
)

surface!(pl[2],
    kx_sorted, ky_sorted, ε_num_sorted',
    title  = "Numerical (Bloch transform)",
    color  = :reds,
    alpha  = 0.8,
)

display(pl)


# ── print max error ───────────────────────────────────────────────────────────
println("Max error: ", maximum(abs.(ε_numerical .- ε_analytical)))
println("Mean error: ", abs.(ε_numerical .- ε_analytical) |> x -> sum(x) / length(x))
end