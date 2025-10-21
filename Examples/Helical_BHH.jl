using Revise
using FockSpace
using LinearAlgebra
using Plots, ColorSchemes
using ProgressMeter
using KrylovKit
using SparseArrays





N = 1
Lx = 60
Ly = 2

geometry = (Lx,Ly)
lattice = Lattice(geometry; periodic=(true,false), helical=(false,true));

lattice.NN
lattice.NN_v
lattice.sites
V = U1FockSpace(geometry, N, N)
states = all_states_U1_O(V)

tx = -exp(1im *3π/4)
ty= -1
U = 0

#### Define Conditions for filling the many_body tensors ####


function fy_2body(sites_tuple)
    s1, s2 = sites_tuple
    return (helical(s1, s2, 2, geometry[2]) + neighbour(s1,s2, 2))* ty 
end

function fx_2body(sites_tuple)
    s1, s2 = sites_tuple
    return (periodic_neighbour(s1, s2, (true, false), geometry) + neighbour(s1, s2, 1))* tx 
end

function fxy_2body(sites_tuple)
    s1,s2 = sites_tuple
    return helical_periodic(s1,s2,geometry) * ty 
end

conditions = (fx_2body, fy_2body, fxy_2body)

t = fill_nbody_tensor(V, lattice, 2, conditions)

H = two_body_Op(V, lattice, t)
H_k = momentum_space_Op(H, lattice, (1,))

t_k = get_tensor_2body(H_k, lattice)

pl = plot(
    xlabel = "kₓ",           # replace "units" with physical units if any
    ylabel = "Eigenvalues ", 
    title = "Band Structure",
    legend = false,
    grid = true,
    framestyle = :box
);

for k in 1:geometry[1]
    H_k = t_k[k,:,k,:]
    es, vs = eigen(H_k)
    
    # Map k to central Brillouin zone [-π, π]
    kx = 2π * (k-1) / geometry[1]       # original BZ [0,2π)
    kx = kx > π ? kx - 2π : kx          # shift to [-π, π]

    # Compute localization on orbital i=1 (first row of eigenvectors)
    loc_i1 = abs2.(vs[1,:])  # weight of each eigenvector on orbital 1

    # Plot each eigenvalue with color indicating localization
    for α in eachindex(es)
        scatter!(pl, [kx], [real(es[α])],
                 marker = (:circle, 3, 0.8),      # size, alpha
                 markerstrokecolor = :black,
                 color = get(ColorSchemes.viridis, loc_i1[α]),
                 label = "")
    end
end

# Display plot
display(pl)





T = zeros(ComplexF64, geometry)

for s in lattice.sites
    for n in lattice.NN[s[1]]
        if (n[1] > s[1][1])  & (n[2]==s[1][2])
            H += FockOperator(((s[2], true) , (lattice.sites[n], false)), tx, V)
        elseif ((n[2] > s[1][2]) & (n[1] == s[1][1])) || ((s[1][2]==geometry[2]) & (n[2] == 1) & (s[1][1]==n[1]-1))
            H += FockOperator(((s[2], true) , (lattice.sites[n], false)), ty, V)
        end
    end
end
J = 1im*H
J -= dagger_FO(j)
H += dagger_FO(H)


M = calculate_matrix_elements_parallel(states,H)

es, vs = eigen(Hermitian(M))

for i in eachindex(es)
    s = create_MFS(vs[:,i], states)
    c = s * (J * s)
    dup = sum(density_onsite(s, lattice.sites, geometry)[:,2])
    ddown = sum(density_onsite(s, lattice.sites, geometry)[:,1])
    println("j: $c \n upperleg loc.: $dup \n down leg loc: $ddown")
end





pl = plot(legend=false,ylims=[-.1,.1]);

for ϕ in LinRange(0,2*π, 5)



H = ZeroFockOperator()

for s in lattice.sites
    for n in lattice.NN[s[1]]
        if (n[1] > s[1][1])  & (n[2]==s[1][2])
            H += FockOperator(((s[2], true) , (lattice.sites[n], false)), tx, V)
        elseif ((n[2] > s[1][2]) & (n[1] == s[1][1])) || ((s[1][2]==geometry[2]) & (n[2] == 1) & (s[1][1]==n[1]-1))
            H += FockOperator(((s[2], true) , (lattice.sites[n], false)), ty, V)
        end
    end
end
H += dagger_FO(H)


M = calculate_matrix_elements_parallel(states,H)

es, vs = eigen(Hermitian(M))


scatter!(pl, ϕ.*ones(length(es)), es , marker=:o, color=:red, markersize=1, markerstrokecolor=:red)
end;

display(pl)

i = findfirst(x->isapprox(x,0.; atol=01e-11), es)
gs = MultipleFockState(vs[:,i] .* states)
d = density_onsite(gs, lattice.sites, geometry)
display(heatmap(real.(d)))

