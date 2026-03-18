using Revise
using FoSpDynamics
using LinearAlgebra
using Plots
using ProgressMeter





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
    states = all_states_U1(V)

    latt = Lattice(geometry)

    Kin, Int = Bose_Hubbard_H(V, latt, J, U)

    M_s = calculate_matrix_elements(Kin + Int, states)
    @assert M_s == M_s'
    M = Matrix(M_s)
    x₀ = rand(ComplexF64, size(M_s)[1])
    es, vs =  eigen(M);

    gs_coeff = vs[:,1]
    pd_gap[(n),ju] =  (es[2]- es[1]) / E_scale
    gs = create_MFS(gs_coeff, states)

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


############## Bogoliubov analysis #############

U = 1.0
J_values = 0.1 :0.1:2.0  # sweep J/U
L_bog = 5
N_bog = 5

geometry = (L_bog,)
latt= Lattice(geometry_bog)
V= U1FockSpace(geometry_bog, N_bog, N_bog)

spectra = []
depletions = []

@showprogress for J in J_values
    Kin, Int = Bose_Hubbard_H(V_bog, latt_bog, J, U)
    H = Kin + Int
    
    bog_rep = construct_BogoliubovRep(H, N_bog)
    res = Bogoliubov_spectrum(bog_rep)

    # only keep positive energy modes
    ω = real.(res.spectrum[1:end÷2])
    depletion = sum(abs2, res.V) / N_bog

    push!(spectra, ω)
    push!(depletions, depletion)
end

# plot lowest few quasiparticle energies vs J/U
n_modes = 5
p1 = plot(
    xlabel = "J/U",
    ylabel = "ω/U",
    title  = "Bogoliubov spectrum vs J/U",
    framestyle = :box,
    legend = :topleft
)
for mode in 1:n_modes
    plot!(p1,
        collect(J_values),
        [s[mode] for s in spectra],
        label = "mode $mode",
        lw = 2
    )
end

# check Goldstone mode — should be zero for all J
goldstone = [s[1] for s in spectra]
plot!(p1, collect(J_values), goldstone, label="Goldstone", lw=2, ls=:dash, c=:black)

# plot depletion vs J/U — sanity check for Bogoliubov validity
p2 = plot(
    collect(J_values), depletions,
    xlabel = "J/U",
    ylabel = "depletion / N",
    title  = "Condensate depletion",
    framestyle = :box,
    lw = 2,
    label = "",
    ylims = (0, 1)
)
hline!(p2, [0.1], ls=:dash, c=:red, label="10% threshold")

# compare Bogoliubov ground state energy to exact diagonalization
# use small system where ED is feasible
E_errors = []
for J in J_values
    L_ed = L_bog
    N_ed = N_bog
    J_ed = J
    V_ed = U1FockSpace((L_ed,), N_ed, N_ed)
    latt_ed = Lattice((L_ed,))
    Kin_ed, Int_ed = Bose_Hubbard_H(V_ed, latt_ed, J_ed, U)
    H_ed = Kin_ed + Int_ed

    # ED ground state energy
    states_ed = all_states_U1(V_ed)
    M_ed = Matrix(calculate_matrix_elements(H_ed, states_ed))
    es_ed, _ = eigen(M_ed)
    E_ed = es_ed[1]

    # Bogoliubov ground state energy
    # E_bog = E_MF + (1/2) * sum of quasiparticle energies - (1/2) Tr(h)
    bog_rep_ed = construct_BogoliubovRep(H_ed, N_bog)
    res_ed = Bogoliubov_spectrum(bog_rep_ed)
    ω_ed = real.(res_ed.spectrum[1:end÷2])
    E_mf = real(eval_MF(H_ed, bog_rep_ed.ψ))
    E_bog = E_mf+ 0.5 * sum(ω_ed) - 0.5 * tr(bog_rep_ed.h)

    println("ED  ground state energy:         $E_ed")
    println("Bog ground state energy:         $E_bog")
    println("Relative error:                  $(abs(E_bog - E_ed) / abs(E_ed))")
    println("Depletion at J/U = $J_ed:        $(depletions[findfirst(J_values .≈ J_ed)])")
    push!(E_errors, abs(E_bog - E_ed) / (abs(E_ed)))
end

p3 = plot((J_values ./ U)[5:end], E_errors[5:end], xlabel="J/U", ylabel="Relative error |E_bog - E_ed| / |E_ed|",title  = "Relative error on gs energy",
    framestyle = :box,
    lw = 2,
    label = "",
    color=:red,
    linestyle=:dashdot
    )

display(plot(p1, p2, p3, layout=(1,3), size=(2200,700)))


###### Bogoliubov in Frequency space ######

U = 1.
J = 4. 
gaps = []
Ls =  [ 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

for l in Ls
    L = l
    N = div(l, 2)

    geometry = (L,)
    latt = Lattice(geometry; periodic=(true,))
    V = U1FockSpace(geometry, N, N)

    K, Int = Bose_Hubbard_H(V, latt, J, U)
    H = K + Int
    Hk = transform(H, latt, bloch_matrix(L))
    ks = [m * 2π / L for m in 0:L-1]

    H_k_bog = construct_BogoliubovRep(Hk, N)
    H_k_bog = Bogoliubov_spectrum(H_k_bog)
    push!(gaps, real.(H_k_bog.spectrum[2]))

    if l == 20
        E_bog(k, J, U, n0) = sqrt(2J * (1 - cos(k)) * (2J * (1 - cos(k)) + 2U * n0))
        k_sorted = sort(fold.(ks))

        pl_disp = plot(
            k_sorted, E_bog.(k_sorted, J, U, N / L) / J;
            color = :steelblue,
            linewidth = 2,
            label = "analytical",
            xlabel = "k",
            ylabel = "E(k) (ħJ)",
            title = "Bogoliubov Dispersion  (L=$l, J/U=$(J/U))",
            framestyle = :box,
            xticks = ([0, π/2, π], ["0", "π/2", "π"]),
            xlims = (0, π),
            ylims = (0, Inf),
            legend = :topleft,
            grid = true,
            gridalpha = 0.3,
            size = (600, 400),
            dpi = 150,
        )

        for (i, e) in enumerate(H_k_bog.spectrum[1:l])
            label = i == 1 ? "numerical" : ""
            k = ks[argmax(abs2.(H_k_bog.U_bog[:, i]))]
            scatter!(pl_disp, [fold(k)], [real.(e)/J];
                color = :crimson,
                marker = :circle,
                markersize = 6,
                markerstrokewidth = 0,
                label = label,
            )
        end
        display(pl_disp)
    end
end

# --- Gaps plot ---
pl_gaps = plot(
    Ls, gaps;
    marker = :circle,
    markersize = 6,
    markerstrokewidth = 0,
    color = :steelblue,
    linewidth = 2,
    xlabel = "L",
    ylabel = "Gap  Δ",
    title = "Bogoliubov Gap vs System Size  (J/U=$(J/U))",
    framestyle = :box,
    legend = false,
    grid = true,
    gridalpha = 0.3,
    xticks = Ls,
    ylims = (0, Inf),
    size = (600, 400),
    dpi = 150,
);
display(pl_gaps)

