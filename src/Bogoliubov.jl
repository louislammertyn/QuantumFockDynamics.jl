"""
    This page implements Bogoliubov methods for Fock Operators which expect FockOperators of type 

        O = ∑ᵢⱼ 2hᵢⱼ a†ᵢaⱼ + Δᵢⱼ a†ᵢa†ⱼ +  Δᵢⱼ∗ aᵢaⱼ 

    Either the Hamiltonian is already of this form or one has a many body Hamiltonian for which one assumes the transformation

        aᵢ → αᵢ + bᵢ, αᵢ ∈ C and bᵢ satisfying ⟨bᵢ⟩ = 0 

    Concretely it assumes aᵢ → αᵢ + bᵢ where α ∈ C and b  is an operator satis
"""

struct BogoliubovRep 
    h::Matrix{ComplexF64}
    Δ::Matrix{ComplexF64} 
    ψ::Vector{ComplexF64}
    μ::Float64
    spectrum::Vector{ComplexF64}
    U_bog::Matrix{ComplexF64}
    V_bog::Matrix{ComplexF64}
end

function construct_BogoliubovRep(H::MultipleFockOperator, ψ::Vector{ComplexF64})
    @assert H - dagger_FO(H) == ZeroFockOperator() "Bogoliubov representation is only defined for self adjoint operators"
   
    H_mf = construct_MF(H)
    eoms = construct_eoms(H)
    μ = get_μ(eoms, ψ)

    V = first(H.terms).space
    geometry = V.geometry 
    h = zeros(ComplexF64, prod(geometry), prod(geometry))
    Δ = copy(h)

    for term in H.terms 
        op_string = term.product
        if length(op_string) == 2
            type = [x[2] for x in op_string]
            ids = [x[1] for x in op_string]
            if type == [true, false]
                h[ids...] += term.coefficient 
            elseif type ==[true, true]
                Δ[ids... ] += term.coefficient *2
            elseif type == [false, false]
                nothing  # aᵢaⱼ terms → contribute to Δ* block, determined by Δ via H = H†
            elseif type == [false, true]
                nothing  # aᵢaⱼ† terms → contribute to h* block, determined by h via H = H†
            end
            

        elseif length(op_string) > 2
            for c in combinations(collect(1:length(op_string)), 2)
                rem = setdiff(1:length(op_string), c)
                type = [op_string[id][2] for id in c]
                ids = [op_string[id][1] for id in c]

                if type == [false, false]
                    continue  # aᵢaⱼ terms → contribute to Δ* block, determined by Δ via H = H† (see later in loop)
                elseif type == [false, true]
                    continue  # aᵢaⱼ† terms → contribute to h* block, determined by h via H = H† (see later in loop)
                end

                coeff = term.coefficient 
                for i in rem 
                    site = op_string[i]
                    if site[2]
                        coeff *= ψ[site[1]]' 
                    else 
                        coeff *= ψ[site[1]]
                    end
                end
                if type == [true, false]
                    h[ids...] += coeff 
                elseif type ==[true, true]
                    Δ[ids... ] += coeff * 2
                end
            end
        end
    end

    Δ = (Δ + transpose(Δ)) / 2

    for i in axes(h, 1)
        h[i, i] -= μ
    end

    @assert norm(h - h') < 1e-10            "h must be Hermitian"

    return BogoliubovRep(h, Δ, ψ, μ, zeros(ComplexF64, size(h)[1] *2), zeros(ComplexF64, size(h)),zeros(ComplexF64, size(h)))
end

function construct_BogoliubovRep(H::MultipleFockOperator, N::Int=1)
    H_mf = construct_MF(H)
    eoms = construct_eoms(H)
    V = first(H.terms).space 
    modes = prod(V.geometry)
    ψ_init = rand(ComplexF64, modes)
    ψ_init ./= (norm(ψ_init) / sqrt(N))
    ψ = get_mf_groundstate(ψ_init, H_mf, eoms, N)
    return construct_BogoliubovRep(H, ψ)
end

function Bogoliubov_spectrum(H_B::BogoliubovRep)
    
    A = H_B.h 
    B = H_B.Δ
    ψ = H_B.ψ
    D = size(A)[1]

    res = eigen(vcat(hcat(A, B), hcat(-conj.(B), -conj.(A))))

    # --- filter and normalize zero modes ---
    threshold = 1e-5
    nonzero_idx = findall(i -> abs(res.values[i]) > threshold, 1:2D)
    zero_idx    = setdiff(1:2D, nonzero_idx)
    n_zero      = length(zero_idx) ÷ 2
    D_red       = D - n_zero
   

    # normalize zero mode vectors to analytic form (ψ, -ψ*) / norm
    zero_vecs = zeros(ComplexF64, 2D, 2*n_zero)
    for (k, i) in enumerate(zero_idx[1:n_zero])
        u = res.vectors[1:D, i]
        c = norm(u)
        zero_vecs[1:D,     k] .= ψ ./ norm(ψ)
        zero_vecs[D+1:end, k] .= -conj.(ψ) ./ norm(ψ)
        @info "Zero mode $k: removed scale factor |c|=$(c)"
    end
    for (k, i) in enumerate(zero_idx[n_zero+1:end])
        zero_vecs[1:D,     n_zero+k] .=  conj.(ψ) ./ norm(ψ)
        zero_vecs[D+1:end, n_zero+k] .= -ψ ./ norm(ψ)
    end

    vals = res.values[nonzero_idx]
    vecs = res.vectors[:, nonzero_idx]
    # --------------------------

    vals, vecs = sort_bg(vals, vecs, D_red)
    vals, vecs = symplectic_orthogonalise(vals, vecs)

    for (i, es) in enumerate(vals)
        if !isapprox(imag(es), 0; atol=1e-8)
            @warn "nonzero imaginary part with energy $es"
        end
    end
    
    neg_norm = []
    @views for i in 1:2*D_red
        del = (norm(vecs[1:D,i])^2 - norm(vecs[D+1:2*D,i])^2) * (i > D_red ? -1 : 1)
        if del < 0
            push!(neg_norm, i)
            @warn "vector $i gives negative norm with energy $(vals[i])"
            del *= -1
        end
        @assert del > 0 "The diagonalisation is not compatible with a bogoliubov transform for vector $i the energy is given by $(vals[i]) for norm $del"        
        vecs[:,i] .*= 1 / sqrt(del)  
    end
    
    for i in neg_norm
        if i > D_red
            break
        end
        vecs[:, i], vecs[:, i+D_red] = copy(vecs[:, i+D_red]), copy(vecs[:, i])
        vals[i], vals[i+D_red] = vals[i+D_red], vals[i]
    end

    @assert isapprox(vecs' * J(D) * vecs, J(D_red); atol=sqrt(D_red) * eps() * 1e4) "The symplectic condition is not satisfied by this diagonalisation"
    # --- add zero modes back ---
    # full vecs: [zero_pos | nonzero_pos | zero_neg | nonzero_neg]
    full_vecs = hcat(
        zero_vecs[:, 1:n_zero],          # positive zero modes
        vecs[:, 1:D_red],                # positive nonzero modes
        zero_vecs[:, n_zero+1:end],      # negative zero modes
        vecs[:, D_red+1:end]             # negative nonzero modes
    )

    full_vals = vcat(
        zeros(ComplexF64, n_zero),
        vals[1:D_red],
        zeros(ComplexF64, n_zero),
        vals[D_red+1:end]
    )

    H_B.spectrum .= full_vals
    H_B.U_bog .= full_vecs[1:D,    1:D]
    H_B.V_bog .= full_vecs[D+1:end, 1:D]

    return H_B
end


function J(D::Int)
    I_n = diagm(ones(D))  # Create n x n identity matrix
    j = [I_n zeros(D,D); zeros(D,D) -I_n]
    return j 
end


function sort_bg(ϵ, ψ, n)
    ψ = hcat(ψ[:,n+1:end], ψ[:,1:n])
    ϵ = vcat(ϵ[n+1:end], ϵ[1:n])
    
    ψ_ = similar(ψ)
    ϵ_ = similar(ϵ)
    matched = falses(2n)  # track which j indices have been used

    ψ_[:, 1:n] = ψ[:, 1:n]
    ϵ_[1:n] = ϵ[1:n]

    for i in 1:n
        best_j = -1
        best_err = Inf
        for j in n+1:2n
            if matched[j]
                continue
            end
            # check symplectic partner condition
            
            err = abs(ϵ[i] + conj(ϵ[j]))
            if err < best_err
                best_err = err
                best_j = j
            end
        
        end
        if best_j == -1 || best_err > 1e-6
            error("No symplectic partner found for mode $i, best error $best_err")
        end
        matched[best_j] = true
        ψ_[:, i+n] = ψ[:, best_j]
        ϵ_[i+n] = ϵ[best_j]
    end
    return ϵ_, ψ_
end

function group_degenerate_blocks(ϵ, ψ)
    deg_blocks = Vector{Matrix{ComplexF64}}()
    block = false
    block_m = [ψ[:,1]]
    for i in 2:length(ϵ)
        if  isapprox(ϵ[i], ϵ[i-1]; atol=1e-8)
            block = true 
        else 
            block=false 
        end

        if !block
            block_m = hcat(block_m...)
            push!(deg_blocks, block_m)
            block_m = []
        end
        push!(block_m, ψ[:,i])
    end
    block_m = hcat(block_m...)
    push!(deg_blocks, block_m)

    return deg_blocks
end


function orthogonalise_degenerate_blocks!(deg_blocks::Vector)
    for (i,deg_b) in enumerate(deg_blocks)
        D = div(size(deg_b)[1] ,2)
        S = deg_b' * J(D) * deg_b 
        es, vs = eigen(Hermitian(S))
        for e in es 
            if isapprox(e, 0)
                @warn "?"
            end
        end
        deg_blocks[i] =  deg_b * vs
    end
    return deg_blocks
end

function symplectic_orthogonalise(ϵ, ψ)
    deg_blocks = group_degenerate_blocks(ϵ, ψ)
    deg_blocks = orthogonalise_degenerate_blocks!(deg_blocks)
    ψ = hcat(deg_blocks...)
    return ϵ, ψ 
end



function plot_Bogoliubov_spectrum(res, N, xs=-15.0:0.1:15.0, bdg_idx=1:3, howmany=3)
    λmax = length(res.spectrum) ÷ 2 - 1
    hs = hermite.(0:λmax, xs')
    omegas = real.(res.spectrum)[1:(isnothing(howmany) ? end : howmany)] ./ N

    spectrum = scatter(omegas, zeros(length(omegas)), ylims=[-0.25, 0.25], c=1, lab="")
    vline!(omegas, ls=:dash, c=1, lw=1.5, lab="")
    hline!([0], c=:black, ls=:solid, lab="")
    plot!(framestyle=:box, legend=:topright, xlabel="|Eᵦ|/ħω" ) #xticks=floor(Int, minimum(omegas)):ceil(Int, maximum(omegas))

    #mf = plot(xs, vec(abs2.(sum(res.gs .* hs, dims=1))), framestyle=:box, lab=L"\Psi_0(x)", lw=2)

    #us = [plot(xs, vec(abs2.(sum(res.u[:, i] .* hs, dims=1))), framestyle=:box, lab=latexstring("u_$i(x)"), lw=1.5) for i in bdg_idx]
    #vs = [plot(xs, vec(abs2.(sum(res.v[:, i] .* hs, dims=1))), framestyle=:box, lab=latexstring("v_$i(x)"), lw=1.5) for i in bdg_idx]
    #l = @layout [
        #a{0.2h}; [grid(length(bdg_idx), 1) b{0.5w} grid(length(bdg_idx), 1)]
    #]

    pl = plot(spectrum,size=(1000, 500), fg_legend=false)
    display(pl)
end

function Bogoliubov_groundstate(H::BogoliubovRep; third=false)

    res = bogoliubov_spectrum(H)

    # Set appropriate values A = ⟨a†a⟩, B = ⟨aa⟩ and T=⟨a†aa⟩
    α = H_B.ψ 
    A = res.v' * res.v 
    B = -1 .* (res.u' * res.v)
    @assert isapprox(A , A') "A is not hermitian"
    @assert isapprox(B , Transpose(B)) "B is not symmetric"

    if third
        T = zeros(ComplexF64, params.λmax + 1, params.λmax + 1, params.λmax + 1)
        return [α, A, B, T], res.spectrum
    else
        return [α, A, B], res.spectrum
    end
end
