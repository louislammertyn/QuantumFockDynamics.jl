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
end

function construct_BogoliubovRep(H::MultipleFockOperator, ψ::Vector{ComplexF64})
    @assert H - dagger_FO(H) == ZeroFockOperator() "Bogoliubov representation is only defined for self adjoint operators"
   
     H_mf = construct_MF(H)
    μ = get_μ(H_mf, ψ)

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
                h[ids...] += term.coefficient / 2
            elseif type ==[true, true]
                Δ[ids... ] += term.coefficient 
            elseif type == [false, false]
                nothing  # aᵢaⱼ terms → contribute to Δ* block, determined by Δ via H = H†
            elseif type == [false, true]
                nothing  # aᵢaⱼ† terms → contribute to h* block, determined by h via H = H†
            end
            

        elseif length(op_string) > 2
            for c in combinations(1:length(op_string), 2)
                rem = setdiff(1:length(op_string), c)
                type = [op_string[id][2] for id in c]
                ids = [op_string[id][1] for id in c]

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
                    h[ids...] += coeff / 2
                elseif type ==[true, true]
                    Δ[ids... ] += coeff 
            
                elseif type == [false, false]
                    nothing  # aᵢaⱼ terms → contribute to Δ* block, determined by Δ via H = H†
                elseif type == [false, true]
                    nothing  # aᵢaⱼ† terms → contribute to h* block, determined by h via H = H†
                end
            end
        end
    end

    Δ = (Δ + Transpose(Δ)) / 2

    for i in axes(h, 1)
        h[i, i] -= μ
    end

    @assert norm(h - h') < 1e-10            "h must be Hermitian"
    @assert norm(Δ - transpose(Δ)) < 1e-10  "Δ must be symmetric"

            
    return BogoliubovRep(h, Δ, ψ, μ)
end

function construct_BogoliubovRep(H::MultipleFockOperator)
    H_mf = construct_MF(H)
    ψ = get_mf_groundstate(H_mf)
    return construct_BogoliubovRep(H, ψ)
end


function bogoliubov_spectrum(BogoliubovRep::MultipleFockOperator, ψ::Vector{ComplexF64}, μ::Float64)
    (; g, U, λmax, frame) = params
    n = (λmax+1)
    #U = npzread(joinpath(root, "data", "precompute", "hermite.npy"))[1:n, 1:n, 1:n, 1:n]
    h = zeros(ComplexF64, aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa)
    @tullio A[i, j] := g * U[i, k, l, j] * psi[k]' * psi[l]
    @tullio A[i, j] += g * U[i, k, j, l] * psi[k]' * psi[l]
    @tullio A[i, j] += g * U[k, i, l, j] * psi[k]' * psi[l]
    @tullio A[i, j] += g * U[k, i, j, l] * psi[k]' * psi[l]
    @tullio B[i, j] := g * U[i, j, k, l] * psi[k] * psi[l]

    @assert isapprox(A, adjoint(A)) "The interaction term is not hermitian"
    mu = μ * diagm(ones(n))
    if frame == RotatingFrame
        
        ham = Ham_Generator(params, 0) - mu
        A += (ham)
        A .*= 1/2
        @assert isapprox(A, adjoint(A))  "the hamiltonian is not Hermitian"
        #println(ham)
    else
        A += diagm((1:(params.λmax+1)).- 1) - mu
    end
    
    #sort the vectors and eigenvalues such that the symplectic structure satisfied
    if g != 0
        res = eigen(vcat(hcat(A, B), hcat(-conj.(B), -conj.(A))))
        vals, vecs = sort_bg(res.values, res.vectors, params.λmax+1)
    else 
        res = eigen(A)
        res2 = eigen(-conj.(A))
        vectors = zeros(ComplexF64, 2*n,2*n)
        vectors[1:n, 1:n] = res.vectors 
        vectors[n+1:2*n,n+1:2*n ] = res2.vectors
        vals, vecs = vcat(res.values, res2.values) , vectors
    end
    for es in vals
        if !iszero(imag(es))
            @warn "nonzero imaginary part with energy $es"
        end
    end
    
    # Normalisation of the eigenvectors such that the symplectic condition is satisfied
    neg_norm = []
    @views for i in 1:2*n 
        del = (norm(vecs[1:n,i])^2 - norm(vecs[n+1:2*n,i])^2) * (i>n ? -1 : 1)
        if del < 0
            push!(neg_norm, i)
            @warn "vector $i gives negative norm with energy $(vals[i])"
            del *= -1
        end
        @assert del > 0 "The diagonalisation is not compatible with a bogoliubov transform for vector $i the energy is given by $(vals[i]) for norm $del"        
        vecs[:,i] .*= 1/ sqrt(del)  
        vals[i] *= 1/ sqrt(del) 
        
    end
    
    # in case of negative energy modes, rearrange again to ensure symplectic condition
    for i in neg_norm
        if i > n 
            break
        end
        vecs[:, i], vecs[:, i+n] = copy(vecs[:, i+n]), copy(vecs[:, i])
        vals[i], vals[i+n] = vals[i+n], vals[i]
    end

    #check symplectic condition
    I_n = diagm(ones(params.λmax+1))  # Create n x n identity matrix
    J = [I_n zeros(params.λmax+1,params.λmax+1); zeros(params.λmax+1,params.λmax+1) -I_n]
    
    @assert isapprox(vecs * J * vecs', J) "The symplectic condition is not satisfied by this diagonalisation"
    

    return (
        spectrum = vals,
        u=vecs[1:n, 1:n],
        v=vecs[1:n, n+1:end],
        vec_matrix = vecs,
        gs=psi
    )
end

function plot_spectrum(res, N, xs=-15.0:0.1:15.0, bdg_idx=1:3, howmany=3)
    λmax = length(res.spectrum) ÷ 2 - 1
    hs = hermite.(0:λmax, xs')
    omegas = real.(res.spectrum)[1:(isnothing(howmany) ? end : howmany)] ./ N

    spectrum = scatter(omegas, zeros(length(omegas)), ylims=[-0.25, 0.25], c=1, lab="")
    vline!(omegas, ls=:dash, c=1, lw=1.5, lab="")
    hline!([0], c=:black, ls=:solid, lab="")
    plot!(framestyle=:box, legend=:topright, xlabel=L"|E_{BdG}|/\hbar\omega" ) #xticks=floor(Int, minimum(omegas)):ceil(Int, maximum(omegas))

    #mf = plot(xs, vec(abs2.(sum(res.gs .* hs, dims=1))), framestyle=:box, lab=L"\Psi_0(x)", lw=2)

    #us = [plot(xs, vec(abs2.(sum(res.u[:, i] .* hs, dims=1))), framestyle=:box, lab=latexstring("u_$i(x)"), lw=1.5) for i in bdg_idx]
    #vs = [plot(xs, vec(abs2.(sum(res.v[:, i] .* hs, dims=1))), framestyle=:box, lab=latexstring("v_$i(x)"), lw=1.5) for i in bdg_idx]

    #l = @layout [
        #a{0.2h}; [grid(length(bdg_idx), 1) b{0.5w} grid(length(bdg_idx), 1)]
    #]

    pl = plot(spectrum,size=(1000, 500), fg_legend=false)
    display(pl)
end



function sort_bg(ϵ, ψ, n)
    #function that sorts the eigenvectors and eigenvalues according to (u, v; v* u*) and sorted from small to large Re(eigenvalues)
    #note that in case of negative energy modes these will be in the wrong half of the columns, correction is done when normalising
    ψ = hcat(ψ[:,n+1:end],ψ[:,1:n] )
    ϵ = vcat(ϵ[n+1:end],ϵ[1:n] )
    ψ_ = similar(ψ)
    ϵ_ = similar(ϵ)

    ψ_[:, 1:n] = ψ[:,1:n]
    ϵ_[1:n] = ϵ[1:n]
    for i in 1:n , j in n+1:2n
        if isapprox(real(ϵ[i]), 0 ,atol=1e-5) && isapprox(real(ϵ[j]), 0, atol=1e-5)
           
            ψ_[:,i+n]= ψ[:,j]
            ϵ_[i+n] = ϵ[j]
        end
        if isapprox(ϵ[i],-conj(ϵ[ j] ), atol=1e-5)
            
            ψ_[:,i+n]= ψ[:,j]
            ϵ_[i+n] = ϵ[j]
            
        end

    end
    return ϵ_, ψ_
end

function BG_groundstate(params, hs, xs; Floquet=false)
    if Floquet
        gs_params = initialize_params(frame=RotatingFrame,
            Δ = params.Δ, 
            κ = params.κ, 
            g = params.g, 
            ϕ = params.ϕ, 
            pot = params.pot, 
            neighbours = params.neighbours,
            λmax = params.λmax,
            num_particles=params.num_particles
            )
    else
        gs_params = initialize_params(frame=params.frame,
                Δ = params.Δ, 
                κ = 0, 
                g = params.g, 
                ϕ = params.ϕ, 
                pot = params.pot, 
                neighbours = params.neighbours,
                λmax = params.λmax,
                num_particles=params.num_particles
                )
    end
    ψᵢ = psi_sloped(gs_params)
    ψ₀= get_ground_state(ψᵢ,gs_params)
    μ = get_mu(ψ₀, gs_params)
    res = bogoliubov_spectrum(μ,gs_params, ψ₀, hs)
    #plot_spectrum(res, params.num_particles, xs,1:3)

    ##### initialise the cumulants ######
    α = ψ₀ 
    A = res.v' * res.v 
    B = -1 .* (res.u' * res.v)
    @assert isapprox(A , A') "A is not hermitian"
    @assert isapprox(B , Transpose(B)) "B is not symmetric"
    T = zeros(ComplexF64, params.λmax + 1, params.λmax + 1, params.λmax + 1)

    return [α, A, B, T], res.spectrum
end
