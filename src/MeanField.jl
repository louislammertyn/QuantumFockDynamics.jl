struct MFMonomial
    coeff::ComplexF64
    conj_idx::Vector{Int}    # sites of ψ*
    idx::Vector{Int}         # sites of ψ
end
struct MF 
    expr::Vector{MFMonomial} 
end
struct MFeq
    rhs::Vector{MF}
end

function construct_MF(O::MultipleFockOperator)
    mf = Vector{MFMonomial}()
    for term in O.terms 
        conj_idx = Vector{Int}()
        idx = Vector{Int}()
        for (site, bool) in term.product
            if bool 
                push!(conj_idx, site)
            else
                push!(idx, site)
            end
        end
        push!(mf, MFMonomial(term.coefficient, conj_idx, idx))      
    end

    return MF(mf)
end

function construct_eoms(H::MultipleFockOperator)
    V = first(H.terms).space 
    nmodes = prod(V.geometry)
    eoms = Vector{MF}()
    for i in 1:nmodes 
        C = commutator(a(V, i), H)
        push!(eoms, construct_MF(C))
    end
    return MFeq(eoms)
end

function eval_MF(H::MultipleFockOperator, ψ::Vector{ComplexF64})
    mf_H = construct_MF(H)
    return eval_MF(mf_H, ψ)
end

function eval_MF(mf::MF, ψ::Vector{ComplexF64})
    result = 0. + 0im
    for monom in mf.expr
        c = monom.coeff
        for s_c in monom.conj_idx
            c *= conj(ψ[s_c])
        end
        for s in monom.idx 
            c *= ψ[s]
        end
        result += c
    end
    return result
end

function GPE!(du, u, (MFeq, im_time), t)
    sign = im_time ? -1. : -1.0im
    for (i,mf) in enumerate(MFeq.rhs) 
        du[i] = sign * eval_MF(mf, u)
    end
    nothing
end

##### Callbacks ####

function make_normalisation_cb( N::Int )
    return DiscreteCallback(
        (u, t, integrator) -> true,          # condition: always
        (integrator) -> begin
            norm = sqrt(sum(abs2.(integrator.u)))
            integrator.u ./= (norm / sqrt(N))
        end
    )
end

function make_convergence_cb(eoms::MFeq, tol::Float64=1e-10, check_every::Int=10, maxstep::Int=10000)
    μ_prev = Ref(0.0)
    step_count = Ref(0)

    condition = (u, t, integrator) -> begin
        step_count[] += 1
        step_count[] % check_every == 0
    end

    affect! = (integrator) -> begin
        u = integrator.u
        
        # chemical potential: μ = <ψ|H|ψ>
        μ = get_μ(eoms, u) 
        
        Δμ = abs(μ - μ_prev[])
        @info "step $(step_count[]) | μ=$μ | Δμ=$Δμ"
        
        if Δμ < tol
            @info "Converged! μ = $μ"
            terminate!(integrator)
        elseif maxstep < step_count[] * check_every
            @warn "No convergence in $(maxstep*check_every) steps | μ=$μ | Δμ=$Δμ \n Increase max steps or check model."
            terminate!(integrator)
        end

        μ_prev[] = μ
    end

    return DiscreteCallback(condition, affect!)
end

function get_mf_groundstate(ψ₀::Vector{ComplexF64}, H::MultipleFockOperator, N::Int=1; tol::Float64=1e-10, check_every::Int=20, maxstep::Int=100_000, dt::Float64=0.1, save_history=false)
    H_mf = construct_MF(H)
    eom = construct_eoms(H)
    return get_mf_groundstate(ψ₀, H_mf, eom, N; tol=tol, check_every=check_every, maxstep=maxstep, dt=dt, save_history= save_history)
end

function get_mf_groundstate(ψ₀::Vector{ComplexF64}, H_mf::MF, eoms::MFeq, N::Int=1; tol::Float64=1e-10, check_every::Int=20, maxstep::Int=100_000, dt::Float64=0.1, save_history=false)
    if !(norm(ψ₀)^2 ≈ N)
        @warn "Initial state not normalised to N=$N, normalising before proceeding"
        # normalize
        ψ₀ ./= (norm(ψ₀) / sqrt(N))
    end

    cb = CallbackSet(make_normalisation_cb(N), make_convergence_cb(eoms, tol, check_every, maxstep))
    prob = ODEProblem(GPE!, ψ₀, (0., Inf), (eoms, true), callback=cb, save_everystep=false)
    sol = solve(prob, Tsit5(), dtmax=dt)

    return (save_history) ? sol : sol.u[end]
end

function mf_time_evolution(ψ₀::Vector{ComplexF64}, H_mf::MF, eoms::MFeq, tmax::Float64, save_ts::Vector{Float64}, dt::Float64=0.1; atol::Float64=1e-8, rtol::Float64=1e-8, solver=Vern7())
    prob = ODEProblem(GPE!, ψ₀, (0, tmax), (H_mf, false))
    sol = solve(prob, solver, dtmax=dt; saveat=save_ts, save_everystep=false, atol=atol, rtol=rtol)
    return sol 
end

function get_μ(eoms::MFeq, ψ::Vector{ComplexF64})
    # μψᵢ = -i∂ψᵢ/∂t|_{stationary} = eval_MF(eomᵢ, ψ)
    Hψ = [eval_MF(mf, ψ) for mf in eoms.rhs]  # = H_mf * ψ in matrix sense
    return real(ψ' * Hψ) / real(ψ' * ψ)        # Rayleigh quotient → correct μ
end