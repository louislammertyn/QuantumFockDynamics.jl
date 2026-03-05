

struct MFCumulantMonomial{N}
    coeff::ComplexF64
    indices::NTuple{N, Int}      # sites involved
    types::NTuple{N, Symbol}        # :ψ, :ψd, :n, :b, :Tψψψ, :Tψdψψ
end

struct MFCumulantEq

end



function construct_MFeom(H::MultipleFockOperator)
    V = H.terms[1].space
    N = prod(V.geometry)
    rhs = Vector{MF}(undef, N)
    for i in 1:N
        rhs_i = commutator(H, a(V, i))
        polynomial = MFMonomial[]
        for term in rhs_i.terms
            conj_idx = [site for (site, is_dag) in term.product if is_dag]
            idx      = [site for (site, is_dag) in term.product if !is_dag]
            push!(polynomial, MFMonomial(term.coefficient, conj_idx, idx))
        end
        rhs[i] = MF(polynomial)
    end
    return MFeq(rhs)
end

function eval_MFeom!(dψ, ψ, mfeom::MFeq, imtime::Bool=false)
    for (i,eom) in enumerate(mfeom.rhs)
        result = eval_MF(eom, ψ)
        dψ[i] = imtime ? -1 * result :  -1im * result
    end
    
    nothing
end

function make_E_convergence_cb(eoms::MFeq; tol=1e-8)
    prev_E = Ref(Inf)
    dψ_buf = zeros(ComplexF64, length(eoms.rhs))

    condition = (u, t, integrator) -> begin
        eval_MFeom!(dψ_buf, u, eoms, true)
        E = real(dot(u, dψ_buf))
        converged = abs(E - prev_E[]) < tol
        prev_E[] = E
        return converged
    end

    return DiscreteCallback(condition, 
                            integrator -> terminate!(integrator),
                            save_positions=(false, false))
end

function get_mf_groundstate(eoms::MFeq, ψ_init::Vector{ComplexF64};
                             tol=1e-8, maxτ=1e4)
    renorm_cb = FunctionCallingCallback(
        (u, t, integrator) -> (u ./= norm(u)),
        func_everystep=true, func_start=false
    )
    conv_cb = make_E_convergence_cb(eoms; tol=tol)

    prob = ODEProblem(
        (dψ, ψ, p, t) -> eval_MFeom!(dψ, ψ, p, true),
        ψ_init ./ norm(ψ_init),
        (0.0, maxτ),
        eoms
    )

    sol = solve(prob, Tsit5(), callback=CallbackSet(renorm_cb, conv_cb),
                save_everystep=false, abstol=tol/10, reltol=tol/10)

    return sol.u[end]
end
