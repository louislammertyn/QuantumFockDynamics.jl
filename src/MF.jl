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
        push!(polynomial, MFMonomial(term.coefficient, conj_idx, idx))      
    end

    return MF(mf)
end

function eval_MF(H::MultipleFockOperator, ψ::Vector{ComplexF64})
    mf_H = construct_MF(H)
    return eval_MF(mf_H, ψ)
end

function eval_MF(mf::MF, ψ)
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

function get_μ(H::MultipleFockOperator, ψ::Vector{ComplexF64})
    denom = ψ' * ψ
    numer = eval_MF(H, ψ)
    return numer / denom 
end


function get_μ(H_mf::MF, ψ::Vector{ComplexF64})
    denom = ψ' * ψ
    numer = eval_MF(H_mf, ψ)
    return numer / denom 
end