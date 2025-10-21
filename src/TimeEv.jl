############################################################
# Exact diagonalization evolution for Fock operators
############################################################

"""
    Time_Evolution_ed(Ops_dict::Dict, t0::Float64, t1::Float64, δt::Float64)

Performs exact diagonalization time evolution for a Hamiltonian of the form:

    H(t) = ∑ₖ fₖ(t) * Oₖ

where each `Oₖ` is a Fock operator (e.g., `MultipleFockOperator`), and 
`fₖ(t)` is either a constant (time-independent) or a time-dependent 
coefficient provided as an `Interpolation` object.

# Arguments
- `Ops_dict::Dict`: mapping operators `Oₖ` → coefficients `fₖ`  
    - If `fₖ` is a `Number`, the operator is treated as time-independent.  
    - If `fₖ` is an `Interpolation`, it is evaluated at each time step.
- `t0::Float64`: initial time  
- `t1::Float64`: final time  
- `δt::Float64`: time step for the evolution

# Returns
- `U::AbstractMatrix{ComplexF64}`: total time-evolution operator from `t0` to `t1`
"""
## !!! this function needs to be altered !!! ##
function Time_Evolution_ed(Ops_dict::Dict, t0::Float64, t1::Float64, δt::Float64)
    times = t0:δt:t1            # discretized time points
    U = I                        # initialize evolution operator

    # Loop over all time steps
    for t in times
        H = ZeroFockOperator()   # initialize Hamiltonian as zero operator

        # Construct Hamiltonian at this time step
        for O in keys(Ops_dict)
            coeff = Ops_dict[O]
            if isa(coeff, Number)
                H += coeff * O                # time-independent term
            else
                # assume coeff is an Interpolation object
                H += ComplexF64(coeff(t)) * O  # evaluate at current time
            end
        end

        # Diagonalize Hamiltonian and compute time-step evolution
        es, vs = eigen(H)
        U_step = vs * Diagonal(exp.(-im .* es .* δt)) * vs'
        U = U_step * U  # accumulate total evolution
    end

    return U
end

# ==========================================================
# Time evolution using DifferentialEquations.jl (TI Hamiltonian)
# ==========================================================
"""
    Time_Evolution(init, H, tspan; rtol, atol, solver)

Integrates the time-independent Schrödinger equation:

Arguments:
- `init`: initial state vector
- `H`: Hamiltonian matrix
- `tspan`: tuple (t0, t1)
- `rtol`, `atol`: solver tolerances
- `solver`: ODE solver algorithm (default: Vern7)

Returns:
- `sol`: solution object from DifferentialEquations.jl
"""
function Time_Evolution(init::Vector{ComplexF64}, H::AbstractMatrix{ComplexF64},
                        tspan::Tuple{Float64, Float64};
                        rtol::Float64 = 1e-9, atol::Float64 = 1e-9,
                        solver = Vern7())
    prob = ODEProblem(schrodinger!, init, tspan, H)
    sol = solve(prob, solver; reltol=rtol, abstol=atol, save_everystep=false, saveat=tpoints)
    return sol
end


# ==========================================================
# Time-independent RHS for ODEProblem
# ==========================================================
"""
    schrodinger!(dψ, ψ, H, t)

Time-independent Schrödinger equation RHS:

    dψ/dt = -i H ψ

Arguments:
- `dψ`: derivative vector to update (output)
- `ψ`: current state vector
- `H`: Hamiltonian matrix
- `t`: current time (ignored)
"""
function schrodinger!(dψ::Vector{ComplexF64}, ψ::Vector{ComplexF64},
                      H::AbstractMatrix{ComplexF64}, t::Float64)
    dψ .= -1im * (H * ψ)
    return nothing
end

# ==========================================================
# Time evolution using DifferentialEquations.jl (TD Hamiltonian)
# ==========================================================
"""
    Time_Evolution_TD(init, ops_and_interps, tspan; rtol, atol, solver)

Integrates the time-dependent Schrödinger equation:

Arguments:
- `init`: initial state vector
- `ops_and_interps`: tuple of (operator matrices, interpolation functions)
- `tspan`: tuple (t0, t1)
- `rtol`, `atol`: solver tolerances
- `solver`: ODE solver algorithm (default: Vern7)

Returns:
- `sol`: solution object from DifferentialEquations.jl
"""
function Time_Evolution_TD(init::Vector{ComplexF64},
                           ops_and_interps::Tuple{Vector{Matrix{ComplexF64}},
                                                  Vector{T}},
                           tspan::Tuple{Float64, Float64}, tpoints::NTuple{N, Float64};
                           rtol::Float64 = 1e-9, atol::Float64 = 1e-9,
                           solver = Vern7()) where {T,N}
    prob = ODEProblem(schrodinger_TD!, init, tspan, ops_and_interps)
    sol = solve(prob, solver; reltol=rtol, abstol=atol, save_everystep=false, saveat=tpoints)
    return sol
end

# ==========================================================
# Time-dependent RHS for ODEProblem (DifferentialEquations.jl)
# ==========================================================
"""
    schrodinger_TD!(dψ, ψ, ops_and_interps, t)

Time-dependent Schrödinger equation RHS:

    dψ/dt = -i ∑_k fₖ(t) * O_k * ψ

Arguments:
- `dψ`: derivative vector to update (output)
- `ψ`: current state vector
- `ops_and_interps`: tuple of (operator matrices, interpolation functions)
- `t`: current time
"""
function schrodinger_TD!(dψ::Vector{ComplexF64}, ψ::Vector{ComplexF64},
                         ops_and_interps::Tuple{Vector{<:AbstractMatrix{ComplexF64}},
                                                Vector{T}}, 
                         t::Float64) where T
    Ops, interps = ops_and_interps
    @inbounds for i in eachindex(dψ)
        dψ[i] = 0.0 + 0im   # reset derivative vector
    end
    @inbounds for k in eachindex(Ops)
        fk_t = ComplexF64(interps[k](t))   # evaluate coefficient at time t
        dψ .+= -1im * fk_t * (Ops[k] * ψ)
    end
    return nothing
end

function Heisenberg_eom(H::AbstractFockOperator, O::AbstractFockOperator)
    RHS = commutator(H, O)
    typeof(RHS) == FockOperator && (RHS = MultipleFockOperator([RHS], 0))
    return 1im * RHS
end

# ==========================================================
# Generate a string representing the mean field code for integration
# ==========================================================

function generate_eom_string_expl(eoms::Vector{MultipleFockOperator})
    eom_str = ""
    for (i, eom) in enumerate(eoms)
        eom_str *= "dψ[$i] .= "
        for op in eom.terms
            eom_str *= "$(op.coeff) * "
            for f in op.product
                eom_str *= f[2] ? "conj(ψ[$(f[1])]) * " : "ψ[$(f[1])] * "
            end
            eom_str = chop(eom_str, tail=3)
            eom_str *= " + "
        end
        eom_str = chop(eom_str, tail=3)
        eom_str *= " \n"
    end
    eom_str = chop(eom_str, tail=2)
    return eom_str
                  
end

function generate_eom_string_TO(eoms::Vector{MultipleFockOperator}, lattice::Lattice)
    eom_str = ""
    for (i, eom) in enumerate(eoms)
        eom_str *= "dψ[$i] .= "
        tensors = extract_nbody_tensors(eom, lattice)
        for t in tensors
            rank = length(size(t))
            if rank == 0 
                eom_str *= "$t"
            end
            t_index = join('a':'a'+(rank-1),",")
            indices = split(t_index, ",")
            eom_str *= a
        end

    end
    eom_str = chop(eom_str, tail=2)
    return eom_str
end



function mean_field_TE(init::Vector{ComplexF64},
                        H::MultipleFockOperator,
                        tspan::Tuple{Float64, Float64}, tpoints::NTuple{N, Float64};
                        rtol::Float64 = 1e-9, atol::Float64 = 1e-9,
                        solver = Vern7()) where {N}
    V = H.space
    nmodes = prod(collect(V.geometry))
    eoms = Vector{MultipleFockOperator}()
    println("constructing eom...")
    for i in 1:nmodes
        a_i = FockOperator(((i, false)), 0)
        push!(eoms, Heisenberg_eom(H, a_i))
    end

    EOM!(dψ, ψ, t) = (generate_eom_string(eoms) |> Meta.parse |> eval ; nothing)

    prob = ODEProblem(EOM!, init, tspan)
    sol = solve(prob, solver; reltol=rtol, abstol=atol, save_everystep=false, saveat=tpoints)
    return sol 
end



