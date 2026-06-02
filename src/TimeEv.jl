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

"""
    schrodinger!(dψ, ψ, H, t)

In-place RHS for the time-independent Schrödinger equation:

    dψ/dt = -i H ψ

# Arguments
- `dψ::Vector{ComplexF64}`: output derivative vector.
- `ψ::Vector{ComplexF64}`: current state vector.
- `H::AbstractMatrix{ComplexF64}`: Hamiltonian matrix.
- `t::Float64`: current time (unused).
"""
function schrodinger!(dψ::Vector{ComplexF64}, ψ::Vector{ComplexF64},
                      H::AbstractMatrix{ComplexF64}, t::Float64)
    mul!(dψ, H, ψ)
    dψ .*= -1im 
    return nothing
end

"""
    von_neumann!(dρ_vec, ρ_vec, (tmp, H), t)

In-place RHS for the time-independent von Neumann equation (vectorized density matrix):

    dρ/dt = -i [H, ρ]

# Arguments
- `dρ_vec`: output derivative (flattened `N²` vector).
- `ρ_vec`: current density matrix (flattened `N²` vector).
- `(tmp, H)`: parameter tuple — `N×N` scratch buffer, Hamiltonian matrix.
- `t::Float64`: current time (unused).
"""
function von_neumann!(dρ_vec, ρ_vec, (tmp, H), t)
    N = size(H, 1)
    ρ  = reshape(ρ_vec, N, N)
    dρ = reshape(dρ_vec, N, N)
    fill!(dρ_vec, 0.0 + 0.0im)

    mul!(tmp, H, ρ)           # tmp = H*ρ
    axpy!(-1im, tmp, dρ)      # dρ += -i * H*ρ

    mul!(tmp, ρ, H)           # tmp = ρ*H
    axpy!(1im, tmp, dρ)       # dρ -= -i * ρ*H
    
    return nothing
end

"""
    Time_Evolution(init, H, tspan; rtol, atol, solver) -> sol

Integrate the time-independent Schrödinger equation.

# Arguments
- `init::Vector{ComplexF64}`: initial state vector.
- `H::AbstractMatrix{ComplexF64}`: Hamiltonian matrix.
- `tspan::Tuple{Float64,Float64}`: `(t0, t1)` integration interval.
- `rtol`, `atol`: solver tolerances (default: `1e-9`).
- `solver`: ODE algorithm (default: `Vern7()`).

# Returns
- `sol`: solution object from `DifferentialEquations.jl`.
"""
function Time_Evolution(init::Vector{ComplexF64}, H::AbstractMatrix{ComplexF64},
                        tspan::Tuple{Float64, Float64};
                        rtol::Float64 = 1e-9, atol::Float64 = 1e-9,
                        solver = Vern7())
    prob = ODEProblem(schrodinger!, init, tspan, H)
    sol = solve(prob, solver; reltol=rtol, abstol=atol, save_everystep=false, saveat=tpoints)
    return sol
end

"""
    Time_Evolution(init, H, basis, tspan; rtol, atol, solver) -> sol

Integrate the time-independent Schrödinger equation using a `MatrixFreeOperator`
representation of a `MultipleFockOperator` Hamiltonian.

# Arguments
- `init::Vector{ComplexF64}`: initial state vector.
- `H::MultipleFockOperator`: Hamiltonian as a Fock operator.
- `basis::Vector{AbstractFockState}`: basis to build the transition representation.
- `tspan::Tuple{Float64,Float64}`: `(t0, t1)` integration interval.
- `rtol`, `atol`: solver tolerances (default: `1e-9`).
- `solver`: ODE algorithm (default: `Vern7()`).

# Returns
- `sol`: solution object from `DifferentialEquations.jl`.
"""
function Time_Evolution(init::Vector{ComplexF64}, H::MultipleFockOperator,
                        basis::Vector{AbstractFockState},
                        tspan::Tuple{Float64, Float64};
                        rtol::Float64 = 1e-9, atol::Float64 = 1e-9,
                        solver = Vern7())
    H_mfo = transition_representation(H, basis)
    return Time_Evolution(init, H_mfo, tspan; rtol=rtol, atol=atol, solver=solver)
end

"""
    Time_Evolution_VN(init, H, tspan, tpoints; rtol, atol, solver) -> sol

Integrate the time-independent von Neumann equation for a density matrix:

    dρ/dt = -i [H, ρ]

# Arguments
- `init::Matrix{ComplexF64}`: initial density matrix.
- `H::Matrix{ComplexF64}`: Hamiltonian matrix (dense, required for `ρ*H`).
- `tspan::Tuple{Float64,Float64}`: `(t0, t1)` integration interval.
- `tpoints::NTuple{M,Float64}`: times at which to save the solution.
- `rtol`, `atol`: solver tolerances (default: `1e-9`).
- `solver`: ODE algorithm (default: `Vern7()`).

# Returns
- `sol`: solution object from `DifferentialEquations.jl`.
"""
function Time_Evolution_VN(init::Matrix{ComplexF64}, H::Matrix{ComplexF64},
                           tspan::Tuple{Float64, Float64}, tpoints::NTuple{M, Float64};
                           rtol::Float64 = 1e-9, atol::Float64 = 1e-9,
                           solver = Vern7()) where {M}
    tmp = similar(init)
    prob = ODEProblem(von_neumann!, vec(init), tspan, (tmp, H))
    sol = solve(prob, solver; reltol=rtol, abstol=atol, save_everystep=false, saveat=collect(tpoints))
    return sol
end

"""
    Time_Evolution_VN(init, H, basis, tspan, tpoints; rtol, atol, solver) -> sol

Integrate the time-independent von Neumann equation using the dense matrix
representation of a `MultipleFockOperator` Hamiltonian.

# Arguments
- `init::Matrix{ComplexF64}`: initial density matrix.
- `H::MultipleFockOperator`: Hamiltonian as a Fock operator.
- `basis::Vector{AbstractFockState}`: basis to build the matrix representation.
- `tspan::Tuple{Float64,Float64}`: `(t0, t1)` integration interval.
- `tpoints::NTuple{M,Float64}`: times at which to save the solution.
- `rtol`, `atol`: solver tolerances (default: `1e-9`).
- `solver`: ODE algorithm (default: `Vern7()`).

# Returns
- `sol`: solution object from `DifferentialEquations.jl`.
"""
function Time_Evolution_VN(init::Matrix{ComplexF64}, H::MultipleFockOperator,
                           basis::Vector{AbstractFockState},
                           tspan::Tuple{Float64, Float64}, tpoints::NTuple{M, Float64};
                           rtol::Float64 = 1e-9, atol::Float64 = 1e-9,
                           solver = Vern7()) where {M}
    H_dense = Matrix{ComplexF64}(calculate_matrix_elements(H, basis))
    return Time_Evolution_VN(init, H_dense, tspan, tpoints; rtol=rtol, atol=atol, solver=solver)
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
function schrodinger_TD!(dψ, ψ, (tmp, Ops, interps), t)
    fill!(dψ, 0)

    @inbounds for k in eachindex(Ops)
        fk = interps[k](t)
        mul!(tmp, Ops[k], ψ)
        @inbounds @simd for i in eachindex(dψ)
            dψ[i] -= im * fk * tmp[i]
        end
    end

    return nothing
end

function Heisenberg_eom(H::AbstractFockOperator, O::AbstractFockOperator)
    RHS = commutator(H, O)
    typeof(RHS) == FockOperator && (RHS = MultipleFockOperator([RHS], 0))
    return 1im * RHS
end



function Von_Neumann_TD!(dψ, ψ, (tmp, Ops, f_ts), t)
    N = size(Ops[1], 1)
    O = reshape(ψ, N, N)
    dρ = reshape(dψ, N, N)
    fill!(dψ, 0.0 + 0.0im)

    for (H, f) in zip(Ops, f_ts)
        α = -im * f(t)

        mul!(tmp, H, O)          # tmp = H*O
        axpy!(α, tmp, dρ)        # dρ += α * tmp  (no allocation)

        mul!(tmp, O, H)          # tmp = O*H
        axpy!(-α, tmp, dρ)       # dρ -= α * tmp
    end

    return nothing
end

function Von_Neumann_TDM!(dψ, ψ, (tmp, Ops_t), t)
    N = size(tmp, 1)
    O = reshape(ψ, N, N)
    dρ = reshape(dψ, N, N)
    fill!(dψ, 0.0 + 0.0im)

    for Op_t in Ops_t
        Op = Op_t(t)
        α = -im 

        mul!(tmp, Op, O)          # tmp = H*O
        axpy!(α, tmp, dρ)        # dρ += α * tmp  (no allocation)

        mul!(tmp, O, Op)          # tmp = O*H
        axpy!(-α, tmp, dρ)       # dρ -= α * tmp
    end

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
                           ops::NTuple{N, AbstractMatrix{ComplexF64}}, f_ts::Tuple{Vararg{<:Function, N}},
                           tspan::Tuple{Float64, Float64}, tpoints::NTuple{M, Float64};
                           rtol::Float64 = 1e-9, atol::Float64 = 1e-9,
                           solver = Vern7()) where {N, M}
    prob = ODEProblem(schrodinger_TD!, init, tspan, (similar(init), ops, f_ts))
    sol = solve(prob, solver; reltol=rtol, abstol=atol, save_everystep=false, saveat=tpoints)
    return sol
end

function Time_Evolution_TD(init::Vector{ComplexF64},
                           ops::NTuple{N, MultipleFockOperator}, f_ts::Tuple{Vararg{<:Function, N}}, basis::Vector{AbstractFockState},
                           tspan::Tuple{Float64, Float64}, tpoints::NTuple{M, Float64};
                           rtol::Float64 = 1e-9, atol::Float64 = 1e-9,
                           solver = Vern7()) where {N, M}
    ops_tr_rep = Vector{MatrixFreeOperator}()
    for op in ops 
        push!(ops_tr_rep, transition_representation(op, basis))
    end
    return Time_Evolution_TD(init, Tuple(ops_tr_rep), f_ts, tspan, tpoints; rtol=rtol, atol=atol, solver=solver) 
end


function Time_Evolution_TD_VN(init::Matrix{ComplexF64},
                           ops::NTuple{N, AbstractMatrix{ComplexF64}}, f_ts::Tuple{Vararg{<:Function, N}},
                           tspan::Tuple{Float64, Float64}, tpoints::NTuple{M, Float64};
                           rtol::Float64 = 1e-9, atol::Float64 = 1e-9,
                           solver = Vern7()) where {N, M}
    prob = ODEProblem(Von_Neumann_TD!, init, tspan, (similar(init), ops, f_ts))
    sol = solve(prob, solver; reltol=rtol, abstol=atol, save_everystep=false, saveat=tpoints)
    return sol
end

function Time_Evolution_TDM_VN(init::Matrix{ComplexF64},
                           ops_ts::Tuple{Vararg{<:Function, N}},
                           tspan::Tuple{Float64, Float64}, tpoints::NTuple{M, Float64};
                           rtol::Float64 = 1e-9, atol::Float64 = 1e-9,
                           solver = Vern7()) where {N, M}
    prob = ODEProblem(Von_Neumann_TDM!, init, tspan, (similar(init), ops_ts))
    sol = solve(prob, solver; reltol=rtol, abstol=atol, save_everystep=false, saveat=tpoints)
    return sol
end




function Unitary_Ev(H::Matrix{ComplexF64}, ti::Float64, te::Float64)
    U = exp(-1im * H * (te-ti))
    return U  
end

function Unitary_Ev_TD(Ops::NTuple{N, Matrix{ComplexF64}}, f_ts::Tuple{Vararg{<:Function, N}}, ti::Float64, te::Float64, dt::Float64) where {N}
    U = zeros(ComplexF64, size(Ops[1])...)
    for i in 1:size(U, 1)
        U[i,i] = one(ComplexF64)
    end
    H_mid = similar(U)
    U_step = similar(U)
    tmp = similar(U)

    t = ti

    while t < te 
        fill!(H_mid, 0.0 + 0.0im)
        for (H, f) in zip(Ops, f_ts)
            H_mid .+= f(t + dt/2) * H
        end

        # compute U_step = exp(-i H_mid dt)
        U_step .= exp(-1im * H_mid * dt)

        # U = U_step * U, in-place via mul!
        mul!(tmp, U_step, U)
        U .= tmp

        t += dt
    end

    return U
end

function Unitary_Ev_Op(X::Matrix{ComplexF64}, H::Matrix{ComplexF64},
                                    save_times::NTuple{M, Float64}; ρ=false) where {M}
    N = size(X,1)
    tmp = similar(X)
    t0 = 0.0
    snapshots = Vector{Matrix{ComplexF64}}(undef, N)

    for (i, t) in enumerate(save_times)
        Δt = t - t0
        U = exp(-1im * H * Δt)
        Udag = adjoint(U)

        if ρ
            mul!(tmp, U, X)
            mul!(X, tmp, Udag)
        else
            mul!(tmp, Udag, X)
            mul!(X, tmp, U)
        end

        snapshots[i] = copy(X)
        t0 = t
    end

    return snapshots
end


function Unitary_Ev_Op_TD(O::Matrix{ComplexF64}, Ops::NTuple{N, Matrix{ComplexF64}}, f_ts::Tuple{Vararg{<:Function, N}}, 
                              tspan::Tuple{Float64,Float64}, dt::Float64, save_times::NTuple{M, Float64}, ρ=false) where {N,M}
    
    U = zeros(ComplexF64, size(O)...)
    for i in 1:size(U, 1)
        U[i,i] = one(ComplexF64)
    end
    H_mid = similar(O)
    U_step = similar(O)
    tmp = similar(O)
    
    t = tspan[1]
    save_index = 1
    snapshots = Vector{Matrix{ComplexF64}}(undef, M)
    times_recorded = Float64[]

    while t < tspan[2] + 1e-12
        # Construct midpoint Hamiltonian
        fill!(H_mid, 0.0 + 0.0im)
        for (H, f) in zip(Ops, f_ts)
            H_mid .+= f(t + dt/2) * H
        end

        # Compute small-step unitary
        U_step .= exp(-1im * H_mid * dt)

        # Update full unitary: U = U_step * U
        mul!(tmp, U_step, U)
        U .= tmp

        if ρ
            # Update density matrix in-place: ρ -> U ρ U†
            mul!(tmp, U, O)
            mul!(O, tmp, adjoint(U))
        else
            # Update Operator in-place: O -> U† O  U
            mul!(tmp, adjoint(U), O)
            mul!(O, tmp, U)
        end

        t += dt

        # Save snapshots if we passed the next save time
        while save_index <= length(save_times) && t >= save_times[save_index] - dt/2
            snapshots[save_index] = copy(O)  # store a copy
            push!(times_recorded, t)
            save_index += 1
        end
    end

    return times_recorded, snapshots
end










