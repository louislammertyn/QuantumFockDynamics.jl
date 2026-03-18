function combinations(v::Vector, n::Int)
    if n == 0
        return [Vector{eltype(v)}()]  # single empty combination
    elseif n > length(v)
        return Vector{Vector{eltype(v)}}()  # no combinations
    else
        result = Vector{Vector{eltype(v)}}()
        for (i, x) in enumerate(v)
            # take x and combine with all combinations of remaining elements
            for tail in combinations(v[i+1:end], n-1)
                push!(result, [x; tail])
            end
        end
        return result
    end
end

function combinations(v::Tuple, n::Int)
    return combinations(collect(v),n)
end

function Identity(n::Int)
    M = zeros(ComplexF64, n, n)
    for i in axes(M, 1)
        M[i,i] +=1 
    end
    return M 
end

function fold(k::Float64, a::Float64=1.)
    return k > π/a ? -(2π/a - k) : k
end

function folded_k_from_modes(ks::Vector{Int}, L::Int)
    ks .*= 2*π/L
    return fold.(ks)
end

