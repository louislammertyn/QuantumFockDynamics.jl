using SparseArrayKit
using TensorOperations

function randn_sparse(T::Type{<:Number}, sz::Dims, p=0.5)
    a = SparseArray{T}(undef, sz)
    for I in keys(a)
        if rand() < p
            a[I] = randn(T)
        end
    end
    return a
end

dim = 100

T = randn_sparse(ComplexF64, (dim,dim,dim), 0.001);

v1 = SparseArray(randn(ComplexF64, dim));
v2 = SparseArray(randn(ComplexF64, dim));
v3 = SparseArray(randn(ComplexF64, dim));

@time @tensor a = v1[i] * v2[j] * v3[k] * T[i,j,k];

T = randn(ComplexF64, (dim,dim,dim));

v1 = randn(ComplexF64, dim);
v2 = randn(ComplexF64, dim);
v3 = randn(ComplexF64, dim);

@time @tensor a = v1[i] * v2[j] * v3[k] * T[i,j,k];

T = randn_sparse(ComplexF64, (dim,dim,dim), 0.1);

v1 = randn(ComplexF64, dim);
v2 = randn(ComplexF64, dim);
v3 = randn(ComplexF64, dim);

a=0;
i = nonzero_keys(T);
@time for indices in i
    a += v1[indices[1]] * v2[indices[2]] * v3[indices[3]] * T[indices[1],indices[2],indices[3]];
end
