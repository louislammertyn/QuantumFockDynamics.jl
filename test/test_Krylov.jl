using Revise
using FockSpace, VectorInterface, KrylovKit

N = 5
geometry = (3,)
V = U1FockSpace(geometry, N,N)

basisFS(V)
s = fock_state(V, [1, 1, 3], 2. + 3im)
w = fock_state(V, [3, 0, 2], 1. + 4im)
scalartype(s)
s = MultipleFockState([s])
scale(s, 2)
scale(ZeroFockState(),  2)

H = FockOperator(((1, true), (2, false) ), 1. +0im, V)
H +=  FockOperator(((2, true), (1, false) ), 1. +0im, V)
H+=  FockOperator(((2, true), (3, false) ), 1. +0im, V)
H+=  FockOperator(((3, true), (2, false) ), 1. +0im, V)
(H*s) * (H*(H * s))

eigsolve(x->H*x, s; ishermitian=true)
Tx = typeof(s)
Tfx = Core.Compiler.return_type(apply, Tuple{typeof(x->H*x),Tx})
T = Core.Compiler.return_type(dot, Tuple{Tx,Tfx})
T
Core.Compiler.return_type(norm, Tuple{MultipleFockState})
@which dot