#
# kernels.jl --
#
# Implementation of local operations with a general purpose kernel which
# is a rectangular array of coefficients over a, possibly off-centered,
# rectangular neighborhood.
#
#------------------------------------------------------------------------------
#
# This file is part of the `LocalFilters.jl` package licensed under the MIT
# "Expat" License.
#
# Copyright (C) 2017, Éric Thiébaut.
#

"""
A kernel can be used to define a versatile type of structuring elements.
"""
struct Kernel{T,N} <: Neighborhood{N}
    coefs::Array{T,N}
    anchor::CartesianIndex{N}
end

# The index in the array of kernel coefficients is `k + anchor` hence:
#
#     1 ≤ k + anchor ≤ dim
#     1 - anchor ≤ k ≤ dim - anchor
#
# thus `first = 1 - anchor` and `last = dim - anchor`.

eltype(B::Kernel{T,N}) where {T,N} = T
length(B::Kernel) = length(coefs(B))
size(B::Kernel) = size(coefs(B))
size(B::Kernel, i) = size(coefs(B), i)
first(B::Kernel) = (I = anchor(B); one(I) - I)
last(B::Kernel) = CartesianIndex(size(coefs(B))) - anchor(B)
getindex(B::Kernel, I::CartesianIndex) = getindex(coefs(B), I + anchor(B))
getindex(B::Kernel, inds::Integer...) = getindex(B, CartesianIndex(inds))

coefs(B::Kernel) = B.coefs
anchor(B::Kernel) = B.anchor

Kernel(B::CartesianBox) = Kernel(ones(Bool, size(B)), anchor(B))
Kernel(B::CenteredBox) = Kernel(ones(Bool, size(B)))

# Wrap an array into a kernel (call copy if you do not want to share).
Kernel(arr::Array{T,N}) where {T,N} =
    Kernel{T,N}(arr, anchor(arr))

function Kernel(arr::AbstractArray{T,N},
                off::CartesianIndex{N}=anchor(arr)) where {T,N}
    Kernel{T,N}(copy!(Array(T, size(arr)), arr), off)
end

function Kernel(::Type{T},
                arr::AbstractArray{T,N},
                off::CartesianIndex{N}=anchor(arr)) where {T,N}
    Kernel(arr, off)
end

function Kernel(tup::Tuple{T,T},
                msk::AbstractArray{Bool,N},
                off::CartesianIndex{N}=anchor(msk)) where {T,N}
    arr = Array(T, size(msk))
    vtrue, vfalse = tup[1], tup[2]
    @inbounds for i in eachindex(arr, msk)
        arr[i] = msk[i] ? vtrue : vfalse
    end
    Kernel{T,N}(arr, off)
end

Kernel(tup::Tuple{T,T}, B::Kernel{Bool,N}) where {T,N} =
    Kernel(tup, coefs(B), anchor(B))

# Make a flat structuring element from a boolean kernel.
function Kernel(::Type{T},
                msk::AbstractArray{Bool,N},
                off::CartesianIndex{N}=anchor(msk)) where {T<:AbstractFloat,N}
    Kernel((zero(T), -T(Inf)), msk, off)
end

Kernel(::Type{T}, B::Kernel{Bool,N}) where {T,N} =
    Kernel(T, coefs(B), anchor(B))


Kernel(::Type{T}, B::Kernel{Bool,N}) where {T<:AbstractFloat,N} =
    Kernel(T, coefs(B), anchor(B))

Kernel(::Type{T1}, msk::AbstractArray{T2,N}) where {T1,T2,N} =
    Kernel(T1, msk, anchor(msk))

Kernel(B::Kernel) = B

Kernel(::Type{Bool}, B::Kernel{Bool,N}) where {N} = B

function strictfloor(::Type{T}, x) where {T}
    n = floor(T, x)
    (n < x ? n : n - one(T)) :: T
end

function ball(rank::Integer, radius::Real)
    b = radius + 1/2
    r = strictfloor(Int, b)
    dim = 2*r + 1
    dims = ntuple(d->dim, rank)
    arr = Array(Bool, dims)
    qmax = strictfloor(Int, b^2)
    _ball!(arr, 0, qmax, r, 1:dim, tail(dims))
    arr
end

@inline function _ball!{N}(arr::AbstractArray{Bool,N},
                           q::Int, qmax::Int, r::Int,
                           range::UnitRange{Int},
                           dims::Tuple{Int}, I::Int...)
    nextdims = tail(dims)
    x = -r
    for i in range
        _ball!(arr, q + x*x, qmax, r, range, nextdims, I..., i)
        x += 1
    end
end

@inline function _ball!(arr::AbstractArray{Bool,N},
                        q::Int, qmax::Int, r::Int,
                        range::UnitRange{Int},
                        ::Tuple{}, I::Int...) where {N}
    x = -r
    for i in range
        arr[I...,i] = (q + x*x ≤ qmax)
        x += 1
    end
end

#
# Pseudo-code for a local operation on `A` in a neighborhood `B` is:
#
#     for i ∈ Sup(A)
#         v = initial()
#         for j ∈ Sup(A) and i - j ∈ Sup(B)
#             v = update(v, A[j], kernel[i-j+off])
#         end
#         dst[i] = final(v)
#     end
#
# where `off` is the anchor offset; the bounds for `j` are:
#
#    imin ≤ j ≤ imax   and   kmin ≤ i - j ≤ kmax
#
# where `imin` and `imax` are the bounds for `A` while `kmin` and `kmax` are
# the bounds for `B`.  The above constraints are identical to:
#
#    max(imin, i - kmax) ≤ j ≤ min(imax, i - kmin)
#

function localfilter!(dst,
                      A::AbstractArray{T,N},
                      B::Kernel{K,N},
                      initial::Function,
                      update::Function,
                      store::Function) where {T,K,N}
    R = CartesianRange(size(A))
    imin, imax = limits(R)
    kmin, kmax = limits(B)
    ker, off = coefs(B), anchor(B)
    @inbounds for i in R
        v = initial()
        k = i + off
        for j in CartesianRange(max(imin, i - kmax), min(imax, i - kmin))
            v = update(v, A[j], ker[k-j])
        end
        store(dst, i, v)
    end
    return dst
end

function localmean!(dst::AbstractArray{T,N},
                    A::AbstractArray{T,N},
                    B::Kernel{Bool,N}) where {T,N}
    @assert size(dst) == size(A)
    localfilter!(dst, A, B,
                 ()      -> (zero(T), 0),
                 (v,a,b) -> b ? (v[1] + a, v[2] + 1) : v,
                 (d,i,v) -> d[i] = v[1]/v[2])
end

function erode!(dst::AbstractArray{T,N},
                A::AbstractArray{T,N},
                B::Kernel{Bool,N}) where {T,N}
    @assert size(dst) == size(A)
    localfilter!(dst, A, B,
                 ()      -> typemax(T),
                 (v,a,b) -> b && a < v ? a : v,
                 (d,i,v) -> d[i] = v)
end

function dilate!(dst::AbstractArray{T,N},
                 A::AbstractArray{T,N},
                 B::Kernel{Bool,N}) where {T,N}
    @assert size(dst) == size(A)
    localfilter!(dst, A, B,
                 ()      -> typemin(T),
                 (v,a,b) -> b && a > v ? a : v,
                 (d,i,v) -> d[i] = v)
end

function localextrema!(Amin::AbstractArray{T,N},
                       Amax::AbstractArray{T,N},
                       A::AbstractArray{T,N},
                       B::Kernel{Bool,N}) where {T,N}
    @assert size(Amin) == size(Amax) == size(A)
    localfilter!((Amin, Amax), A, B,
                 ()      -> (typemax(T),
                             typemin(T)),
                 (v,a,b) -> (b && a < v[1] ? a : v[1],
                             b && a > v[2] ? a : v[2]),
                 (d,i,v) -> (Amin[i], Amax[i]) = v)
end

# Erosion and dilation with a shaped structuring element
# (FIXME: for integers satured addition/subtraction would be needed)

function localmean!(dst::AbstractArray{T,N},
                    A::AbstractArray{T,N},
                    B::Kernel{T,N}) where {T,N}
    @assert size(dst) == size(A)
    localfilter!(dst, A, B,
                 ()      -> (zero(T), zero(T)),
                 (v,a,b) -> (v[1] + a*b, v[2] + b),
                 (d,i,v) -> d[i] = v[1]/v[2])
end

function erode!(dst::AbstractArray{T,N},
                A::AbstractArray{T,N},
                B::Kernel{T,N}) where {T,N}
    localfilter!(dst, A, B,
                 ()      -> typemax(T),
                 (v,a,b) -> min(v, a - b),
                 (d,i,v) -> d[i] = v)
end

function dilate!(dst::AbstractArray{T,N},
                 A::AbstractArray{T,N},
                 B::Kernel{T,N}) where {T,N}
    @assert size(dst) == size(A)
    localfilter!(dst, A, B,
                 ()      -> typemin(T),
                 (v,a,b) -> max(v, a + b),
                 (d,i,v) -> d[i] = v)
end

function localextrema!(Amin::AbstractArray{T,N},
                       Amax::AbstractArray{T,N},
                       A::AbstractArray{T,N},
                       B::Kernel{T,N}) where {T,N}
    @assert size(Amin) == size(Amax) == size(A)
    localfilter!((Amin, Amax), A, B,
                 ()      -> (typemax(T),
                             typemin(T)),
                 (v,a,b) -> (min(v[1], a - b),
                             max(v[2], a + b)),
                 (d,i,v) -> (Amin[i], Amax[i]) = v)
end

function convolve!(dst::AbstractArray{T,N},
                   A::AbstractArray{T,N},
                   B::Kernel{Bool,N}) where {T,N}
    @assert size(dst) == size(A)
    localfilter!(dst, A, B,
                 ()      -> zero(T),
                 (v,a,b) -> b ? v + a : v,
                 (d,i,v) -> d[i] = v)
end

function convolve!(dst::AbstractArray{T,N},
                   A::AbstractArray{T,N},
                   B::Kernel{T,N}) where {T,N}
    @assert size(dst) == size(A)
    localfilter!(dst, A, B,
                 ()      -> zero(T),
                 (v,a,b) -> v + a*b,
                 (d,i,v) -> d[i] = v)
end
