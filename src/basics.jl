#
# basics.jl --
#
# Basic methods for local filters.
#
#------------------------------------------------------------------------------
#
# This file is part of the `LocalFilters.jl` package licensed under the MIT
# "Expat" License.
#
# Copyright (C) 2017-2018, Éric Thiébaut.
#

# Default implementation of common methods.
ndims(::Neighborhood{N}) where N = N
length(B::Neighborhood) = prod(size(B))
size(B::Neighborhood{N}) where N = ntuple(i -> size(B, i), N)

"""

    anchor(B)    -> I::CartesianIndex{N}

yields the anchor of the structuring element `B` that is the Cartesian index of
the central position in the structuring element within its bounding-box.  `N`
is the number of dimensions.  Argument can also be `K` or `size(K)` to get the
default anchor for kernel `K` (an array).

"""
anchor(dims::NTuple{N,Integer}) where {N} =
    CartesianIndex(ntuple(d -> (Int(dims[d]) >> 1) + 1, N))
anchor(B::Neighborhood) = (I = first(B); one(I) - I)
anchor(A::AbstractArray) = anchor(size(A))

"""
The `limits` method yields the corners (as a tuple of 2 `CartesianIndex`)
of `B` (an array, a `CartesianRange` or a `Neighborhood`) and the
infium and supremum of a type `T`:

    limits(B) -> first(B), last(B)
    limits(T) -> typemin(T), typemax(T)

"""
limits(R::CartesianRange) = first(R), last(R)
limits(::Type{T}) where {T} = typemin(T), typemax(T)
limits(A::AbstractArray) = limits(CartesianRange(size(A)))
limits(B::Neighborhood) = first(B), last(B)

CartesianRange(B::Neighborhood{N}) where {N} =
    CartesianRange{CartesianIndex{N}}(first(B), last(B))

#------------------------------------------------------------------------------

# To implement variants and out-of-place versions, we define conversion rules
# to convert various types of arguments into a neighborhood suitable with the
# source (e.g., of given rank `N`).

convert(::Type{Neighborhood{N}}, dim::Integer) where {N} =
    CenteredBox(ntuple(i->dim, N))

convert(::Type{Neighborhood{N}}, dims::Vector{T}) where {N,T<:Integer} =
    (@assert length(dims) == N; CenteredBox(dims...))

convert(::Type{Neighborhood{N}}, A::AbstractArray{T,N}) where {T,N} =
    Kernel(A)

function convert(::Type{Neighborhood{N}},
                 R::CartesianRange{CartesianIndex{N}}) where {N}
    CartesianBox(R)
end

function convert(::Type{Neighborhood{N}},
                 inds::NTuple{N,AbstractUnitRange{T}}) where {N,T<:Integer}
    CartesianBox(inds)
end

#------------------------------------------------------------------------------
# METHODS FOR CENTERED BOXES

@inline function halfdim(n::Integer)
    @assert n ≥ 1 && isodd(n) "dimensions of centered box must be ≥ 1 and odd"
    Int(n)>>1
end

CenteredBox(B::CenteredBox) = B

CenteredBox(siz::Integer...) =
    (N = length(siz);
     CenteredBox{N}(CartesianIndex(ntuple(i -> halfdim(siz[i]), N))))

CenteredBox(siz::Vector{<:Integer}) =
    (N = length(siz);
     CenteredBox{N}(CartesianIndex(ntuple(i -> halfdim(siz[i]), N))))

CenteredBox(siz::NTuple{N,Integer}) where {N} =
    CenteredBox{N}(CartesianIndex(ntuple(i -> halfdim(siz[i]), N)))

eltype(B::CenteredBox) = Bool
size(B::CenteredBox, i) = 2*last(B)[i] + 1
first(B::CenteredBox) = -last(B)
last(B::CenteredBox) = B.last
getindex(B::CenteredBox, i::CartesianIndex) = true
getindex(B::CenteredBox, i::Integer...) = true

#------------------------------------------------------------------------------
#
# METHODS FOR CARTESIAN BOXES
# ===========================
#
# A CartesianBox is a neighborhood defined by a rectangular box, possibly
# off-centered.
#

CartesianBox(B::CartesianBox) = B

CartesianBox(B::CenteredBox) = CartesianBox(CartesianRange(B))

CartesianBox(I0::CartesianIndex{N}, I1::CartesianIndex{N}) where {N} =
    CartesianBox(CartesianRange(I0, I1))

CartesianBox(dims::NTuple{N,Integer}, offs::NTuple{N,Integer}) where {N} =
    (I = CartesianIndex(offs);
     CartesianBox(one(I) - I, CartesianIndex(dims) - I))

"""

A `CartesianBox` can be defined by the index ranges along all the dimensions.
For example:

```julia

CartesianBox(-3:3, -2:1)
```

yields a 2-dimensional `CartesianBox` of size `7×4` and whose first index
varies on `-3:3` while its second index varies on `-2:1`.

"""
CartesianBox(inds::AbstractUnitRange{T}...) where {T<:Integer} =
    CartesianBox(inds)

CartesianBox(inds::NTuple{N,AbstractUnitRange{T}}) where {N,T<:Integer} =
    CartesianBox(CartesianIndex(map(first, inds)),
                 CartesianIndex(map(last, inds)))

eltype(B::CartesianBox) = Bool
size(B::CartesianBox) = size(B.bounds)
size(B::CartesianBox, i) = max(last(B)[i] - first(B)[i] + 1, 0)
first(B::CartesianBox) = first(B.bounds)
last(B::CartesianBox) = last(B.bounds)
limits(B::CartesianBox) = first(B), last(B)
getindex(B::CartesianBox, I::CartesianIndex) = true
getindex(B::CartesianBox, inds::Integer...) = true

bounds(B::Kernel) = B.bounds

# Cartesian boxes can be used as iterators.
Base.start(B::CartesianBox) = start(B.bounds)
Base.done(B::CartesianBox, state) = done(B.bounds, state)
Base.next(B::CartesianBox, state) = next(B.bounds, state)

#------------------------------------------------------------------------------
# METHODS FOR KERNELS

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
    Kernel{T,N}(copy!(Array{T}(size(arr)), arr), off)
end

function Kernel(::Type{T},
                arr::AbstractArray{T,N},
                off::CartesianIndex{N}=anchor(arr)) where {T,N}
    Kernel(arr, off)
end

function Kernel(tup::Tuple{T,T},
                msk::AbstractArray{Bool,N},
                off::CartesianIndex{N}=anchor(msk)) where {T,N}
    arr = Array{T}(size(msk))
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

# Conversion of the data type of the kernel coefficients.
for F in (:Float64, :Float32, :Float16)
    @eval begin
        Base.$F(K::Kernel{$F,N}) where {N} = K
        Base.$F(K::Kernel{T,N}) where {T,N} =
            Kernel{$F,N}(convert(Array{$F,N}, K.coefs), K.anchor)
    end
end

function strictfloor(::Type{T}, x) where {T}
    n = floor(T, x)
    (n < x ? n : n - one(T)) :: T
end

function ball(rank::Integer, radius::Real)
    b = radius + 1/2
    r = strictfloor(Int, b)
    dim = 2*r + 1
    dims = ntuple(d->dim, rank)
    arr = Array{Bool}(dims)
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
