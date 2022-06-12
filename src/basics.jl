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
# Copyright (C) 2017-2022, Éric Thiébaut.
#

# Extend `CartesianIndices`.
CartesianIndices(B::Neighborhood) = CartesianIndices(axes(B))
convert(::Type{CartesianIndices}, B::Neighborhood) = CartesianIndices(B)
convert(::Type{CartesianIndices{N}}, B::Neighborhood{N}) where N =
    CartesianIndices(B)

# Default implementation of common methods.
ndims(::Neighborhood{N}) where {N} = N
length(B::Neighborhood) = prod(size(B))
size(B::Neighborhood) = map(_length,
                            Tuple(first_cartesian_index(B)),
                            Tuple(last_cartesian_index(B)))
size(B::Neighborhood, d) = _length(first_cartesian_index(B)[d],
                                   last_cartesian_index(B)[d])
@inline axes(B::Neighborhood) = map((i,j) -> i:j,
                                    Tuple(first_cartesian_index(B)),
                                    Tuple(last_cartesian_index(B)))
axes(B::Neighborhood, d) =
    (first_cartesian_index(B)[d]:last_cartesian_index(B)[d])
getindex(B::Neighborhood, inds::Union{Integer,CartesianIndex}...) =
    getindex(B, CartesianIndex(inds...))
setindex!(B::Neighborhood, val, inds::Union{Integer,CartesianIndex}...) =
    setindex!(B, val, CartesianIndex(inds...))

@inline _length(start::Int, stop::Int) = max(Int(stop) - Int(start) + 1, 0)

"""
    default_start(A) -> I::CartesianIndex

yields the initial (multi-dimensional) index of a rectangular region which has
the same size as the array `A` but whose origin (that is, index
`zero(CartesianIndex{N})`) is at the geometrical center of the region (with the
same conventions as `fftshift`.

"""
default_start(A::AbstractArray) =
    CartesianIndex(map(rng -> -(Int(length(rng)) >> 1), axes(A)))

"""
    first_cartesian_index(B) -> Imin::CartesianIndex{N}
    last_cartesian_index(B)  -> Imax::CartesianIndex{N}

respectively yield the first and last multi-dimensional Cartesian index for
indexing the Cartesian region defined by `B`.  A Cartesian region defines a
rectangular set of indices whose edges are aligned with the indexing axes.

Compared to similar methods [`firstindex`](@ref), [`lastindex()`](@ref),
[`first()`](@ref) and [`last()`](@ref), the returned value is always an
instance of `CartesianIndex{N}` with `N` the number of dimensions.

Any multi-dimensional index `I::CartesianIndex{N}` is in the Cartesian region
defined `B` if and only if `Imin ≤ I ≤ Imax`.

Also see: [`limits`](@ref).

"""
first_cartesian_index(B::Neighborhood) = B.start
last_cartesian_index(B::Neighborhood) = B.stop

first_cartesian_index(R::CartesianIndices) = first(R)
last_cartesian_index(R::CartesianIndices) = last(R)

first_cartesian_index(A::AbstractArray) = first_cartesian_index(axes(A))
last_cartesian_index(A::AbstractArray) = last_cartesian_index(axes(A))

first_cartesian_index(inds::UnitIndexRanges{N}) where {N} =
    CartesianIndex(map(first, inds))
last_cartesian_index(inds::UnitIndexRanges{N}) where {N} =
     CartesianIndex(map(last, inds))

first_cartesian_index(inds::NTuple{2,CartesianIndex{N}}) where {N} = inds[1]
last_cartesian_index(inds::NTuple{2,CartesianIndex{N}}) where {N} = inds[2]

"""
    limits(T::DataType) -> typemin(T), typemax(T)

yields the infimum and supremum of a type `T`.

    limits(B) -> Imin, Imax

yields the corners (as a tuple of 2 `CartesianIndex`) of the Cartesian
region defined by `B`.

Also see: [`first_cartesian_index`](@ref) and [`last_cartesian_index`](@ref).

"""
limits(T::Type) = (typemin(T), typemax(T))
limits(A::AbstractArray) = limits(axes(A)) # provides a slight optimization?
limits(B::Neighborhood) = (first_cartesian_index(B),
                           last_cartesian_index(B))
limits(R::CartesianIndices) = (first_cartesian_index(R),
                               last_cartesian_index(R))
limits(inds::UnitIndexRanges) = (first_cartesian_index(inds),
                                 last_cartesian_index(inds))
limits(inds::NTuple{2,CartesianIndex{N}}) where {N} = inds

"""
    cartesian_region(args...) -> R

yields the rectangular region (as an instance of `CartesianIndices`) specified
by the arguments which can be:

* an abstract array whose axes define the region (see [`axes`](@ref));

* a list of unit range indices and/or indices along each dimension;

* the corners of the bounding box, say `start` and `stop`, specified as two
  instances of `CartesianIndex`;

* a neighborhood (see [`Neighborhood`](@ref));

* an instance of `CartesianIndices.

This method is mostly similar to `CartesianIndices`, it is introduced in
`LocalFilters` to avoid type-piracy when dealing with arguments not handled
`CartesianIndices`.

See also: [`first_cartesian_index`](@ref), [`last_cartesian_index`](@ref) and [`limits`](@ref).

"""
cartesian_region(B::Neighborhood) =
    cartesian_region(first_cartesian_index(B), last_cartesian_index(B))
cartesian_region(A::AbstractArray) = cartesian_region(axes(A))
cartesian_region(R::CartesianIndices) = R
cartesian_region(inds::UnitIndexRanges) = CartesianIndices(inds)

# The most critical version of `cartesian_region` is the one which takes the
# first and last indices of the region and which is inlined.
@inline function cartesian_region(start::CartesianIndex{N},
                                 stop::CartesianIndex{N}) where N
    return CartesianIndices(map((i,j) -> i:j, Tuple(start), Tuple(stop)))
end

#------------------------------------------------------------------------------
# UTILITIES

"""
    check_indices(A...)

throws an exception if arrays `A...` have different indices.

---

    check_indices(Bool, [I,] A...)

yields whether arrays `A...` all have the same indices, or all have indices
`I` if specified.

"""
check_indices(A::AbstractArray) = nothing

function check_indices(A::AbstractArray, B::AbstractArray...)
    throw(DimensionMismatch(
        "arrays have different number of dimensions"))
end

# This version is forced to be in-lined to unroll the recursion.
@inline function check_indices(A::AbstractArray{<:Any,N},
                               B::AbstractArray{<:Any,N}...) where {N}
    check_indices(Bool, axes(A), B...) || throw(DimensionMismatch(
        "arrays have different indices"))
    return nothing
end

check_indices(::Type{Bool}) = false
check_indices(::Type{Bool}, A::AbstractArray) = true
check_indices(::Type{Bool}, A::AbstractArray...) = false

# This version is forced to be in-lined to unroll the recursion.
@inline function check_indices(::Type{Bool},
                               A::AbstractArray{<:Any,N},
                               B::AbstractArray{<:Any,N}...) where {N}
    return check_indices(Bool, axes(A), B...)
end

function check_indices(::Type{Bool},
                       I::Tuple{Vararg{AbstractUnitRange{<:Integer}}},
                       A::AbstractArray...)
    return false
end

# This version is forced to be in-lined to unroll the recursion.
@inline function check_indices(::Type{Bool},
                               I::NTuple{N,AbstractUnitRange{<:Integer}},
                               A::AbstractArray{<:Any,N},
                               B::AbstractArray{<:Any,N}...) where {N}
    return axes(A) == I && check_indices(Bool, I, B...)
end

function check_indices(::Type{Bool},
                       I::NTuple{N,AbstractUnitRange{<:Integer}},
                       A::AbstractArray{<:Any,N}) where {N}
    return axes(A) == I
end


#------------------------------------------------------------------------------
# CONVERSIONS

# To implement variants and out-of-place versions, we define conversion rules
# to convert various types of arguments into a neighborhood suitable with the
# source (e.g., of given rank `N`).

convert(::Type{Neighborhood}, x) = Neighborhood(x)
convert(::Type{Neighborhood}, x::Neighborhood) = x
convert(::Type{Neighborhood{N}}, x) where {N} = Neighborhood{N}(x)
convert(::Type{Neighborhood{N}}, x::Neighborhood{N}) where {N} = x

convert(::Type{RectangularBox}, x) = RectangularBox(x)
convert(::Type{RectangularBox}, x::RectangularBox) = x
convert(::Type{RectangularBox{N}}, x) where {N} = RectangularBox{N}(x)
convert(::Type{RectangularBox{N}}, x::RectangularBox{N}) where {N} = x

convert(::Type{Kernel}, x) = Kernel(x)
convert(::Type{Kernel}, x::Kernel) = x
convert(::Type{Kernel{T}}, x) where {T} = Kernel{T}(x)
convert(::Type{Kernel{T}}, x::Kernel{T}) where {T} = x
convert(::Type{Kernel{T,N}}, x) where {T,N} = Kernel{T,N}(x)
convert(::Type{Kernel{T,N}}, x::Kernel{T,N}) where {T,N} = x

#------------------------------------------------------------------------------
# METHODS FOR NEIGHBORHOODS

# Outer constructors for Neighborhood.  All constructors taking a single
# argument must account for any explicit parametrization.

Neighborhood(B::Neighborhood) = B
Neighborhood{N}(B::Neighborhood{N}) where {N} = B

Neighborhood(A::AbstractArray) = Kernel(A)
Neighborhood{N}(A::AbstractArray{T,N}) where {T,N} = Kernel(A)

Neighborhood(dim::Integer) = RectangularBox(dim)
Neighborhood{N}(dim::Integer) where {N} = RectangularBox{N}(dim)
Neighborhood(dims::Integer...) = RectangularBox(dims)
Neighborhood{N}(dims::Integer...) where {N} = RectangularBox{N}(dims)
Neighborhood(dims::Dimensions{N}) where {N} = RectangularBox(dims)
Neighborhood{N}(dims::Dimensions{N}) where {N} = RectangularBox(dims)

Neighborhood(rng::UnitIndexRange) = RectangularBox(rng)
Neighborhood{N}(rng::UnitIndexRange) where {N} = RectangularBox{N}(rng)
Neighborhood(rngs::UnitIndexRange...) = RectangularBox(rngs)
Neighborhood{N}(rngs::UnitIndexRange...) where {N} = RectangularBox{N}(rngs)
Neighborhood(rngs::UnitIndexRanges{N}) where {N} = RectangularBox(rngs)
Neighborhood{N}(rngs::UnitIndexRanges{N}) where {N} = RectangularBox(rngs)

Neighborhood(R::CartesianIndices) = RectangularBox(R)
Neighborhood{N}(R::CartesianIndices{N}) where {N} = RectangularBox(R)

#------------------------------------------------------------------------------
# METHODS FOR RECTANGULAR BOXES

RectangularBox(B::RectangularBox) = B
RectangularBox{N}(B::RectangularBox{N}) where {N} = B

RectangularBox(dim::Integer) = RectangularBox{1}(dim)
RectangularBox{N}(dim::Integer) where {N} = RectangularBox{N}(_range(dim))
RectangularBox(dims::Integer...) = RectangularBox(dims)
RectangularBox{N}(dims::Integer...) where {N} = RectangularBox{N}(dims)
RectangularBox(dims::Dimensions{N}) where {N} =
    RectangularBox(map(_range, dims))
RectangularBox{N}(dims::Dimensions{N}) where {N} = RectangularBox(dims)

RectangularBox(rng::UnitIndexRange) = RectangularBox{1}(rng)
function RectangularBox{N}(rng::UnitIndexRange) where {N}
    imin = Int(first(rng))
    imax = Int(last(rng))
    Imin = CartesianIndex(ntuple(d -> imin, Val(N)))
    Imax = CartesianIndex(ntuple(d -> imax, Val(N)))
    return RectangularBox{N}(Imin, Imax)
end
RectangularBox(rngs::UnitIndexRange...) = RectangularBox(rngs)
RectangularBox{N}(rngs::UnitIndexRange...) where {N} = RectangularBox{N}(rngs)
function RectangularBox(rngs::UnitIndexRanges{N}) where {N}
    I1 = CartesianIndex(map(r -> Int(first(r)), rngs))
    I2 = CartesianIndex(map(r -> Int(last(r)), rngs))
    return RectangularBox{N}(I1, I2)
end
RectangularBox{N}(rngs::UnitIndexRanges{N}) where {N} = RectangularBox(rngs)

RectangularBox(R::CartesianIndices) =
    RectangularBox(first_cartesian_index(R), last_cartesian_index(R))
RectangularBox{N}(R::CartesianIndices{N}) where {N} =
    RectangularBox(first_cartesian_index(R), last_cartesian_index(R))


"""
    ismmbox(B)

yields whether neighborhood `B` has the same effect as a rectangular box for
mathematical morphology operations.  This may be used to use fast separable
versions of mathematical morphology operations like the van Herk-Gil-Werman
algorithm.

See also: [`LocalFilters.RectangularBox`](@ref).
"""
ismmbox(::RectangularBox) = true
ismmbox(B::Kernel{Bool}) = all(identity, coefs(B))
ismmbox(B::Kernel{T}) where {T<:AbstractFloat} =
    all(x -> x == zero(T), coefs(B))
ismmbox(::Neighborhood) = false

_range(dim::Integer) = _range(Int(dim))

function _range(dim::Int)
    dim ≥ 1 ||
        throw(ArgumentError("neighborhood dimension(s) must be at least one"))
    imin = -(dim >> 1)
    imax = dim + imin - 1
    return imin:imax
end

#------------------------------------------------------------------------------
# METHODS FOR KERNELS

eltype(B::Kernel) = eltype(typeof(B))
eltype(::Type{<:Kernel{T,N}}) where {T,N} = T
ndims(B::Kernel) = ndims(typeof(B))
ndims(::Type{<:Kernel{T,N}}) where {T,N} = N
length(B::Kernel) = length(coefs(B))
size(B::Kernel) = size(coefs(B))
size(B::Kernel, d) = size(coefs(B), d)
getindex(B::Kernel, I::CartesianIndex) = getindex(coefs(B), I + offset(B))
setindex!(B::Kernel, val, I::CartesianIndex) =
    setindex!(coefs(B), val, I + offset(B))

"""
    LocalFilters.coefs(B)

yields the array of coefficients embedded in kernel `B`.

See also: [`LocalFilters.offset`](@ref).

"""
coefs(B::Kernel) = B.coefs

"""
    LocalFilters.offset(B)

yields the index offset of the array of coefficients embedded in kernel `B`.
That is, `B[k] ≡ coefs(B)[k + offset(B)]`.

See also: [`LocalFilters.coefs`](@ref).

"""
offset(B::Kernel) = B.offset

# Kernel constructors given a Kernel instance.
Kernel(K::Kernel) = K
Kernel{T}(K::Kernel{T}) where {T} = K
Kernel{T}(K::Kernel{S}) where {S,T} = Kernel(convert_coefs(T, coefs(K)),
                                             first_cartesian_index(K))
Kernel{T,N}(K::Kernel{T,N}) where {T,N} = K
Kernel{T,N}(K::Kernel{S,N}) where {S,T,N} = Kernel{T}(K)

# Kernel constructors given a RectangularBox instance.
Kernel(B::RectangularBox{N}) where {N} = Kernel{Bool,N}(B)
Kernel{T}(B::RectangularBox{N}) where {T,N} = Kernel{T,N}(B)
Kernel{T,N}(B::RectangularBox{N}) where {T,N} =
    Kernel{T,N,Array{T,N}}(ones(T, size(B)), first_cartesian_index(B))

# Kernel constructors given an array of coefficients.
Kernel(A::AbstractArray) = Kernel(A, default_start(A))
Kernel{T}(A::AbstractArray{T}) where {T} = Kernel(A)
Kernel{T}(A::AbstractArray{S}) where {S,T} = Kernel(convert_coefs(T, A))
Kernel{T,N}(A::AbstractArray{T,N}) where {T,N} = Kernel(A)
Kernel{T,N}(A::AbstractArray{S,N}) where {S,T,N} = Kernel{T}(A)

# Kernel constructors given an array of coefficients and starting indices.
Kernel(A::AbstractArray{T,N}, start::CartesianIndex{N}) where {T,N} =
    Kernel{T,N,typeof(A)}(A, start)
Kernel{T}(A::AbstractArray{T,N}, start::CartesianIndex{N}) where {T,N} =
    Kernel{T,N,typeof(A)}(A, start)
Kernel{T,N}(A::AbstractArray{T,N}, start::CartesianIndex{N}) where {T,N} =
    Kernel{T,N,typeof(A)}(A, start)

# Kernel constructors given an array of coefficients and any argument
# suitable to define a Cartesian region.

function Kernel(A::AbstractArray{T,N}, bnds::CartesianRegion{N}) where {T,N}
    # Bounds for indexing the kernel.
    kmin, kmax = limits(bnds)

    # Bounds for indexing the array of coefficients.
    jmin, jmax = limits(A)

    # Check size is identical for all dimensions.
    all(d -> (jmax[d] - jmin[d]) == (kmax[d] - kmin[d]), 1:N) ||
        throw(DimensionMismatch("dimensions must be the same"))

    # Make a kernel with the correct initial index.
    return Kernel(A, kmin)
end

Kernel{T}(A::AbstractArray{T,N}, bnds::CartesianRegion{N}) where {T,N} =
    Kernel(A, bnds)
Kernel{T}(A::AbstractArray{S,N}, bnds::CartesianRegion{N}) where {S,T,N} =
    Kernel(convert_coefs(T, A), bnds)
Kernel{T,N}(A::AbstractArray{S,N}, bnds::CartesianRegion{N}) where {S,T,N} =
    Kernel{T}(A, bnds)

Kernel(A::AbstractArray, rngs::UnitIndexRange...) = Kernel(A, rngs)
Kernel{T}(A::AbstractArray{S}, rngs::UnitIndexRange...) where {S,T} =
    Kernel{T}(A, bnds)
Kernel{T,N}(A::AbstractArray{S,N}, rngs::UnitIndexRange...) where {S,T,N} =
    Kernel{T}(A, rngs)

# Another ways to specify the element type of the kernel coefficients is to
# have their type the first parameter.
@deprecate Kernel(T::Type, B, args...) Kernel{T}(B, args...) false

# Methods to convert other neighborhoods.  Beware that booleans mean something
# specific, i.e. the result is a so-called *flat* structuring element
# when a kernel whose coefficients are boolean is converted to some
# floating-point type.  See [`convert_coefs`](@ref).

# Make a flat structuring element from a boolean mask.
Kernel(tup::Tuple{<:Any,<:Any}, msk::AbstractArray{Bool}, args...) =
    Kernel(convert_coefs(tup, msk), args...)

Kernel(tup::Tuple{<:Any,<:Any}, B::Kernel{Bool}) =
    Kernel(convert_coefs(tup, coefs(B)), first_cartesian_index(B))

# Make a kernel from a function and anything suitable to define a Cartesian
# region.  The element type of the kernel coefficients can be imposed.
Kernel(f::Function, bnds::CartesianRegion) =
    Kernel(map(f, cartesian_region(bnds)), first_cartesian_index(bnds))
function Kernel{T}(f::Function, bnds::CartesianRegion{N}) where {T,N}
    kmin, kmax = limits(bnds)
    dims = map(_length, Tuple(kmin), Tuple(kmax))
    W = Array{T,N}(undef, dims)
    offs = first_cartesian_index(W) - kmin
    @inbounds for i in cartesian_region(bnds)
        W[i + offs] = f(i)
    end
    return Kernel{T,N,Array{T,N}}(W, kmin)
end
Kernel{T,N}(f::Function, bnds::CartesianRegion{N}) where {T,N} =
    Kernel{T}(f, bnds)

# Conversion of the data type of the kernel coefficients.
for F in (:Float64, :Float32, :Float16)
    @eval begin
        Base.$F(K::Kernel{$F}) = K
        Base.$F(K::Kernel{T}) where {T} = Kernel{$F}(K)
    end
end

"""
    strel(T, A)

yields a *structuring element* suitable for mathematical morphology operations.
The result is a `Kernel` whose elements have type `T` (which can be `Bool` or a
floating-point type).  Argument `A` can be a Cartesian box or a `Kernel`
with boolean elements.

If `T` is a floating-point type, then the result is a so-called *flat*
structuring element whose coefficients are `zero(T)` inside the shape defined
by `A` and `-T(Inf)` elsewhere.

"""
strel(::Type{Bool}, K::Kernel{Bool,N}) where {N} = K
strel(::Type{T}, K::Kernel{Bool}) where {T<:AbstractFloat} =
    Kernel(convert_coefs((zero(T), -T(Inf)), coefs(K)), first_cartesian_index(K))
strel(::Type{Bool}, B::RectangularBox) = Kernel{Bool}(B)
strel(::Type{T}, B::RectangularBox) where {T<:AbstractFloat} =
    Kernel(zeros(T, size(B)), first_cartesian_index(B))

"""
    convert_coefs(T, A)

yields an array of kernel coefficients equivalent to array `A` but whose
elements have type `T`.

If `T` is a floating-point type and `A` is a boolean array, then the values of
the result are `one(T)` where `A` is `true` and `zero(T)` elsewhere.  To use
different values (for instance, to define *flat* *structuring* *elements*), you
may call:

    convert_coefs((vtrue, vfalse), A)

with `A` a boolean array to get an array whose elements are equal to `vtrue`
where `A` is `true` and to `vfalse` otherwise.

See also: [`strel`](@ref).

"""
convert_coefs(::Type{T}, A::AbstractArray{T,N}) where {T,N} = A

function convert_coefs(::Type{T}, A::AbstractArray{S,N}) where {S,T,N}
    B = similar(Array{T,N}, axes(A))
    @inbounds @simd for i in eachindex(A, B)
        B[i] = A[i]
    end
    return B
end

convert_coefs(tup::Tuple{<:Any,<:Any}, A::AbstractArray{Bool}) =
    convert_coefs(promote(tup...), A)

function convert_coefs(tup::Tuple{T,T}, A::AbstractArray{Bool,N}) where {T,N}
    B = similar(Array{T,N}, axes(A))
    vtrue, vfalse = tup
    @inbounds for i in eachindex(A, B)
        B[i] = A[i] ? vtrue : vfalse
    end
    return B
end

"""
    LocalFilters.reverse(B::LocalFilters.Neighborhood)

yields neighborhood `B` reversed along all its dimensions.  This can be used
to correlate by `B` rather than convolving by `B`.

"""
reverse(box::RectangularBox) =
    RectangularBox(-last_cartesian_index(box), -first_cartesian_index(box))

function reverse(ker::Kernel{T,N}) where {T,N}
    kmin = -last_cartesian_index(ker)
    kmax = -first_cartesian_index(ker)
    rev = Kernel(Array{T}(undef, size(ker)), kmin)
    @inbounds for k in cartesian_region(kmin, kmax)
        rev[k] = ker[-k]
    end
    return rev
end

function strict_floor(::Type{T}, x)::T where {T}
    n = floor(T, x)
    return (n < x ? n : n - one(T))
end

"""
    LocalFilters.ball(N, r)

yields a boolean mask which is a `N`-dimensional array with all dimensions odd
and equal and set to true where position is inside a `N`-dimensional ball of
radius `r`.

"""
function ball(rank::Integer, radius::Real)
    b = radius + 1/2
    r = strict_floor(Int, b)
    dim = 2*r + 1
    dims = ntuple(d->dim, rank)
    arr = Array{Bool}(undef, dims)
    qmax = strict_floor(Int, b^2)
    _ball!(arr, 0, qmax, r, 1:dim, tail(dims))
    return arr
end

@inline function _ball!(arr::AbstractArray{Bool,N},
                        q::Int, qmax::Int, r::Int,
                        range::AbstractUnitRange{Int},
                        dims::Tuple{Int}, I::Int...) where {N}
    nextdims = tail(dims)
    x = -r
    for i in range
        _ball!(arr, q + x*x, qmax, r, range, nextdims, I..., i)
        x += 1
    end
end

@inline function _ball!(arr::AbstractArray{Bool,N},
                        q::Int, qmax::Int, r::Int,
                        range::AbstractUnitRange{Int},
                        ::Tuple{}, I::Int...) where {N}
    x = -r
    for i in range
        arr[I...,i] = (q + x*x ≤ qmax)
        x += 1
    end
end
