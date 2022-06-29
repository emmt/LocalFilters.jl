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

Base.IndexStyle(A::Indices) = IndexStyle(typeof(A))
Base.IndexStyle(::Type{<:Indices{S}}) where {S} = S()

@inline (::Indices{IndexLinear})(A::AbstractArray) = Base.OneTo{Int}(length(A))
@inline (::Indices{IndexLinear})(A::AbstractVector) = to_int(Base.axes1(A))
@inline (::Indices{IndexCartesian})(A::AbstractArray) = CartesianIndices(A)

Indices(::S) where {S<:IndexStyle} = Indices{S}()
#Indices(::S, A::AbstractArray) where {S<:IndexStyle} = Indices{S}()
#Indices(::S, A::AbstractArray, B::AbstractArray...) where {S<:IndexStyle} =
#    Indices{S}()

Indices(A::AbstractVector) = Indices(IndexStyle(A))
Indices(A::AbstractVector, B::AbstractVector...) = Indices(IndexStyle(A, B...))
Indices(A::AbstractArray) = Indices{IndexCartesian}()
Indices(A::AbstractArray, B::AbstractArray...) = Indices{IndexCartesian}()

"""
    LocalFilters.kernel_range(x)

yields an `Int`-valued index range based on `x`.  If `x` is an integer, a
centered range of this length is returned (for even lengths, the same
conventions as in `fftshift` are used).  Otherwise, `x` must be a an integer
valued range.

    LocalFilters.kernel_range(start, [step = 1,] stop)

yields an `Int`-valued index range similar to `start:step:stop`.  Beware that
non-unit steps may not be fully supported in `LocalFilters`.

See [`LocalFilters.kernel`](@ref),
[`LocalFilters.first_centered_index`](@ref), [`LocalFilters.centered`](@ref),
and alias [`LocalFilters.WindowRange`](@ref).

"""
kernel_range(rng::IntegerRange) = to_int(rng)

function kernel_range(len::Integer)
    len ≥ 0 || throw(ArgumentError("invalid range length"))
    n = to_int(len)
    start = first_centered_index(n)
    return kernel_range(start, n - 1 + start)
end

kernel_range(start::Integer, stop::Integer) = to_int(start):to_int(stop)

kernel_range(start::Integer, step::Integer, stop::Integer) =
    to_int(start):to_int(step):to_int(stop)

"""
    LocalFilters.kernel([Dims{N},] args...)

yields an `N`-dimensional abstract array builds from `args...` and which can be
used as a kernel in local filtering operations.

* If `args...` is composed of `N` integers and/or ranges or if it is an
  `N`-tuple of integers and/or ranges, a uniformly true abstract array is
  returned whose axes are specified by `args...`.  Each integer argument is
  converted in a centered unit range of this length (see
  [`LocalFilters.kernel_range`](@ref)).

* If `Dims{N}` is provided and `args...` is a single integer or range, it is
  interpreted as being the same for all dimensions.

* If `args...` is a pair of Cartesian indices, say `I_first` and `I_last`, a
  uniformly true abstract array is returned whose first and last indices are
  given by `args...`.

* If `args...` is a Cartesian range, a uniformly true abstract array is
  returned whose axes are given by `args...`.

* If `args...` is an array of any other type than a Cartesian range,
  it is returned unchanged.

Optional leading argument `Dims{N}` can be specified to assert the number of
dimensions of the result or to provide the number of dimensions when it cannot
be guessed from the arguments.  For example, when `args...` is a single integer
length or range which should be interpreted as being the same for all
dimensions.

See also [`LocalFilters.kernel_range`](@ref) and
[`LocalFilters.cartesian_limits`](@ref).

"""
kernel(x::Axis...) = box(x)
kernel(::Type{Dims{N}}, x::Axis...) where {N} = Box{N}(x...)

kernel(x::Tuple{Vararg{Axis}}) = Box(x)
kernel(::Type{Dims{N}}, x::NTuple{N,Axis}) where {N} = Box(x)

kernel(x::CartesianIndices) = Box(x)
kernel(::Type{Dims{N}}, x::CartesianIndices{N}) where {N} = Box(x)

kernel(x::AbstractArray) = x
kernel(::Type{Dims{N}}, x::AbstractArray{<:Any,N}) where {N} = x

kernel(x::NTuple{2,CartesianIndex{N}}) where {N} = Box(x...)
kernel(::Type{Dims{N}}, x::NTuple{2,CartesianIndex{N}}) where {N} = Box(x...)

kernel(a::CartesianIndex{N}, b::CartesianIndex{N}) where {N} = Box(a, b)
kernel(::Type{Dims{N}}, a::CartesianIndex{N}, b::CartesianIndex{N}) where {N} =
    Box(a, b)
kernel(::Type{Dims{N}}, x...) where {N} =
    throw(ArgumentError(
        "cannot create an instance of CartesianIndices{$N} for argument of type $(typeof(x))"))

# Implement abstract array interface for Box objects which are uniformly true
# arrays.
Base.length(A::Box) = prod(size(A))
Base.axes(A::Box) = ranges(CartesianIndices(A))
Base.size(A::Box) = map(length, axes(A))
Base.CartesianIndices(A::Box) = getfield(A, :inds)
Base.IndexStyle(::Type{<:Box}) = IndexLinear()
@inline Base.getindex(A::Box, I...) = true # FIXME: check indices

# `Box` constructors, also see the `kernel` method.
Box{N}(x::Axis) where {N} = Box(replicate(NTuple{N}, kernel_range(x)))
Box{N}(x::Axis...) where {N} = Box{N}(x)
Box{N}(x::NTuple{N,Axis}) where {N} = Box(x)
Box(x::Axis...) = Box(x)
Box(x::NTuple{N,Axis}) where {N} = Box(CartesianIndices(map(kernel_range, x)))
Box{N}(a::CartesianIndex{N}, b::CartesianIndex{N}) where {N} = Box(a, b)
Box(a::CartesianIndex{N}, b::CartesianIndex{N}) where {N} =
    Box(CartesianIndices(map(kernel_range, Tuple(a), Tuple(b))))
Box{N}(A::AbstractArray{N}) where {N} = Box(A)
Box(A::AbstractArray) = Box(CartesianIndices(A))
Box(A::Box) = A

"""
    LocalFilters.subset(A_inds, ord, B_inds, i) -> J

yields the subset `J` of all indices `j` such that:

* `A[j]` and `B[j-i]` are in-bounds if `ord = Forward`;

* `A[j]` and `B[i-j]` are in-bounds if `ord = Reverse`;

with `A` and `B` any arrays whose index ranges are given by `A_inds` and
`B_inds`.

"""
@inline function subset(A::CartesianIndices{N},
                        ::typeof(Forward),
                        B::CartesianIndices{N},
                        I::CartesianIndex{N}) where {N}
    return @range A ∩ (I + B)
end

@inline function subset(A::CartesianIndices{N},
                        ::typeof(Reverse),
                        B::CartesianIndices{N},
                        I::CartesianIndex{N}) where {N}
    return @range A ∩ (I - B)
end

@inline function subset(A::IntegerRange,
                        ::typeof(Forward),
                        B::IntegerRange,
                        I::Integer)
    return @range A ∩ (I + B)
end

@inline function subset(A::IntegerRange,
                        ::typeof(Reverse),
                        B::IntegerRange,
                        I::Integer)
    return @range A ∩ (I - B)
end

"""
    LocalFilters.getval(ord, B, i, j)

yields the value of the kernel `B` according to ordering `ord` and for indices
`i` and `j` respectively in the destination and in the source.  If `B` is a
Cartesian window, `true` is returned; otherwise `B[j-i]` is returned if `ord =
Forward` and `B[i-j]` is returned if `ord = Reverse`.

"""
@inline function getval(::typeof(Forward), B::Box, i, j)
    # FIXME: check bounds (and idem for others)
    return true
end

@inline function getval(::typeof(Reverse), B::Box, i, j)
    return true
end

@inline @propagate_inbounds function getval(::typeof(Forward),
                                            B::AbstractArray, i, j)
    return B[j-i]
end

@inline @propagate_inbounds function getval(::typeof(Reverse),
                                            B::AbstractArray, i, j)
    return B[i-j]
end

"""
   LocalFilters.centered(A) -> B

yields an abstract array `B` sharing the entries of array `A` but with offsets
on indices so that the axes of `B` are *centered* (for even dimension lengths,
the same conventions as in `fftshift` are used).

This method is purposely not exported because it could introduce some
confusions.  For example `OffsetArrays.centered` is similar but has a slightly
different semantic.

Argument `A` can also be an index range (linear or Cartesian), in which case
a centered index range of same size is returned.

See [`LocalFilters.kernel_range`](@ref),
[`LocalFilters.first_centered_index`](@ref).

"""
centered(A::AbstractArray) =
    OffsetArray(A, map(x -> first_centered_index(x) - 1, size(A)))
centered(A::OffsetArray) = centered(parent(A))
centered(R::CartesianIndices{N}) where {N} =
    CartesianIndices(map(centered, ranges(R)))
centered(R::AbstractUnitRange{<:Integer}) = kernel_range(length(R))
centered(R::IntegerRange) = begin
    abs(step(R)) == 1 || throw(ArgumentError("invalid non-unit step index range"))
    return kernel_range(length(R))
end

"""
   LocalFilters.first_centered_index(len)

yields the first index along a centered dimension of length `len`.  That is,
`-div(len,2)` converetd to `Int`.  For even dimension lengths, this amounts to
using the same conventions as in `fftshift`.

See [`LocalFilters.kernel_range`](@ref) and [`LocalFilters.entered`](@ref).

"""
function first_centered_index(len::Integer)
    len ≥ 0 || throw(ArgumentError("invalid dimension length"))
    n = to_int(len)
    return -(n >> 1)
end

"""
    LocalFilters.replicate(NTuple{N}, val)

yields the `N`-tuple `(val,val,...)`.

    LocalFilters.replicate(NTuple{N,T}, val)

yields the `N`-tuple `(x,x,...)` where `x` is `val` converted to type `T`.

"""
replicate(::Type{NTuple{N}}, val) where {N} = ntuple((x) -> val, Val(N))
replicate(::Type{NTuple{N,T}}, val::T) where {N,T} = ntuple((x) -> val, Val(N))
replicate(::Type{NTuple{N,T}}, val) where {N,T} =
    replicate(NTuple{N,T}, convert(T, val)::T)

"""
    limits(T::DataType) -> typemin(T), typemax(T)

yields the infimum and supremum of a type `T`.

"""
limits(T::Type) = (typemin(T), typemax(T))

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

"""
    result_eltype(f, A[, B]) -> T

yields the element type `T` of the result of applying function `f` to
source `A`, optionally with kernel/neighborhood `B`.

"""
result_eltype(::typeof(+), ::AbstractArray{T}) where {T} = T
result_eltype(::typeof(+), ::AbstractArray{T}) where {T<:Integer} =
    # Widen type for integers smaller than standard ones.
    (sizeof(T) < sizeof(Int) ? widen(T) : T)

"""
    LocalFilters.is_morpho_math_box(R)

yields whether structuring element `R` has the same effect as an
hyperrectangular box for mathematical morphology operations.  This may be used
to use fast separable versions of mathematical morphology operations like the
van Herk-Gil-Werman algorithm.

"""
is_morpho_math_box(::CartesianIndices) = true
is_morpho_math_box(R::AbstractArray{Bool}) = all(R)
is_morpho_math_box(R::AbstractArray{<:AbstractFloat}) = all(iszero, R)

"""
    strel(T, A)

yields a *structuring element* suitable for mathematical morphology operations.
The result is an array whose elements have type `T` (which can be `Bool` or a
floating-point type).  Argument `A` can be a hyperrectangular Cartesian sliding
window or an array with boolean elements.

If `T` is a floating-point type, then the result is a so-called *flat*
structuring element whose coefficients are `zero(T)` inside the shape defined
by `A` and `-T(Inf)` elsewhere.

"""
strel(::Type{Bool}, A::AbstractArray{Bool}) = A
strel(::Type{T}, A::AbstractArray{Bool}) where {T<:AbstractFloat} =
    map(x -> ifelse(x, zero(T), -T(Inf)), A)
strel(::Type{Bool}, A::CartesianIndices) =
    OffsetArray(UniformArray(true, size(A)), ranges(A))
strel(T::Type{<:AbstractFloat}, A::CartesianIndices) =
    OffsetArray(UniformArray(zero(T), size(A)), ranges(A))

"""
    LocalFilters.ball(Dims{N}, r)

yields a boolean mask which is a `N`-dimensional array with all dimensions odd
and equal and set to true where position is inside a `N`-dimensional ball of
radius `r`.

To have a mask with centered index ranges, call:

    LocalFilters.centered(LocalFilters.ball(Dims{N}, r))

"""
function ball(::Type{Dims{N}}, radius::Real) where {N}
    b = radius + 1/2
    r = ceil(Int, b - one(b))
    dim = 2*r + 1
    dims = ntuple(d->dim, Val(N))
    arr = Array{Bool}(undef, dims)
    bb = b^2
    qmax = ceil(Int, bb - one(bb))
    _ball!(arr, 0, qmax, r, 1:dim, tail(dims))
    return arr
end

@deprecate ball(N::Integer, radius::Real) ball(Dims{to_int(N)}, radius) false

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
