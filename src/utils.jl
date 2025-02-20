#
# utils.jl --
#
# Useful methods for local filters.
#
#-----------------------------------------------------------------------------------------
#
# This file is part of the `LocalFilters.jl` package licensed under the MIT "Expat"
# License.
#
# Copyright (c) 2017-2025, Éric Thiébaut.
#

# Constructors for `Indices`. Use linear indexing for vectors, Cartesian indices
# otherwise.
Indices(A::AbstractVector) = Indices{IndexLinear}()
Indices(A::AbstractVector, B::AbstractVector...) = Indices{IndexLinear}()
Indices(A::AbstractArray) = Indices{IndexCartesian}()
Indices(A::AbstractArray{<:Any,N}, B::AbstractArray{<:Any,N}...) where {N} =
    Indices{IndexCartesian}()

Base.IndexStyle(A::Indices) = IndexStyle(typeof(A))
Base.IndexStyle(::Type{<:Indices{S}}) where {S} = S()

# `Indices` objects can be called like `eachindex` to yield the indices of an array (see
# `abstractarray.jl`).
@inline (::Indices{IndexLinear})(A::AbstractArray) = to_axis(length(A))
@inline (::Indices{IndexLinear})(A::AbstractVector) = to_axis(Base.axes1(A))
@inline (::Indices{IndexCartesian})(A::AbstractArray) = CartesianIndices(A)
@inline function (I::Indices)(A::AbstractArray, B::AbstractArray...)
    have_same_axes(A, B...)
    return I(A)
end

to_axis(len::Integer) = Base.OneTo{Int}(len)
to_axis(rng::AbstractUnitRange{Int}) = rng
to_axis(rng::AbstractUnitRange{<:Integer}) = convert_eltype(Int, rng)
to_axis(rng::AbstractRange{<:Integer}) =
    isone(step(rng)) ? convert_eltype(Int, rng) : throw(ArgumentError(
        "invalid non-unit step ($(step(rng))) array axis"))

"""
    LocalFilters.have_same_axes(A...)

throws an exception if not all arrays `A...` have the same axes.

"""
@inline have_same_axes(A::AbstractArray, B::AbstractArray...) =
    have_same_axes(Bool, A, B...) ? nothing : throw(DimensionMismatch(
        "arrays must have the same axes"))

"""
    LocalFilters.have_same_axes(Bool, A...) -> bool

yields whether all arrays `A...` have the same axes.

"""
have_same_axes(::Type{Bool}, A::AbstractArray) = true
have_same_axes(::Type{Bool}, A::AbstractArray, B::AbstractArray...) = false
@inline function have_same_axes(::Type{Bool}, A::AbstractArray{<:Any,N},
                                B::AbstractArray{<:Any,N}...) where {N}
    return _have_same_axes(axes(A), B...)
end

@inline _have_same_axes(I::Axes) = true
@inline _have_same_axes(I::Axes, A::AbstractArray) = axes(A) == I
@inline _have_same_axes(I::Axes, A::AbstractArray, B::AbstractArray...) =
     _have_same_axes(I, A) && _have_same_axes(I, B...)

"""
    kernel([Dims{N},] args...)

yields an `N`-dimensional abstract array built from `args...` and which can be used as a
kernel in local filtering operations.

* If `args...` is composed of `N` integers and/or ranges or if it is an `N`-tuple of
  integers and/or ranges, a uniformly true abstract array is returned whose axes are
  specified by `args...`. Each integer argument is converted in a centered unit range of
  this length (see [`LocalFilters.kernel_range`](@ref)).

* If `Dims{N}` is provided and `args...` is a single integer or range, it is interpreted
  as being the same for all dimensions of an `N`-dimensional kernel. For example,
  `kernel(Dims{3},5)` yields a 3-dimensional uniformly true array with index range `-2:2`
  in every dimension.

* If `args...` is 2 Cartesian indices or a 2-tuple of Cartesian indices, say `I_first` and
  `I_last`, a uniformly true abstract array is returned whose first and last indices are
  `I_first` and `I_last`.

* If `args...` is a Cartesian range, say `R::CartesianIndices{N}`, a uniformly true
  abstract array is returned whose axes are given by `R`.

* If `args...` is an abstract array of any other type than an instance of
  `CartesianIndices`, it is returned unchanged.

Optional leading argument `Dims{N}` can be specified to assert the number of dimensions of
the result or to provide the number of dimensions when it cannot be inferred from the
arguments. For example, when `args...` is a single integer length or range which should be
interpreted as being the same for all dimensions.

"""
kernel(::Type{Dims{N}}, A::AbstractArray{<:Any,N}) where {N} = kernel(A)
kernel(A::AbstractArray) = A

kernel(::Type{Dims{N}}, x::LocalAxis) where {N} = kernel(replicate(NTuple{N}, kernel_range(x)))

kernel(::Type{Dims{N}}, args::Vararg{LocalAxis,N}) where {N} = kernel(args)
kernel(::Type{Dims{N}}, args::NTuple{N,LocalAxis}) where {N} = kernel(args)
kernel(args::LocalAxis...) = kernel(args)
kernel(args::Tuple{Vararg{LocalAxis}}) = FastUniformArray(true, map(kernel_range, args))

kernel(::Type{Dims{N}}, inds::CartesianIndices{N}) where {N} = kernel(inds)
kernel(inds::CartesianIndices) = kernel(ranges(inds))

kernel(::Type{Dims{N}}, inds::NTuple{2,CartesianIndex{N}}) where {N} = kernel(inds)
kernel(::Type{Dims{N}}, inds::Vararg{CartesianIndex{N},2}) where {N} = kernel(inds)
kernel(inds::NTuple{2,CartesianIndex{N}}) where {N} = kernel(inds...)
kernel(a::CartesianIndex{N}, b::CartesianIndex{N}) where {N} =
    FastUniformArray(true, map(kernel_range, Tuple(a), Tuple(b)))

kernel(::Type{Dims{0}}) = kernel()

# Error catcher.
kernel(::Type{Dims{N}}) where {N} = throw(ArgumentError(
    "cannot create a $N-dimensional kernel with no additional argument(s)"))
kernel(::Type{Dims{N}}, args...) where {N} = throw(ArgumentError(
    "cannot create a $N-dimensional kernel for argument(s) of type $(typeof(args))"))

"""
    LocalFilters.kernel_range(start, stop)
    LocalFilters.kernel_range(rng)
    LocalFilters.kernel_range(dim)

yield an `Int`-valued unit range based on first and last indices `start` and `stop`, unit
range `rng`, or dimension length `dim`. In the case of a given dimension length, a
centered range of this length is returned (for even lengths, the same conventions as in
`fftshift` are used).

See [`LocalFilters.kernel`](@ref) and [`LocalFilters.centered`](@ref).

"""
kernel_range(start::Integer, stop::Integer) = as(Int, start):as(Int, stop)
kernel_range(rng::AbstractUnitRange{Int}) = rng
kernel_range(rng::AbstractUnitRange{<:Integer}) = convert_eltype(Int, rng)
function kernel_range(rng::AbstractRange{<:Integer})
    isone(step(rng)) || throw(ArgumentError("not a unit step range, got step of $(step(rng))"))
    return kernel_range(first(rng), last(rng))
end
function kernel_range(dim::Integer)
    dim ≥ zero(dim) || throw(ArgumentError("dimension must be ≥ 0, got $dim"))
    dim = as(Int, dim)
    start = -(dim ÷ 2)
    stop = dim - 1 + start
    return kernel_range(start, stop)
end

"""
    reverse_kernel(A) -> B

yields an array `B` such that `B[i] = A[-i]` holds for all indices `i` such that
`-i` is a valid index in `A`.

"""
reverse_kernel(A::AbstractArray) = OffsetArray(reverse(A), reverse_axes(A))
reverse_kernel(A::OffsetArray) = OffsetArray(reverse(parent(A)), reverse_axes(A))
for cls in (:MutableUniformArray, :UniformArray, :FastUniformArray)
    @eval begin
        # Rebuild a uniform array of same type and value but reversed axes.
        reverse_kernel(A::$cls{T}) where {T} =
            $cls{T}(StructuredArrays.value(A), reverse_axes(A))
    end
end

reverse_axes(A::AbstractArray) = map(reverse_axis, axes(A))
function reverse_axis(I::AbstractUnitRange{<:Integer})
    I_first, I_last = EasyRanges.first_last(I)
    return (-I_last):(-I_first)
end

"""
   LocalFilters.centered(A) -> B

yields an abstract array `B` sharing the entries of array `A` but with offsets on indices
so that the axes of `B` are *centered* (for even dimension lengths, the same conventions
as in `fftshift` are used).

This method is purposely not exported because it could introduce some confusions. For
example `OffsetArrays.centered` is similar but has a slightly different semantic.

Argument `A` can also be an index range (linear or Cartesian), in which case a centered
index range of same size is returned.

See [`LocalFilters.kernel_range`](@ref).

"""
centered(A::AbstractArray) = OffsetArray(A, map(kernel_range, size(A)))
centered(A::OffsetArray) = centered(parent(A))
centered(R::CartesianIndices) = CartesianIndices(map(centered, ranges(R)))
centered(R::AbstractUnitRange{<:Integer}) = kernel_range(length(R))
centered(R::IntegerRange) = begin
    abs(step(R)) == 1 || throw(ArgumentError("invalid non-unit step range"))
    return kernel_range(length(R))
end
for cls in (:MutableUniformArray, :UniformArray, :FastUniformArray)
    @eval begin
        # Rebuild a uniform array of same type and value but reversed axes.
        centered(A::$cls{T}) where {T} =
            $cls{T}(StructuredArrays.value(A), map(kernel_range, size(A)))
    end
end

"""
    FORWARD_FILTER

is an exported constant object used to indicate *forward* ordering of indices in local
filter operations. It can be called as:

    FORWARD_FILTER(i, j) -> j - i

to yield the index in the filter kernel. See also [`REVERSE_FILTER`](@ref) for *reverse*
ordering and [`LocalFilters.localindices`](@ref) for building a range of valid indices
`j`.

"""
const FORWARD_FILTER = ForwardFilterOrdering()

"""
    REVERSE_FILTER

is an exported constant object used to indicate *reverse* ordering of indices in local
filter operations. It can be called as:

    REVERSE_FILTER(i, j) -> i - j

to yield the index in the filter kernel. See also [`FORWARD_FILTER`](@ref) for *forward*
ordering and [`LocalFilters.localindices`](@ref) for building a range of valid indices
`j`.

"""
const REVERSE_FILTER = ReverseFilterOrdering()

Base.reverse(::ForwardFilterOrdering) = REVERSE_FILTER
Base.reverse(::ReverseFilterOrdering) = FORWARD_FILTER

@inline (::ForwardFilterOrdering)(i::Int, j::Int) = j - i
@inline (::ForwardFilterOrdering)(i::Integer, j::Integer) = Int(j) - Int(i)
@inline (::ForwardFilterOrdering)(i::T, j::T) where {N,T<:CartesianIndex{N}} = j - i

@inline (::ReverseFilterOrdering)(i::Int, j::Int) = i - j
@inline (::ReverseFilterOrdering)(i::Integer, j::Integer) = Int(i) - Int(j)
@inline (::ReverseFilterOrdering)(i::T, j::T) where {N,T<:CartesianIndex{N}} = i - j

"""
    LocalFilters.localindices(A_inds, ord, B_inds, i) -> J

yields the subset `J` of all indices `j` such that:

- `A[j]` and `B[ord(i,j)] = B[j-i]` are in-bounds if `ord = FORWARD_FILTER`;

- `A[j]` and `B[ord(i,j)] = B[i-j]` are in-bounds if `ord = REVERSE_FILTER`;

with `A` and `B` arrays whose index ranges are given by `A_inds` and `B_inds`. To make the
code agnostic to the ordering, use `A[i]` and `B[ord(i,j)]` to retrieve the values in `A`
and `B`.

Index ranges `A_inds` and `B_inds` and index `i` must be of the same kind:

- linear index ranges for `A_inds` and `B_inds` and linear index for `i`;

- Cartesian index ranges for `A_inds` and `B_inds` and Cartesian index for `i` of same
  number of dimensions.

Constructor [`LocalFilters.Indices`](@ref) may by used to build a callable object that
yields the index ranges of `A` and `B` in a consistent way:

    indices = LocalFilters.Indices(A, B)
    A_inds = indices(A)
    B_inds = indices(B)

"""
@inline function localindices(A::IntegerRange,
                              ::ForwardFilterOrdering,
                              B::IntegerRange,
                              I::Integer)
    return @range A ∩ (I + B)
end

@inline function localindices(A::CartesianIndices{N},
                              ::ForwardFilterOrdering,
                              B::CartesianIndices{N},
                              I::CartesianIndex{N}) where {N}
    return @range A ∩ (I + B)
end

@inline function localindices(A::IntegerRange,
                              ::ReverseFilterOrdering,
                              B::IntegerRange,
                              I::Integer)
    return @range A ∩ (I - B)
end

@inline function localindices(A::CartesianIndices{N},
                              ::ReverseFilterOrdering,
                              B::CartesianIndices{N},
                              I::CartesianIndex{N}) where {N}
    return @range A ∩ (I - B)
end

"""
    LocalFilters.replicate(NTuple{N}, val)

yields the `N`-tuple `(val, val, ...)`.

    LocalFilters.replicate(NTuple{N,T}, val)

yields the `N`-tuple `(x, x,...)` where `x` is `val` converted to type `T`.

See [`LocalFilters.Yields`](@ref).

"""
replicate(::Type{NTuple{0}}, val) = ()
replicate(::Type{NTuple{N}}, val) where {N} = ntuple(Yields(val), Val(N))
replicate(::Type{NTuple{N,T}}, val) where {N,T} = ntuple(Yields{T}(val), Val(N))

"""
    limits(T::DataType) -> typemin(T), typemax(T)

yields the infimum and supremum of a type `T`.

"""
limits(T::Type) = (typemin(T), typemax(T))

"""
    LocalFilters.is_morpho_math_box(B)

yields whether structuring element `B` has the same effect as an hyper-rectangular box for
mathematical morphology operations. This may be used to use fast separable versions of
mathematical morphology operations like the van Herk-Gil-Werman algorithm.

"""
is_morpho_math_box(::Box) = true
is_morpho_math_box(B::AbstractArray{Bool}) = all(B)
is_morpho_math_box(B::AbstractArray{<:AbstractFloat}) = all(iszero, B)
is_morpho_math_box(::CartesianIndices) =
    error("Cartesian range must be converted to a kernel")

"""
    strel(T, A)

yields a *structuring element* suitable for mathematical morphology operations. The result
is an array whose elements have type `T` (which can be `Bool` or a floating-point type).
Argument `A` can be a hyper-rectangular Cartesian sliding window or an array with boolean
elements.

If `T` is a floating-point type, then the result is a so-called *flat* structuring element
whose coefficients are `zero(T)` inside the shape defined by `A` and `-T(Inf)` elsewhere.

"""
strel(::Type{Bool}, A::AbstractArray{Bool}) = A
strel(::Type{Bool}, A::CartesianIndices) = FastUniformArray(true, ranges(A))
strel(T::Type{<:AbstractFloat}, A::CartesianIndices) = FastUniformArray(zero(T), ranges(A))

function strel(::Type{T}, A::AbstractArray{Bool}) where {T<:AbstractFloat}
    flat(::Type{T}, x::Bool) where {T<:AbstractFloat} = ifelse(x, zero(T), -T(Inf))
    map(flat, A)
end

"""
    LocalFilters.nearest(T, x)

converts value `x` to the nearest value of type `T`. By default, the result is given by
`convert(T,x)` unless `T` is floating-point type and `x` is integer in which case the
result is given by `round(T,x)`.

This method may be extended for foreign types to implement other conversions.

"""
nearest(::Type{T}, x::T) where {T} = x
nearest(::Type{T}, x::Any) where {T} = convert(T, x)
nearest(::Type{T}, x::Integer) where {T<:AbstractFloat} = round(T, x)

"""
    LocalFilters.ball(Dims{N}, r)

yields a boolean mask which is a `N`-dimensional array with all dimensions odd and equal
and set to true where position is inside a `N`-dimensional ball of radius `r`.

To have a mask with centered index ranges, call:

    LocalFilters.centered(LocalFilters.ball(Dims{N}, r))

"""
function ball(::Type{Dims{N}}, radius::Real) where {N}
    b = radius + 1/2
    r = ceil(Int, b - one(b))
    dim = 2*r + 1
    dims = replicate(Dims{N}, dim)
    b² = b*b
    qmax = ceil(Int, b² - one(b²))
    return _ball!(Array{Bool}(undef, dims), 0, qmax, r, 1:dim, tail(dims))
end

@deprecate ball(N::Integer, radius::Real) ball(Dims{as(Int, N)}, radius) false

@inline function _ball!(arr::AbstractArray{Bool,N},
                        q::Int, qmax::Int, r::Int,
                        range::AbstractUnitRange{Int},
                        dims::Tuple{Int}, I::Int...) where {N}
    # Iterate over coordinates along dimension.
    nextdims = tail(dims)
    x = -r
    for i in range
        _ball!(arr, q + x*x, qmax, r, range, nextdims, I..., i)
        x += 1
    end
    return arr
end

@inline function _ball!(arr::AbstractArray{Bool,N},
                        q::Int, qmax::Int, r::Int,
                        range::AbstractUnitRange{Int},
                        ::Tuple{}, I::Int...) where {N}
    # Iterate over coordinates along last dimension. This ends the recursion.
    x = -r
    for i in range
        arr[I...,i] = (q + x*x ≤ qmax)
        x += 1
    end
    return arr
end

# Boundary conditions.
#
# NOTE: Remember that the linear index range of a vector `V` are given by `axes(V,1)`
#       while the linear index range of a multi-dimensional array `A` is given by
#       `1:length(A)` (in fact `Base.OneTo(lenght(A))`).
FlatBoundaries(A::AbstractVector) = FlatBoundaries(Base.axes1(A))
FlatBoundaries(A::AbstractArray, d::Integer) = FlatBoundaries(axes(A,d))
FlatBoundaries(A::AbstractArray) = FlatBoundaries(CartesianIndices(A))

indices(B::FlatBoundaries) = getfield(B, :indices)

(B::FlatBoundaries{<:AbstractUnitRange{Int}})(i::Int) = clamp_to_range(i, indices(B))
(B::FlatBoundaries{<:AbstractUnitRange{Int}})(i::Integer) =
    clamp_to_range(Int(i), indices(B))
(B::FlatBoundaries{<:CartesianUnitRange{N}})(i::CartesianIndex{N}) where {N} =
    CartesianIndex(map(clamp_to_range, Tuple(i), ranges(indices(B))))

# Clamp to range. Range must not be empty.
clamp_to_range(idx::T, rng::AbstractUnitRange{T}) where {T<:Integer} =
    clamp(idx, first(rng), last(rng))
