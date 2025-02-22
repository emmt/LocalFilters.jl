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

"""
    I = LocalFilters.Indices(A::AbstractArray...)

builds a callable object, `I`, that can be used to produce ranges of indices for each of
the arrays `A...`. These ranges will all be of the same type: linear index ranges, if all
arrays `A...` are vectors, Cartesian index ranges otherwise.

`I` is similar to the `eachindex` method but is specialized for a style of indexing, it
can be called as `I(B...)` to yield a suitable index range to access all the entries of
array(s) `B...` which are any number of the `A...` specified when building `I`. If `B...`
consists in several arrays, they must have the same axes otherwise `I(B...)` will throw a
`DimensionMismatch` exception.

Call:

   I = LocalFilters.Indices{S}()

with `S = IndexLinear` or `S = IndexCartesian` to specifically choose the indexing style.

""" Indices
@public Indices
Indices(A::AbstractVector) = Indices{IndexLinear}()
Indices(A::AbstractVector, B::AbstractVector...) = Indices{IndexLinear}()
Indices(A::AbstractArray) = Indices{IndexCartesian}()
Indices(A::AbstractArray{<:Any,N}, B::AbstractArray{<:Any,N}...) where {N} =
    Indices{IndexCartesian}()

Base.IndexStyle(A::Indices) = IndexStyle(typeof(A))
Base.IndexStyle(::Type{<:Indices{S}}) where {S} = S()

# `Indices` objects can be called like `eachindex` to yield the indices of an array (see
# `abstractarray.jl`).
@inline (::Indices{IndexLinear})(A::AbstractArray) = OneTo{Int}(length(A))
@inline (::Indices{IndexLinear})(A::AbstractVector) = AbstractUnitRange{Int}(Base.axes1(A))
@inline (::Indices{IndexCartesian})(A::AbstractArray) = CartesianIndices(A)
@inline (I::Indices)(A::AbstractArray, B::AbstractArray...) = (check_axes(A, B...); I(A))

"""
    LocalFilters.check_axes([I,] A...)

throws an exception if not all arrays `A...` have the same axes, or all have axes `I` if
specified.

""" check_axes
@public check_axes
@inline check_axes(A::AbstractArray...) =
    check_axes(Bool, A...) ? nothing : throw(DimensionMismatch(
        "arrays must have the same axes"))
@inline check_axes(I::ArrayAxes, A::AbstractArray...) =
    check_axes(Bool, I, A...) ? nothing : throw(DimensionMismatch(
        "arrays must have the given axes"))

"""
    LocalFilters.check_axes(Bool, [I,] A...)

yields whether all arrays `A...` have the same axes, or all have axes `I` if specified.

"""
check_axes(::Type{Bool}, A::AbstractArray...) = false
check_axes(::Type{Bool}, A::AbstractArray) = true
@inline check_axes(::Type{Bool}, A::AbstractArray{<:Any,N}, B::AbstractArray{<:Any,N}...) where {N} =
    check_axes(Bool, axes(A), B...)

@inline check_axes(::Type{Bool}, I::ArrayAxes,    A::AbstractArray...) = false
@inline check_axes(::Type{Bool}, I::ArrayAxes{N}, A::AbstractArray{<:Any,N}) where {N} = axes(A) == I
@inline check_axes(::Type{Bool}, I::ArrayAxes{N}, A::AbstractArray{<:Any,N}, B::AbstractArray{<:Any,N}...) where {N} =
    axes(A) == I && check_axes(Bool, B...)

"""
    kernel([Dims{N},] args...)

yields an `N`-dimensional abstract array built from `args...` and which can be used as a
kernel in local filtering operations. The kernel is also called a *neighborhood* when its
element type is `Bool`.

* If `args...` is composed of `N` integers and/or ranges or if it is an `N`-tuple of
  integers and/or ranges, a uniformly true abstract array is returned whose axes are
  specified by `args...`. Each integer argument is converted in a centered unit range of
  this length (see [`LocalFilters.kernel_range`](@ref)).

* If `Dims{N}` is provided and `args...` is a single integer or range, it is interpreted
  as being the same for all dimensions of an `N`-dimensional kernel. For example,
  `kernel(Dims{3},5)` yields a 3-dimensional uniformly true array with index range `-2:2`
  in every dimension.

* If `args...` is two Cartesian indices or a 2-tuple of Cartesian indices, say `start` and
  `stop`, a uniformly true abstract array is returned whose first and last indices are
  `start` and `stop`.

* If `args...` is a Cartesian range, say `R::CartesianIndices{N}`, a uniformly true
  abstract array is returned whose axes are given by `R`.

* If `args...` is an abstract array of any other type than an instance of
  `CartesianIndices`, it is returned unchanged.

Optional leading argument `Dims{N}` can be specified to assert the number of dimensions of
the result or to provide the number of dimensions when it cannot be inferred from the
arguments. For example, when `args...` is a single integer length or range which should be
interpreted as being the same for all dimensions.

See also [`LocalFilters.strel`](@ref), [`LocalFilters.ball`](@ref),
[`LocalFilters.kernel_range`](@ref), [`LocalFilters.reverse_kernel`](@ref), and
[`LocalFilters.cartesian_limits`](@ref).

"""
kernel(::Type{Dims{N}}, arg::Axis) where {N} = kernel(Dims{N}, kernel_range(arg))
kernel(::Type{Dims{N}}, rng::AbstractUnitRange{Int}) where {N} =
    kernel(ntuple(Returns(rng), Val(N)))
kernel(::Type{Dims{N}}, inds::Vararg{Axis,N}) where {N} = kernel(inds)
kernel(::Type{Dims{N}}, inds::NTuple{N,Axis}) where {N} = kernel(inds)
kernel(::Type{Dims{N}}, R::CartesianIndices{N}) where {N} = kernel(R)
kernel(::Type{Dims{N}}, A::AbstractArray{<:Any,N}) where {N} = kernel(A)
kernel(::Type{Dims{N}}, inds::NTuple{2,CartesianIndex{N}}) where {N} = kernel(inds)
kernel(::Type{Dims{N}}, start::CartesianIndex{N}, stop::CartesianIndex{N}) where {N} =
    kernel(start, stop)

kernel(inds::Axis...) = kernel(inds)
kernel(inds::Tuple{Vararg{Axis}}) = kernel(map(kernel_range, inds))
kernel(inds::Tuple{Vararg{AbstractUnitRange{Int}}}) = box(inds)
kernel(R::CartesianIndices) = kernel(R.indices)
kernel(A::AbstractArray) = A
kernel(inds::NTuple{2,CartesianIndex{N}}) where {N} = kernel(inds...)
kernel(start::CartesianIndex{N}, stop::CartesianIndex{N}) where {N} =
    kernel(map(kernel_range, Tuple(start), Tuple(stop)))

# Error catcher.
@noinline function kernel(::Type{Dims{N}}, args...) where {N}
    len = length(args)
    msg = "cannot create a $N-dimensional kernel from $len additional argument"
    if len == 1
        msg *= " of type `$(typeof(args[1]))`"
    elseif len == 2
        msg *= "s of types `$(typeof(args[1]))` and `$(typeof(args[2]))`"
    elseif len > 2
        msg *= "s of types `"*join(map(typeof, args), "`, `", "`, and `")*"`"
    end
    throw(ArgumentError(msg))
end

"""
   LocalFilters.centered_offset(len)

yields the index offset along a centered dimension of length `len`. That is,
`-div(Int(len)+2,2)`. For even dimension lengths, this amounts to using the same
conventions as in `fftshift`.

See [`LocalFilters.kernel_range`](@ref) and [`LocalFilters.centered`](@ref).

""" centered_offset
@public centered_offset
function centered_offset(len::Integer)
    len ≥ 0 || throw(ArgumentError("invalid dimension length"))
    return -((Int(len) + 2) >> 1)
end

"""
    LocalFilters.kernel_range([ord=FORWARD_FILTER,] rng::AbstractRange{<:Integer})
    LocalFilters.kernel_range([ord=FORWARD_FILTER,] len::Integer)
    LocalFilters.kernel_range([ord=FORWARD_FILTER,] start::Integer, stop::Integer)

yield an unit-step `Int`-valued index range based on range `rng`, dimension length `len`,
or first and last indices `start` and `stop`. In the case of a given dimension length, a
centered range of this length is returned (for even lengths, the same conventions as in
`fftshift` are used).

If ordering `ord` is specified, the returned range is suitable for this ordering.

See also [`LocalFilters.kernel`](@ref), [`LocalFilters.centered_offset`](@ref), and
[`LocalFilters.centered`](@ref).

""" kernel_range
@public kernel_range
kernel_range(start::Integer, stop::Integer) = unit_range(start, stop)
kernel_range(rng::AbstractRange{<:Integer}) = unit_range(rng)
function kernel_range(len::Integer)
    len = as(Int, len)
    off = centered_offset(len)
    return unit_range(off + 1, off + len)
end

kernel_range(org::FilterOrdering, arg::Axis) = kernel_range(org, kernel_range(arg))
kernel_range(org::FilterOrdering, start::Integer, stop::Integer) =
    kernel_range(org, kernel_range(start, stop))

kernel_range(::ForwardFilterOrdering, rng::AbstractUnitRange{Int}) = rng
kernel_range(::ReverseFilterOrdering, rng::AbstractUnitRange{Int}) = reverse_kernel_axis(rng)

"""
    LocalFilters.unit_range(r::Union{AbstractRange{<:Integer},CartesianIndices})

converts `r` into an `Int`-valued unit step index range. `r` may be a linear or a
Cartesian index range. If `r` is a linear range, the absolute value of its step must be 1.

    LocalFilters.unit_range(start::Integer, stop::Integer)

yields the `Int`-valued unit step range `Int(start):Int(stop)`.

""" unit_range
@public unit_range
unit_range(r::OneTo{Int}) = r
unit_range(r::OneTo{<:Integer}) = OneTo{Int}(length(r))

unit_range(r::AbstractUnitRange{Int}) = r
unit_range(r::AbstractUnitRange{<:Integer}) = unit_range(first(r), last(r))

unit_range(start::Integer, stop::Integer) = unit_range(as(Int, start), as(Int, stop))
unit_range(start::Int, stop::Int) = start:stop

function unit_range(rng::AbstractRange{<:Integer})
    step = Base.step(rng)
    isone(abs(step)) || throw(ArgumentError("invalid non-unit step range"))
    start, stop = as(Int, first(rng)), as(Int, last(rng)) # convert to Int prior to swap
    if step < zero(step)
        start, stop = stop, start
    end
    return unit_range(start, stop)
end

function unit_range(R::CartesianIndices{N}) where {N}
    # Since Julia 1.6, non-unit step Cartesian ranges may be defined.
    if R isa CartesianIndices{N,<:NTuple{N,AbstractUnitRange{Int}}}
        return R
    else
        return CartesianIndices(map(unit_range, R.indices))
    end
end

"""
    B = reverse_kernel(args...)

yields a kernel `B` which is similar to `A = kernel(args...)` but with reversed ordering
in the sense that `B[i] == A[-i]` holds for all indices `i` such that `-i` is a valid
index in `A`. As a consequence, a correlation by `B` yields the same result as a
convolution by `A` and conversely.

See also [`LocalFilters.kernel`](@ref) and [`LocalFilters.strel`](@ref).

"""
reverse_kernel(args...; kwds...) = reverse_kernel(kernel(args...; kwds...)::AbstractArray)
reverse_kernel(::Type{Dims{N}}, A::AbstractArray{N}) where {N} = reverse_kernel(A)
reverse_kernel(R::CartesianIndices) = box(map(reverse_kernel_axis, R.indices))
reverse_kernel(A::AbstractArray) = OffsetArray(reverse(A), map(reverse_kernel_axis, axes(A)))
reverse_kernel(A::OffsetArray) = OffsetArray(reverse(parent(A)), map(reverse_kernel_axis, axes(A)))
for type in (:UniformArray, :MutableUniformArray)
    @eval reverse_kernel(A::$type) =
        $type(StructuredArrays.value(A), map(reverse_kernel_axis, axes(A)))
end
reverse_kernel(A::FastUniformArray{T,N,V}) where {T,N,V} =
    FastUniformArray{T,N,V}(map(reverse_kernel_axis, axes(A)))

reverse_kernel_axis(start::Integer, stop::Integer) =
    unit_range(-as(Int, stop), -as(Int, start)) # convert to Int prior to negate
reverse_kernel_axis(rng::AbstractUnitRange{<:Integer}) =
    reverse_kernel_axis(first(rng), last(rng))
reverse_kernel_axis(rng::AbstractRange{<:Integer}) =
    reverse_kernel_axis(unit_range(rng))

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
@inline function localindices(A::AbstractRange{<:Integer},
                              ::ForwardFilterOrdering,
                              B::AbstractRange{<:Integer},
                              I::Integer)
    return @range A ∩ (I + B)
end

@inline function localindices(A::CartesianIndices{N},
                              ::ForwardFilterOrdering,
                              B::CartesianIndices{N},
                              I::CartesianIndex{N}) where {N}
    return @range A ∩ (I + B)
end

@inline function localindices(A::AbstractRange{<:Integer},
                              ::ReverseFilterOrdering,
                              B::AbstractRange{<:Integer},
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
   LocalFilters.centered(A) -> B

yields an abstract array `B` sharing the entries of array `A` but with offsets on indices
so that the axes of `B` are *centered* (for even dimension lengths, the same conventions
as in `fftshift` are used).

This *public* method is purposely not exported because it could introduce some confusions.
For example `OffsetArrays.centered` is similar but has a slightly different semantic.

Argument `A` can also be an index range (linear or Cartesian), in which case a centered
index range of same size is returned.

See also [`LocalFilters.kernel_range`](@ref), [`LocalFilters.centered_offset`](@ref).

""" centered
@public centered
centered(A::AbstractArray) = OffsetArray(A, map(centered_offset, size(A)))
centered(A::OffsetArray) = centered(parent(A))
for type in (:UniformArray, :FastUniformArray, :MutableUniformArray)
    @eval centered(A::$type) =
        $type(StructuredArrays.value(A), map(kernel_range, axes(A)))
end

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

""" is_morpho_math_box
@public is_morpho_math_box
is_morpho_math_box(A::Box) = true
is_morpho_math_box(A::AbstractArray{Bool}) = all(A)
is_morpho_math_box(A::AbstractArray{<:AbstractFloat}) = all(iszero, A)
is_morpho_math_box(R::CartesianIndices) =
    error("Cartesian range must be converted to a kernel")

"""
    box(args...) -> B::Box

yields an hyper-rectangular box for mathematical morphology operations with the same axes
or indices as its argument(s).

See also [`LocalFilters.kernel`](@ref) and [`LocalFilters.strel`](@ref).

""" box
@public box
box(A::Box) = A
box(A::AbstractArray) = box(axes(A))
box(R::CartesianIndices) = box(R.indices)
box(inds::Axis...) = box(inds)
box(inds::NTuple{N,Axis}) where {N} = strel(Bool, inds)

"""
    strel(T, A)

yields a *structuring element* suitable for mathematical morphology operations. The result
is an array whose elements have type `T` (which can be `Bool` or a floating-point type).
Argument `A` can be a hyper-rectangular Cartesian sliding window or an array with boolean
elements.

If `T` is a floating-point type, then the result is a so-called *flat* structuring element
whose coefficients are `zero(T)` inside the shape defined by `A` and `-T(Inf)` elsewhere.

See also [`LocalFilters.kernel`](@ref) and [`LocalFilters.ball`](@ref).

"""
strel(::Type{Bool}, A::AbstractArray{Bool}) = A
strel(::Type{Bool}, A::CartesianIndices) = FastUniformArray(true, ranges(A))
strel(T::Type{<:AbstractFloat}, A::CartesianIndices) = FastUniformArray(zero(T), ranges(A))
strel(::Type{T}, A::AbstractArray{Bool}) where {T<:AbstractFloat} = map(Base.Fix1(_flat, T), A)

_flat(::Type{T}, flag::Bool) where {T<:AbstractFloat} = ifelse(flag, zero(T), -T(Inf))

"""
    LocalFilters.store!(A, I, x)

stores value `x` in array `A` at index `I`, taking care of converting `x` to the nearest
value of type `eltype(A)`. This method propagates the current in-bounds settings.

"""
@propagate_inbounds store!(A, I, x) = setindex!(A, nearest(eltype(A), x), I)

"""
    LocalFilters.ball(Dims{N}, r)

yields a mask approximating a `N`-dimensional ball of radius `r`. The result is
`N`-dimensional array of Boolean's with all dimensions odd and equal and whose values are
`true` inside the ball (that for distance to the center `≤ r`) and `false` otherwise. The
mask may be used to specify the neighborhood, the kernel, or the structuring element in
local filtering operations.

The returned mask has centered axes, to get a mask with 1-based indices, call:

    LocalFilters.ball(Dims{N}, r).parent

See also [`LocalFilters.kernel`](@ref) and [`LocalFilters.strel`](@ref).

""" ball
@public ball

function ball(::Type{Dims{N}}, radius::Real) where {N}
    radius ≥ zero(radius) || throw(ArgumentError("ball radius must be non-negative"))
    if radius isa Integer
        r = as(Int, radius)
        qmax = r*r
    else
        r = floor(Int, radius)
        qmax = floor(Int, radius*radius)
    end
    arr = new_array(Bool, ntuple(Returns(-r:r), Val(N)))
    @inbounds @simd for I in eachindex(IndexCartesian(), arr)
        arr[I] = squared_Euclidean_norm(I) ≤ qmax
    end
    return arr
end

@deprecate ball(N::Int, radius::Real) ball(Dims{N}, radius) false

squared_Euclidean_norm(I::CartesianIndex) = mapreduce(abs2, +, Tuple(I))

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
(B::FlatBoundaries{<:CartesianIndices{N}})(i::CartesianIndex{N}) where {N} =
    CartesianIndex(map(clamp_to_range, Tuple(i), ranges(indices(B))))

# Clamp to range. Range must not be empty.
clamp_to_range(idx::T, rng::AbstractUnitRange{T}) where {T<:Integer} =
    clamp(idx, first(rng), last(rng))
