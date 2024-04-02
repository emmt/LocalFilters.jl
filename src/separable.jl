#
# separable.jl --
#
# Implementation of efficient separable filters by means of the van
# Herk-Gil-Werman algorithm.
#
#------------------------------------------------------------------------------
#
# This file is part of the `LocalFilters.jl` package licensed under the MIT
# "Expat" License.
#
# Copyright (c) 2018-2024, Éric Thiébaut.
#

module Separable

using ..LocalFilters

using ..LocalFilters:
    FilterOrdering,
    ForwardFilterOrdering,
    ReverseFilterOrdering,
    BoundaryConditions,
    FlatBoundaries,
    LocalAxis,
    kernel_range

import ..LocalFilters:
    dilate!,
    dilate,
    erode!,
    erode,
    localfilter!,
    localfilter

# Union of for possible types to specify the dimension(s) of interest.
const Dimensions = Union{Colon, Integer, Tuple{Vararg{Integer}},
                         AbstractVector{<:Integer}}

# Union of for possible types to specify the neighborhoods ranges.
const Ranges = Union{LocalAxis, Tuple{Vararg{LocalAxis}}, AbstractVector{<:LocalAxis}}

"""
    filter_range([ord=ForwardFilter,] len)
    filter_range([ord=ForwardFilter,] rng)

yields an `Int`-valued unit step range for specifying the filter range for
ordering `ord`. The result is of length `len` or is based on index range `rng`.

"""
filter_range(len::Integer) = kernel_range(len)
filter_range(rng::AbstractUnitRange{Int}) = rng
filter_range(rng::AbstractUnitRange{<:Integer}) = Int(first(rng)):Int(last(rng))
filter_range(rng::OrdinalRange{<:Integer,<:Integer}) = begin
    s = step(rng)
    if s == one(s)
        return Int(first(rng)):Int(last(rng))
    elseif s == -one(s)
        return -Int(last(r)):-Int(first(r))
    else
        throw(ArgumentError("unsupported non-unit step index range"))
    end
end
filter_range(::ForwardFilterOrdering, args...) = filter_range(args...)
filter_range(::ReverseFilterOrdering, args...) = begin
    rng = filter_range(args...)
    return -last(rng):-first(rng)
end

"""
    WorkVector(buf, len, skip=0) -> A
    WorkVector(buf, rng, skip=0) -> A

yield an abstract vector `A` of length `len` or of indices given by the range
`rng` which share its elements with the buffer `buf`. The first entry of `A` is
the `skip+1`-th entry of `buf`. If `buf` is not large enough, it is
automatically resized.

"""
struct WorkVector{T} <: DenseVector{T}
    parent::Vector{T}
    indices::UnitRange{Int}
    offset::Int
    function WorkVector(buf::Vector{T},
                        rng::UnitRange{Int},
                        skip::Int = 0) where {T}
        skip ≥ 0 || throw(ArgumentError(
            "number of elements to skip must be nonnegative"))
        minlen = skip + length(rng)
        length(buf) ≥ minlen || resize!(buf, minlen)
        offset = skip + firstindex(buf) - first(rng)
        return new{T}(buf, rng, offset)
    end
end

function WorkVector(A::WorkVector,
                    inds::Union{Integer,OrdinalRange{<:Integer,<:Integer}},
                    skip::Integer = 0)
    buf = parent(A)
    prevskip = offset(A) + firstindex(A) - firstindex(buf)
    return WorkVector(buf, inds, prevskip + Int(skip))
end

function WorkVector(A::Vector, len::Integer, skip::Integer = 0)
    len ≥ 0 || throw(ArgumentError("length must be nonnegative"))
    return WorkVector(A, 1:Int(len), Int(skip))
end
function WorkVector(A::Vector, rng::AbstractUnitRange{<:Integer},
                    skip::Integer = 0)
    return WorkVector(A, Int(first(rng)):Int(last(rng)), Int(skip))
end
function WorkVector(A::Vector, rng::OrdinalRange{<:Integer,<:Integer},
                    skip::Integer = 0)
    step(rng) == 1 || throw(ArgumentError("invalid non unit-step range"))
    return WorkVector(A, Int(first(rng)):Int(last(rng)), Int(skip))
end

offset(A::WorkVector) = getfield(A, :offset)
indices(A::WorkVector) = getfield(A, :indices)
Base.parent(A::WorkVector) = getfield(A, :parent)
Base.length(A::WorkVector) = length(indices(A))
Base.axes(A::WorkVector) = (indices(A),)
Base.axes1(A::WorkVector) = indices(A)
Base.size(A::WorkVector) = (length(A),)
Base.IndexStyle(::Type{<:WorkVector}) = IndexLinear()
@inline function Base.getindex(A::WorkVector, i::Int)
    @boundscheck checkbounds(A, i)
    return @inbounds parent(A)[i + offset(A)]
end
@inline function Base.setindex!(A::WorkVector, x, i::Int)
    @boundscheck checkbounds(A, i)
    @inbounds parent(A)[i + offset(A)] = x
    return A
end

"""
    localfilter([T=eltype(A),] A, dims, op, [ord=ForwardFilter,]
                rngs[, wrk]) -> dst

yields the result of applying van Herk-Gil-Werman algorithm to filter array `A`
along dimension(s) `dims` with (associative) binary operation `op` and
contiguous structuring element(s) defined by the interval(s) `rngs`. Optional
argument `wrk` is a workspace array with elements of type `T` which is
automatically allocated if not provided; otherwise, it must be a vector with
the same element type as `A` and it is resized as needed (by calling the
`resize!` method). The optional argument `T` allows to specify another type of
element than `eltype(A)` for the result.

Argument `dims` specifies along which dimension(s) of `A` the filter is to be
applied, it can be a single integer, a tuple of integers, or a colon `:` to
apply the operation to all dimensions. Dimensions are processed in the order
given by `dims` (the same dimension may appear several times) and there must be
a matching interval in `rngs` to specify the structuring element (except that
if `rngs` is a single interval, it is used for every dimension in `dims`). An
interval is either an integer or an integer valued unit range in the form
`kmin:kmax`. An interval specified as a single integer yields an approximately
centered range og this length.

Assuming a mono-dimensional array `A`, the single filtering pass:

    dst = localfilter(A, :, op, rng)

amounts to computing (assuming forward ordering):

    dst[j] =  A[i+kmin] ⋄ A[i+kmin+1] ⋄ ... ⋄ A[i+kmax-1] ⋄ A[i+kmax]

for all `j ∈ axes(dst,1)`, with `x ⋄ y = op(x, y)`, `kmin = first(rng)` and
`kmax = last(rng)`. Note that if `kmin = kmax = k`, the result of the filter is
to operate a simple shift by `k` along the corresponding dimension and has no
effects if `k = 0`. This can be exploited to not filter some dimension(s).

Flat boundary conditions are assumed for `A[i+k]` in the above formula.

## Examples

The *morphological erosion* (local minimum) of the array `A` on a centered
structuring element of width 7 in every dimension can be obtained by:

    localfilter(A, :, min, -3:3)

Index interval `0:0` may be specified to do nothing along the corresponding
dimension. For instance, assuming `A` is a three-dimensional array:

    localfilter(A, :, max, (-3:3, 0:0, -4:4))

yields the *morphological dilation* (*i.e.* local maximum) of `A` in a centered
local neighborhood of size `7×1×9` (nothing is done along the second
dimension). The same result may be obtained with:

    localfilter(A, (1,3), max, (-3:3, -4:4))

where the second dimension is omitted from the list of dimensions.

The *local average* of the two-dimensional array `A` on a centered moving
window of size 11×11 can be computed as:

    localfilter(A, :, +, (-5:5, -5:5))*(1/11)

See [`localfilter!`](@ref) for an in-place version of the method.

"""
function localfilter(A::AbstractArray,
                     dims::Dimensions,
                     op::Function, args...)
    return localfilter(eltype(A), A, dims, op, args...)
end

function localfilter(T::Type, A::AbstractArray,
                     dims::Dimensions,
                     op::Function, args...)
    return localfilter!(similar(A, T), A, dims, op, args...)
end

"""
    localfilter!([dst = A,] A, dims, op, [ord=ForwardFilter,] rngs[, wrk])

overwrites the contents of `dst` with the result of applying van
Herk-Gil-Werman algorithm to filter array `A` along dimension(s) `dims` with
(associative) binary operation `op` and contiguous structuring element(s)
defined by the interval(s) `rngs` and using optional argument `wrk` as a
workspace array. The destination `dst` must have the same indices as the source
`A` (that is, `axes(dst) == axes(A)`). Operation may be done in-place and `dst`
and `A` can be the same; this is the default behavior if `dst` is not
specified.

See [`localfilter`](@ref) for a full description of the method.

The in-place *morphological erosion* (local minimum) of the array `A` on a
centered structuring element of width 7 in every dimension can be obtained by:

    localfilter!(A, :, min, -3:3)

""" localfilter!

# In-place operation.
localfilter!(A::AbstractArray, dims::Dimensions, op::Function, args...) =
    localfilter!(A, A, dims, op, args...)

# Provide ordering.
function localfilter!(dst::AbstractArray{T,N},
                      A::AbstractArray{<:Any,N},
                      dims::Dimensions,
                      op::Function,
                      rngs::Ranges,
                      args...) where {T,N}
    return localfilter!(dst, A, dims, op, ForwardFilter, rngs, args...)
end

# Provide workspace.
function localfilter!(dst::AbstractArray{T,N},
                      A::AbstractArray{<:Any,N},
                      dims::Dimensions,
                      op::Function,
                      ord::FilterOrdering,
                      rngs::Ranges) where {T,N}
    wrk = Array{T}(undef, workspace_length(A, dims, rngs))
    return localfilter!(dst, A, dims, op, ord, rngs, wrk)
end

# Wrapper methods when destination, ordering, and workspace are specified.
#
# The filter is applied along all chosen dimensions. To reduce page memory
# faults, the operation is performed out-of-place (unless dst and A are the
# same) for the first chosen dimension and the operation is performed in-place
# for the other chosen dimensions.

# Versions for a single given filter range.

function localfilter!(dst::AbstractArray{T,N},
                      A::AbstractArray{<:Any,N},
                      dims::Dimensions,
                      op::Function,
                      ord::FilterOrdering,
                      rng::LocalAxis,
                      wrk::Vector{T}) where {T,N}
    # small optimization: convert range once
    return localfilter!(dst, A, dims, op, ForwardFilter,
                        filter_range(ord, rng), wrk)
end

function localfilter!(dst::AbstractArray{T,N},
                      A::AbstractArray{<:Any,N},
                      ::Colon,
                      op::Function,
                      ord::ForwardFilterOrdering,
                      rng::AbstractUnitRange{Int},
                      wrk::Vector{T}) where {T,N}
    localfilter!(dst, A, 1, op, ord, rng, wrk)
    for d in 2:N
        localfilter!(dst, d, op, ord, rng, wrk)
    end
    return dst
end

function localfilter!(dst::AbstractArray{T,N},
                      A::AbstractArray{<:Any,N},
                      dims::Union{AbstractVector{<:Integer},
                                  Tuple{Vararg{Integer}}},
                      op::Function,
                      ord::ForwardFilterOrdering,
                      rng::AbstractUnitRange{Int},
                      wrk::Vector{T}) where {T,N}
    if length(dims) ≥ 1
        i = firstindex(dims)
        localfilter!(dst, A, dims[i], op, ord, rng, wrk)
        while i < lastindex(dims)
            i += 1
            localfilter!(dst, dims[i], op, ord, rng, wrk)
        end
    end
    return dst
end


# Versions for a multiple given filter ranges.

function localfilter!(dst::AbstractArray{T,N},
                      A::AbstractArray{<:Any,N},
                      ::Colon,
                      op::Function,
                      ord::FilterOrdering,
                      rngs::Union{AbstractVector{<:LocalAxis},
                                  Tuple{Vararg{LocalAxis}}},
                      wrk::Vector{T}) where {T,N}
    length(rngs) == N || throw(DimensionMismatch(
        "there must be as many intervals as dimensions"))
    if N ≥ 1
        dim = 1
        off = firstindex(rngs) - dim
        localfilter!(dst, A, dim, op, ord, rngs[off+dim], wrk)
        while dim < N
            dim += 1
            localfilter!(dst, dim, op, ord, rngs[off+dim], wrk)
        end
    end
    return dst
end

function localfilter!(dst::AbstractArray{T,N},
                      A::AbstractArray{<:Any,N},
                      dims::Union{AbstractVector{<:Integer},
                                  Tuple{Vararg{Integer}}},
                      op::Function,
                      ord::FilterOrdering,
                      rngs::Union{AbstractVector{<:LocalAxis},
                                  Tuple{Vararg{LocalAxis}}},
                      wrk::Vector{T}) where {T,N}
    len = length(dims)
    length(rngs) == len || throw(DimensionMismatch(
        "list of dimensions and list of intervals must have the same length"))
    if len ≥ 1
        idx = firstindex(dims)
        off = firstindex(rngs) - idx
        localfilter!(dst, A, dims[idx], op, ord, rngs[off+idx], wrk)
        while idx < lastindex(dims)
            idx += 1
            localfilter!(dst, dims[idx], op, ord, rngs[off+idx], wrk)
        end
    end
    return dst
end

# Apply filter along a single given dimension. This is the most basic version
# which is called by other versions.
function localfilter!(dst::AbstractArray{T,N},
                      A::AbstractArray{<:Any,N},
                      dim::Integer, # dimension of interest
                      op::Function,
                      ord::FilterOrdering,
                      rng::LocalAxis,
                      wrk::Vector{T}) where {T,N}
    1 ≤ dim ≤ N || throw(ArgumentError("out of bounds dimension"))
    isempty(rng) &&  throw(ArgumentError("invalid filter size"))
    unsafe_localfilter!(dst, A, Val(Int(dim)), op, filter_range(ord, rng), wrk)
    return dst
end

# This version is to break the type instability related to the variable
# dimension of interest.
function unsafe_localfilter!(dst::AbstractArray{T,N},
                             A::AbstractArray{<:Any,N},
                             ::Val{D}, # dimension of interest
                             op::Function,
                             K::AbstractUnitRange{Int},
                             wrk::Vector{T}) where {T,N,D}
    src_inds = axes(A)
    dst_inds = axes(dst)
    for d in 1:D-1
        src_inds[d] == dst_inds[d] || throw(DimensionMismatch(
            "source and destination have incompatible leading indices"))
    end
    for d in D+1:N
        src_inds[d] == dst_inds[d] || throw(DimensionMismatch(
            "source and destination have incompatible trailing indices"))
    end
    I = dst_inds[D]
    J = src_inds[D]
    if I == J && (isempty(I) || (length(K) == 1 && first(K) == 0))
        # Nothing to do and arrays have the same size.
        copyto!(dst, A)
        return
    end
    B = FlatBoundaries(J)
    I1 = CartesianIndices(dst_inds[1:D-1])
    I2 = CartesianIndices(dst_inds[D+1:N])
    if length(K) == 1
        # Perform a simple shift: `dst[j] = A[j+k]`.
        unsafe_shiftarray!(dst, I1, I, I2, A, B, first(K))
    else
        # Apply van Herk-Gil-Werman algorithm.
        unsafe_localfilter!(dst, I1, I, I2, A, B, op, K, wrk)
    end
end

function unsafe_shiftarray!(dst::AbstractArray{<:Any,N},
                            I1, I::AbstractUnitRange{Int}, I2,
                            src::AbstractArray{<:Any,N},
                            B::BoundaryConditions,
                            k::Int) where {N}
    @inbounds for i2 ∈ I2, i1 ∈ I1
        # To allow for in-place operation without using any temporaries, we
        # walk along the dimension in the forward direction if the shift is
        # nonnegative and in the reverse direction otherwise.
        if k ≥ 0
            @simd for i ∈ I
                dst[i1,i,i2] = src[i1,B(i+k),i2]
            end
        else
            @simd for i ∈ reverse(I)
                dst[i1,i,i2] = src[i1,B(i+k),i2]
            end
        end
    end
    nothing
end

function unsafe_localfilter!(dst::AbstractArray{T,N},
                             I1, # range of leading indices
                             I::AbstractUnitRange{Int}, # index range in dst
                             I2, # range of trailing indices
                             src::AbstractArray{<:Any,N},
                             B::BoundaryConditions, # boundary conditions in src
                             op::Function,
                             K::AbstractUnitRange{Int},
                             R::Vector{T}) where {T,N}
    # The code assumes that the workspace R has 1-based indices, all other
    # arrays may have any indices.
    @assert firstindex(R) == 1

    # Get ranges for indices.
    imin, imax = first(I), last(I) # destination index range
    kmin, kmax = first(K), last(K) # neighborhood index range
    n = length(K) # length of neighborhood

    # Make a work vector for storing values from the source accounting for
    # boundary conditions and leaving n values for the array R.
    L = (imin + kmin):(imax + kmax)
    A = WorkVector(R, L, n)

    # The van Herk-Gil-Werman algorithm
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #
    # A monodimensional local filter involving binary operation ⋄ on index
    # range K = kmin:kmax yields the following array C when filtering the array
    # A (with forward ordering):
    #
    #     C[i] = A[i+kmin] ⋄ A[i+kmin+1] ⋄ ... ⋄ A[i+kmax-1] ⋄ A[i+kmax]
    #
    # Assuming ⋄ is an associative binary operation, van Herk-Gil-Werman (HGW)
    # algorithm consists in building 2 auxilliary arrays, R and S, as follows:
    #
    #     R[1] = A[i+kmax-1]
    #     R[j] = A[i+kmax-j] ⋄ R[j-1]   ∀ j ∈ 2:n-1
    #
    # and:
    #
    #     S[1]   = A[i+kmax]
    #     S[j+1] = S[j] ⋄ A[i+kmax+j]      ∀ j ∈ 1:n-1
    #
    # with n = length(K) the length of the structuring element. Graphically
    # (for n = 7):
    #
    #             i+kmin      i+kmax
    #              ↓           ↓
    #     R[n-1] ← □⋄□⋄□⋄□⋄□⋄□ □             → S[1]
    #     R[n-2] ←   □⋄□⋄□⋄□⋄□ □⋄□           → S[2]
    #     R[n-3] ←     □⋄□⋄□⋄□ □⋄□⋄□         → S[3]
    #      ...           □⋄□⋄□ □⋄□⋄□⋄□          ...
    #     R[2]   ←         □⋄□ □⋄□⋄□⋄□⋄□     → S[n-2]
    #     R[1]   ←           □ □⋄□⋄□⋄□⋄□⋄□   → S[n-1]
    #                          □⋄□⋄□⋄□⋄□⋄□⋄□ → S[n]
    #
    # where each □ represents a cell in A. Thanks to the recursion, this only
    # takes 2n - 3 operations ⋄. Then, assuming the result is stored in array
    # C, we have:
    #
    #     C[i+j-1] = R[n-j]⋄S[j]   ∀ j ∈ 1:n-1
    #     C[i+n-1] = S[n]
    #
    # which requires n - 1 more operations ⋄. So the total number of operations
    # ⋄ is 3n - 4 for computing n values in C, hence 3 - 4/n operations per
    # cell on average.
    #
    # The following implementation exploits a single workspace array to
    # temporarily store A[i] to allow for in-place appllication of the filter
    # and with suplementary values to account for boundary conditions. The
    # workspace array is also used to store R[j] while the values of S[j] are
    # computed on the fly.
    #

    @inbounds for i2 ∈ I2, i1 ∈ I1
        # Fill the workspace with the source for all possible indices and
        # taking care of boundary conditions.
        @simd for l ∈ L
            A[l] = src[i1,B(l),i2]
        end

        # Process the input by blocks of as much as n elements (less at the end
        # of the range).
        for i = imin:n:imax
            # Initialize auxilliary array R.
            R[1] = A[i+kmax-1]
            @simd for j ∈ 2:n-1
                R[j] = op(A[i+kmax-j], R[j-1])
            end
            # Apply the recursion to compute at least 1 and at most n resulting
            # values.
            jmax = imax-i+1 # max. value for j in dst[i1,i+j-1,i2]
            # First output (j = 1).
            s = A[i+kmax]                # S[1]
            dst[i1,i,i2] = op(R[n-1], s) # C[i] = R[n-1] ⋄ S[1]
            # Intermediate ouptputs (j ∈ 2:n-1).
            @simd for j ∈ 2:min(n-1, jmax)
                s = op(s, A[i+kmax+j-1])         # S[j] = S[j-1] ⋄ A[i+kmax+j-1]
                dst[i1,i+j-1,i2] = op(R[n-j], s) # C[i+j-1] = R[n-j]⋄S[j]
            end
            # Last output (j = n).
            if n ≤ jmax
                # C[i+n-1] = S[n] = S[n-1] ⋄ A[i+kmax+n-1]
                dst[i1,i+n-1,i2] = op(s, A[i+kmax+n-1])
            end
        end
    end
    nothing
end

# Basic morphological operations.

for (f, op) in ((:erode, :min), (:dilate, :max))
    f! = Symbol(f, "!")
    @eval begin

        function $f(A::AbstractArray{T,N},
                    dims::Dimensions,
                    rngs::Ranges,
                    args...) where {T,N}
            return $f(A, dims, ForwardFilter, rngs, args...)
        end

        function $f!(A::AbstractArray{T,N},
                     dims::Dimensions,
                     rngs::Ranges,
                     args...) where {T,N}
            return $f!(A, dims, ForwardFilter, rngs, args...)
        end

        function $f!(dst::AbstractArray{T,N},
                     A::AbstractArray{T,N},
                     dims::Dimensions,
                     rngs::Ranges,
                     args...) where {T,N}
            return $f!(dst, A, dims, ForwardFilter, rngs, args...)
        end

        function $f(A::AbstractArray{T,N},
                    dims::Dimensions,
                    ord::FilterOrdering,
                    rngs::Ranges,
                    args...) where {T,N}
            return localfilter(T, A, dims, $op, ord, rngs, args...)
        end

        function $f!(A::AbstractArray{T,N},
                     dims::Dimensions,
                     ord::FilterOrdering,
                     rngs::Ranges,
                     args...) where {T,N}
            return localfilter!(A, dims, $op, ord, rngs, args...)
        end

        function $f!(dst::AbstractArray{T,N},
                     A::AbstractArray{T,N},
                     dims::Dimensions,
                     ord::FilterOrdering,
                     rngs::Ranges,
                     args...) where {T,N}
            return localfilter!(dst, A, dims, $op, ord, rngs, args...)
        end

    end
end

"""
    workspace_length(n, p)

yields the minimal length of the workspace array for applying the van
Herk-Gil-Werman algorithm along a dimension of length `n` with a structuring
element of width `p`. If `n < 1` or `p ≤ 1`, zero is returned because a
workspace array is not needed.

    workspace_length(A, dims, rngs)

yields the minimal length of the workspace array for applying the van
Herk-Gil-Werman algorithm along the dimension(s) `dims` of array `A` with a
structuring element defined by the interval(s) `rngs`. If arguments are not
compatible, zero is returned because there is no needs for a workspace array.

"""
workspace_length(n::Int, p::Int) = ifelse((n < 1)|(p ≤ 1), 0, n + 2*(p - 1))
workspace_length(n::Integer, p::Integer) = workspace_length(Int(n), Int(p))

function workspace_length(A::AbstractArray, dims::Dimensions,
                          rng::OrdinalRange{<:Integer,<:Integer})
    return  workspace_length(A, dims, length(rng))
end

function workspace_length(A::AbstractArray, dim::Integer, len::Integer)
    if 1 ≤ dim ≤ ndims(A)
        return workspace_length(length(axes(A, dim)), len)
    else
        return 0
    end
end

function workspace_length(A::AbstractArray, ::Colon, len::Integer)
    if ndims(A) ≥ 1
        return workspace_length(reduce(max, map(length, axes(A))), len)
    else
        return 0
    end
end

function workspace_length(A::AbstractArray,
                          dims::Union{AbstractVector{<:Integer},
                                      Tuple{Vararg{Integer}}},
                          len::Integer)
    result = 0
    for dim in dims
        result = max(result, workspace_length(A, dim, len))
    end
    return result
end

function workspace_length(A::AbstractArray, ::Colon,
                          rngs::Union{AbstractVector{<:LocalAxis},
                                      Tuple{Vararg{LocalAxis}}})
    result = 0
    if length(rngs) == ndims(A)
        for (dim, rng) in enumerate(rngs)
            result = max(result, workspace_length(A, dim, rng))
        end
    end
    return result
end

function workspace_length(A::AbstractArray,
                          dims::Union{AbstractVector{<:Integer},
                                      Tuple{Vararg{Integer}}},
                          rngs::Union{AbstractVector{<:LocalAxis},
                                      Tuple{Vararg{LocalAxis}}})
    result = 0
    if length(rngs) == length(dims)
        for (dim, rng) in zip(dims, rngs)
            result = max(result, workspace_length(A, dim, rng))
        end
    end
    return result
end

end # module
