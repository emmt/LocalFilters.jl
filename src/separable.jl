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
# Copyright (C) 2018-2022, Éric Thiébaut.
#

module Separable

using ..LocalFilters

using ..LocalFilters:
    IndexInterval,
    axes,
    cartesian_region

import ..LocalFilters:
    dilate!,
    dilate,
    erode!,
    erode,
    localfilter!,
    localfilter

"""
    localfilter([T=eltype(A),] A, dims, op, rngs[, wrk]) -> dst

yields the result of applying van Herk-Gil-Werman algorithm to filter array `A`
along dimension(s) `dims` with (associative) binary operation `op` and
contiguous structuring element(s) defined by the interval(s) `rngs`.  Optional
argument `wrk` is a workspace array which is automatically allocated if not
provided; otherwise, it must be a vector with the same element type as `A` and
it is resized as needed (by calling the `resize!` method).  The optional
argument `T` allows to specify another type of element than `eltype(A)` for the
result.

Argument `dims` specifies along which dimension(s) of `A` the filter is to be
applied, it can be a single integer, a tuple of integers, or a colon `:` to
apply the operation to all dimensions.  Dimensions are processed in the order
given by `dims` (the same dimension may appear several times) and there must be
a matching interval in `rngs` to specify the structuring element (except that
if `rngs` is a single interval, it is used for every dimension in `dims`).  An
interval is either an integer or an integer valued unit range in the form
`kmin:kmax` (an interval specified as a single integer, say `k`, is the same as
specifying `k:k`).

Assuming a mono-dimensional array `A`, the single filtering pass:

    dst = localfilter(A, :, op, rng)

amount to computing:

    dst[j] = A[j-kmax] ⋄ A[j-kmax+1] ⋄ A[j-kmax+2] ⋄ ... ⋄ A[j-kmin]

for all `j ∈ [first(axes(A,1)):last(axes(A,1))]`, with `x ⋄ y = op(x, y)`,
`kmin = first(rng)` and `kmax = last(rng)`.  Note that if `kmin = kmax = k`
(which occurs if `rng` is a simple integer), the result of the filter is to
operate a simple shift by `k` along the corresponding dimension and has no
effects if `k = 0`.  This can be exploited to not filter some dimension(s).

The *morphological erosion* (local minimum) of the array `A` on a centered
structuring element of width 7 in every dimension can be obtained by:

    localfilter(A, :, min, -3:3)

Index interval `0:0` may be specified to do nothing along the corresponding
dimension.  For instance, assuming `A` is a three-dimensional array:

    localfilter(A, :, max, (-3:3, 0:0, -4:4))

yields the *morphological dilation* (*i.e.* local maximum) of `A` in a centered
local neighborhood of size `7×1×9` (nothing is done along the second
dimension).  The same result may be obtained with:

    localfilter(A, (1,3), max, (-3:3, -4:4))

where the second dimension is omitted from the list of dimensions.

The *local average* of the two-dimensional array `A` on a centered moving
window of size 11×11 can be computed as:

    localfilter(A, :, +, (-5:5, -5:5))*(1/11)

See [`localfilter!`](@ref) for an in-place version of the method.

"""
function localfilter(A::AbstractArray,
                     dims::Union{Colon, Integer, Tuple{Vararg{Integer}},
                                 AbstractVector{<:Integer}},
                     op::Function, args...)
    return localfilter(eltype(A), A, dims, op, args...)
end

function localfilter(T::Type, A::AbstractArray,
                     dims::Union{Colon, Integer, Tuple{Vararg{Integer}},
                                 AbstractVector{<:Integer}},
                     op::Function, args...)
    return localfilter!(similar(A, T), A, dims, op, args...)
end


"""
    localfilter!([dst = A,] A, dims, op, rngs[, wrk])

overwrites the contents of `dst` with the result of applying van
Herk-Gil-Werman algorithm to filter array `A` along dimension(s) `dims` with
(associative) binary operation `op` and contiguous structuring element(s)
defined by the interval(s) `rngs` and using optional argument `wrk` as a
workspace array.  The destination `dst` must have the same indices as the
source `A` (that is, `axes(dst) == axes(A)`).  Operation may be done in-place
and `dst` and `A` can be the same; this is the default behavior if `dst` is not
specified.

See [`localfilter`](@ref) for a full description of the method.

The in-place *morphological erosion* (local minimum) of the array `A` on a
centered structuring element of width 7 in every dimension can be obtained by:

    localfilter!(A, :, min, -3:3)

"""
function localfilter!(dst::AbstractArray{T,N},
                      A::AbstractArray{<:Any,N}, d::Int,
                      op::Function, kmin::Int, kmax::Int,
                      wrk::Vector{T}) where {T,N}
    #
    # A monodimensional local filter involving binary operation ⋄ on a
    # neighborhood [kmin:kmax] yields the following array B when filtering the
    # array A:
    #
    #     B[j] = A[j-kmax] ⋄ A[j-kmax+1] ⋄ ... ⋄ A[j-kmin]
    #
    # Assuming ⋄ is an associative binary operation, van Herk-Gil-Werman (HGW)
    # algorithm consists in building 2 auxilliary arrays, R and S, as follows:
    #
    #     R[1] = A[j-kmin-1]
    #     R[i] = A[j-kmin-i] ⋄ R[i-1]   for i ∈ [2:p-1]
    #     S[1] = A[j-kmin]
    #     S[i+1] = S[i] ⋄ A[j-kmin+i]   for i ∈ [1:p-1]
    #
    # with p = kmax - kmin + 1 the length of the structuring element.
    # Graphically (for p = 7):
    #
    #             j-kmax      j-kmin
    #              ↓           ↓
    #     R[p-1] ← □⋄□⋄□⋄□⋄□⋄□ □             → S[1]
    #     R[p-2] ←   □⋄□⋄□⋄□⋄□ □⋄□           → S[2]
    #     R[p-3] ←     □⋄□⋄□⋄□ □⋄□⋄□         → S[3]
    #      ...           □⋄□⋄□ □⋄□⋄□⋄□          ...
    #     R[2]   ←         □⋄□ □⋄□⋄□⋄□⋄□     → S[p-2]
    #     R[1]   ←           □ □⋄□⋄□⋄□⋄□⋄□   → S[p-1]
    #                          □⋄□⋄□⋄□⋄□⋄□⋄□ → S[p]
    #
    # where each □ represents a cell in A.  Thanks to the recursion, this only
    # takes 2p - 3 operations ⋄. Then, assuming the result is stored in array
    # B, we have:
    #
    #     B[j]     = R[p-1]⋄S[1]
    #     B[j+1]   = R[p-2]⋄S[2]
    #     ...
    #     B[j+p-2] = R[1]⋄S[p-1]
    #     B[j+p-1] = S[p]
    #
    # which requires p - 1 more operations ⋄.  So the total number of
    # operations ⋄ is 3p - 4 for computing p values in B, hence 3 - 4/p
    # operations per cell on average.
    #
    # The following implementation exploits a single workspace array to
    # temporarily store A[j] to allow for in-place appllication of the filter
    # and with suplementary values to account for boundary conditions.  The
    # workspace array is also used to store R[i] while the values of S[i] are
    # computed on the fly.
    #

    # Check arguments and get dimensions.
    1 ≤ d ≤ N || throw(ArgumentError("out of bounds dimension index"))
    kmin ≤ kmax || throw(ArgumentError("invalid structuring element interval"))
    inds = axes(A)
    axes(dst) == inds || throw(DimensionMismatch(
        "source and destination must have the same indices"))
    jmin, jmax = first(inds[d]), last(inds[d])
    if kmin == kmax == 0 || jmin > jmax
        # Nothing to do!
        return copyto!(dst, A)
    end

    # Get index bounds on other parts of input array A.
    R1 = cartesian_region(inds[1:d-1])
    R2 = cartesian_region(inds[d+1:N])

    if kmin == kmax
        # Perform a simple shift: `dst[j] = A[j-k]`.  To allow for in-place
        # operation without using any temporaries, we walk along the dimension
        # in the forward direction if the shift is negative and in the
        # backward direction otherwise.
        k = kmin
        if k < 0
            _shiftarray!(dst, A, R1, jmin:jmax, R2, k)
        else
            _shiftarray!(dst, A, R1, jmax:-1:jmin, R2, k)
        end
    else
        # Apply van Herk-Gil-Werman algorithm.
        _localfilter!(dst, A, R1, jmin, jmax, R2, op, kmin, kmax, wrk)
    end
    return dst
end

# Private methods to break type uncertainty.

function _shiftarray!(dst::AbstractArray{<:Any,N}, A::AbstractArray{<:Any,N},
                      R1, rng::AbstractRange{Int}, R2, k::Int) where {N}
    jmin, jmax = minimum(rng), maximum(rng)
    @inbounds for J2 in R2, J1 in R1
        @simd for j in rng
            jp = clamp(j - k, jmin, jmax)
            dst[J1,j,J2] = A[J1,jp,J2]
        end
    end
    return nothing
end

function _localfilter!(dst::AbstractArray{T,N},
                       A::AbstractArray{<:Any,N},
                       R1, jmin::Int, jmax::Int, R2,
                       op::Function, kmin::Int, kmax::Int,
                       wrk::Vector{T}) where {T,N}
    n = jmax - jmin + 1 # length of dimension in A
    p = kmax - kmin + 1 # length of neighborhood
    imin, imax = p, workspacelength(n, p) # range for storing A in W
    length(wrk) ≥ imax || resize!(wrk, imax)
    off = imin - jmin + kmax # offset such that W[j-k+off] ≡ A[j-k]
    m = off - kmin # W[j+m] ≡ A[j-kmin]
    pm1 = p - 1

    @inbounds for J2 in R2, J1 in R1
        # Fill the workspace W[imin:imax] with A[jmin-kmax:jmax-kmin] taking
        # care of boundary conditions (here we assume nearest neighbor
        # conditions).
        @simd for i in imin:imax
            j = clamp(i - off, jmin, jmax)
            wrk[i] = A[J1,j,J2]
        end

        # Process the input by blocks of as much as p elements (less at the end
        # of the range).
        for j = jmin:p:jmax
            # Compute auxilliary array W[1:p-1] ≡ R[1:p-1].
            jpm = j + m
            wrk[1] = wrk[jpm-1] # R[1] = A[j-kmin-1]
            @simd for i in 2:pm1
                # R[i] = A[j-kmin-i] ⋄ R[i-1]   for i ∈ [2:p-1]
                wrk[i] = op(wrk[jpm-i], wrk[i-1])
            end

            # Apply the recursion to compute at least 1 and at most p resulting
            # values.
            s = wrk[jpm] # S[1] = A[j-kmin]
            dst[J1,j,J2] = op(wrk[pm1], s) # B[j] = R[p-1]⋄S[1]
            @simd for i in 1:min(p-2, jmax-j)
                s = op(s, wrk[jpm+i]) # S[i+1] = S[i]⋄A[j-kmin+i] for i ∈ [p-1:1]
                dst[J1,j+i,J2] = op(wrk[pm1-i], s) # B[j+i] = R[p-1-i]⋄S[i+1]
            end
            if j + pm1 ≤ jmax
                # B[j+p-1] = S[p-1]⋄A[j-kmin+p-1]
                dst[J1,j+pm1,J2] = op(s,wrk[jpm+pm1])
            end
        end
    end
    return nothing
end

# In-place operation.

function localfilter!(A::AbstractArray{T,N},
                      dims::Union{Colon, Integer, Tuple{Vararg{Integer}},
                                  AbstractVector{<:Integer}},
                      op::Function,
                      rngs::Union{IndexInterval, Integer,
                                  AbstractVector{<:IndexInterval},
                                  Tuple{Vararg{IndexInterval}}},
                      args...) where {T,N}
    return localfilter!(A, A, dims, op, rngs, args...)
end

# Wrapper methods when destination is specified.
#
# The filter is applied along all chosen dimensions.  To reduce page memory
# faults, the operation is performed out-of-place (unless dst and A are the
# same) for the first chosen dimension and the operation is performed in-place
# for the other chosen dimensions.

function localfilter!(dst::AbstractArray{T,N},
                      A::AbstractArray{<:Any,N},
                      d::Integer,
                      op::Function,
                      rng::IndexInterval,
                      wrk::Vector{T} = workspace(T, A, d, rng)) where {T,N}
    return localfilter!(dst, A, Int(d), op, Int(first(rng)), Int(last(rng)), wrk)
end

function localfilter!(dst::AbstractArray{T,N},
                      A::AbstractArray{<:Any,N},
                      ::Colon,
                      op::Function,
                      rng::IndexInterval,
                      wrk::Vector{T} = workspace(T, A, :, rng)) where {T,N}
    kmin, kmax = Int(first(rng)), Int(last(rng))
    N ≥ 1 || return copyto!(dst, A)
    localfilter!(dst, A, 1, op, kmin, kmax, wrk)
    for d in 2:N
        localfilter!(dst, d, op, kmin, kmax, wrk)
    end
    return dst
end

function localfilter!(dst::AbstractArray{T,N},
                      A::AbstractArray{<:Any,N},
                      ::Colon,
                      op::Function,
                      rngs::Union{AbstractVector{<:IndexInterval},
                                  Tuple{Vararg{IndexInterval}}},
                      wrk::Vector{T} = workspace(T, A, :, rngs)) where {T,N}
    length(rngs) == N || throw(DimensionMismatch(
        "there must be as many intervals as dimensions"))
    N ≥ 1 || return copyto!(dst, A)
    localfilter!(dst, A, 1, op, rngs[1], wrk)
    for d in 2:N
        localfilter!(dst, d, op, rngs[d], wrk)
    end
    return dst
end

function localfilter!(dst::AbstractArray{T,N},
                      A::AbstractArray{<:Any,N},
                      dims::Union{AbstractVector{<:Integer},
                                  Tuple{Vararg{Integer}}},
                      op::Function,
                      rng::IndexInterval,
                      wrk::Vector{T} = workspace(T, A, dims, rng)) where {T,N}
    m = length(dims)
    m ≥ 1 || return copyto!(dst, A)
    localfilter!(dst, A, dims[1], op, rng, wrk)
    for d in 2:m
        localfilter!(dst, dims[d], op, rng, wrk)
    end
    return dst
end

function localfilter!(dst::AbstractArray{T,N},
                      A::AbstractArray{<:Any,N},
                      dims::Union{AbstractVector{<:Integer},
                                  Tuple{Vararg{Integer}}},
                      op::Function,
                      rngs::Union{AbstractVector{<:IndexInterval},
                                  Tuple{Vararg{IndexInterval}}},
                      wrk::Vector{T} = workspace(T, A, dims, rngs)) where {T,N}
    m = length(dims)
    length(rngs) == m || throw(DimensionMismatch(
        "list of dimensions and list of intervals must have the same length"))
    m ≥ 1 || return copyto!(dst, A)
    localfilter!(dst, A, dims[1], op, rngs[1], wrk)
    for d in 2:m
        localfilter!(dst, dims[d], op, rngs[d], wrk)
    end
    return dst
end

# Basic morphological operations.

for (f, op) in ((:erode, min), (:dilate, max))
    fp = Symbol(f, "!")
    @eval begin

        function $f(A::AbstractArray{T,N},
                    dims::Union{Colon, Integer, Tuple{Vararg{Integer}},
                                AbstractVector{<:Integer}},
                    rngs::Union{IndexInterval, Tuple{Vararg{IndexInterval}},
                                AbstractVector{<:IndexInterval}},
                    args...) where {T,N}
            return localfilter(T, A, dims, $op, rngs, args...)
        end

        function $fp(A::AbstractArray{T,N},
                     dims::Union{Colon, Integer, Tuple{Vararg{Integer}},
                                 AbstractVector{<:Integer}},
                     rngs::Union{IndexInterval, Tuple{Vararg{IndexInterval}},
                                 AbstractVector{<:IndexInterval}},
                     args...) where {T,N}
            return localfilter!(A, dims, $op, rngs, args...)
        end

        function $fp(dst::AbstractArray{T,N},
                     A::AbstractArray{T,N},
                     dims::Union{Colon, Integer, Tuple{Vararg{Integer}},
                                 AbstractVector{<:Integer}},
                     rngs::Union{IndexInterval, Tuple{Vararg{IndexInterval}},
                                 AbstractVector{<:IndexInterval}},
                     args...) where {T,N}
            return localfilter!(dst, A, dims, $op, rngs, args...)
        end

    end
end

"""
    workspacelength(n, p)

yields the minimal length of the workspace array for applying the van
Herk-Gil-Werman algorithm along a dimension of length `n` with a structuring
element of width `p`.  If `n < 1` or `p ≤ 1`, zero is returned because there is
no needs for a workspace array.

    workspacelength(A, dims, rngs)

yields the minimal length of the workspace array for applying the van
Herk-Gil-Werman algorithm along the dimension(s) `dims` of array `A` with a
structuring element defined by the interval(s) `rngs`.  If arguments are not
compatible, zero is returned because there is no needs for a workspace array.

"""
workspacelength(n::Int, p::Int) = (n < 1 || p ≤ 1 ? 0 : n + 2*(p - 1))

function workspacelength(A::AbstractArray{<:Any,N},
                         d::Integer,
                         rng::IndexInterval)::Int where {N}
    1 ≤ d ≤ N || return 0
    n = length(axes(A, d))
    p = length(rng)
    return workspacelength(n, p)
end

function workspacelength(A::AbstractArray{<:Any,N},
                         ::Colon,
                         rng::IndexInterval)::Int where {N}
    N ≥ 1 || return 0
    p = length(rng)
    n = reduce(max, map(length, axes(A)))
    return workspacelength(n, p)
end

function workspacelength(A::AbstractArray{<:Any,N},
                         ::Colon,
                         rngs::Union{AbstractVector{<:IndexInterval},
                                     Tuple{Vararg{IndexInterval}}})::Int where {N}
    len = 0
    if length(rngs) == N
        for d in 1:N
            n = length(axes(A, d))
            p = length(rngs[d])
            len = max(len, workspacelength(n, p))
        end
    end
    return len
end

function workspacelength(A::AbstractArray{<:Any,N},
                         dims::Union{AbstractVector{<:Integer},
                                     Tuple{Vararg{Integer}}},
                         rng::IndexInterval)::Int where {N}
    len = 0
    p = length(rng)
    for dim in dims
        n = length(axes(A, dim))
        len = max(len, workspacelength(n, p))
    end
    return len
end

function workspacelength(A::AbstractArray{<:Any,N},
                         dims::Union{AbstractVector{<:Integer},
                                     Tuple{Vararg{Integer}}},
                         rngs::Union{AbstractVector{<:IndexInterval},
                                     Tuple{Vararg{IndexInterval}}})::Int where {N}
    len = 0
    if (m = length(dims)) == length(rngs)
        for d in 1:m
            n = length(axes(A, dims[d]))
            p = length(rngs[d])
            len = max(len, workspacelength(n, p))
        end
    end
    return len
end

"""
    workspace(T, A, dims, rngs)

yields a workspace array for applying the van Herk-Gil-Werman algorithm along
the dimension(s) `dims` of array `A` with a structuring element defined by the
interval(s) `rngs`.  The element type of the workspace is `T` which is that of
`A` by default.

"""
function workspace(::Type{T},
                   A::AbstractArray{<:Any,N},
                   dims::Union{Colon, Integer, AbstractVector{<:Integer},
                               Tuple{Vararg{Integer}}},
                   rngs::Union{IndexInterval, AbstractVector{<:IndexInterval},
                               Tuple{Vararg{IndexInterval}}}) where {T,N}
    return Array{T}(undef, workspacelength(A, dims, rngs))
end

end # module
