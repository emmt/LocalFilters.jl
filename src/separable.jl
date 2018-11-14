#
# separable.jl --
#
# Implementation of efficient separables filters by means of the van
# Herk-Gil-Werman algorithm.
#
#------------------------------------------------------------------------------
#
# This file is part of the `LocalFilters.jl` package licensed under the MIT
# "Expat" License.
#
# Copyright (C) 2018, Éric Thiébaut.
#

"""
# Local filter by the van Herk-Gil-Werman algorithm

```julia
localfilter!([dst = A,] A, dims, op, rngs [, w])
```

overwrite the contents of `dst` with the result of applying van Herk-Gil-Werman
algorithm to filter array `A` along dimension(s) `dims` with (associative)
binary operation `op` and contiguous structuring element(s) defined by the
interval(s) `rngs`.  Optional argument `w` is a workspace array which is
automatically allocated if not provided; otherwise, it must be a vector of same
element type as `A` which is resized (with [`resize!`](@ref)) as needed.  The
destination `dst` must have the same indices as the source `A` (see
[`axes`](@ref)).  Operation can be done in-place; that is, `dst` and `A` can be
the same.

Argument `dims` specifies along which dimension(s) of `A` the filter is to be
applied, it can be a single integer, several integers or a colon `:` to specify
all dimensions.  Dimensions are processed in the order given by `dims` (the
same dimension may appear several times) and there must be a matching interval
in `rngs` to specify the structuring element (except that if `rngs` is a single
interval, it is used for every dimension in `dims`).  An interval is either an
integer or an integer unit range in the form `kmin:kmax` (an interval specified
as a single integer, say `k`, is the same as specifying `k:k`).

Assuming mono-dimensional arrays `A` and `dst`, the single filtering pass:

```julia
localfilter!(dst, A, :, op, rng)
```

yields:

```
dst[j] = A[j-kmax] ⋄ A[j-kmax+1] ⋄ A[j-kmax+2] ⋄ ... ⋄ A[j-kmin]
```

for all `j ∈ [first(axes(A,1)):last(axes(A,1))]`, with `x ⋄ y = op(x, y)`,
`kmin = first(rng)` and `kmax = last(rng)`.  Note that if `kmin = kmax = k`
(which occurs if `rng` is a simple integer), the result of the filter is to
operate a simple shift by `k` along the corresponding dimension and has no
effects if `k = 0`.  This can be exploited to not filter some dimension(s).

The out-place version, allocates the destination array and is called as:

```julia
localfilter(A, dims, op, rngs [, w])
```

## Examples

The in-place *morphological erosion* (local minimum) of the array `A` on a
centered structuring element of width 7 in every dimension can be applied by:

```julia
localfilter!(A, :, min, -3:3)
```

One can specify index interval `0` to do nothing along the corresponding
dimension.  For instance:

```julia
localfilter!(A, :, max, (-3:3, 0, -4:4))
```

will overwrite `A` with the local maxima (a.k.a. *morphological dilation*) of
the three-dimensional array `A` in a centered local neighborhood of size
`7×1×9` (nothing is done along the second dimension).  The same result may be
obtained with:

```julia
localfilter!(A, (1,3), max, (-3:3, -4:4))
```

where the second dimension is omitted from the list of dimensions.

The *local average* of the two-dimensional array `A` on a centered
structuring element of size 11×11 can be computed as:

```julia
localfilter(A, :, +, (-5:5, -5:5))*(1/11)
```

## Efficiency and restrictions

The van Herk-Gil-Werman algorithm is very fast for rectangular structuring
elements.  It takes at most 3 operations to filter an element along a given
dimension whatever the width `p` of the considered local neighborhood.  For
`N`-dimensional arrays, the algorithm requires only `3N` operations per element
instead of `p^N - 1` operations for a naive implementation.  This however
requires to make a pass along each dimension so memory page faults may reduce
the performances.  This is somewhat attenuated by the fact that the algorithm
can be applied in-place.  For efficient multi-dimensional out-of-place
filtering, it is recommended to make the first pass with a fresh destination
array and then all other passes in-place on the destination array.

To apply the van Herk-Gil-Werman algorithm, the structuring element must be
separable along the dimensions and its components must be contiguous.  In other
words, the algorithm is only applicable for `N`-dimensional rectangular
neighborhoods.  The structuring element may however be off-centered by
arbitrary offsets along each dimension.

To take into account boundary conditions (for now only least neighborhood is
implemented) and allow for in-place operation, the algorithm allocates a
workspace array.

"""
function localfilter!(dst::AbstractArray{T,N},
                      A::AbstractArray{T,N}, d::Int,
                      op::Function, kmin::Int, kmax::Int,
                      w::Vector{T}) where {T,N}
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
    axes(dst) == inds || throw(DimensionMismatch("source and destination must have the same indices"))
    jmin, jmax = first(inds[d]), last(inds[d])
    if kmin == kmax == 0 || jmin > jmax
        # Nothing to do!
        return copyto!(dst, A)
    end

    # Get index bounds on other parts of input array A.
    R1 = cartesianregion(inds[1:d-1])
    R2 = cartesianregion(inds[d+1:N])

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
        _localfilter!(dst, A, R1, jmin, jmax, R2, op, kmin, kmax, w)
    end
    return dst
end

# Private methods to break type uncertainty.

function _shiftarray!(dst::AbstractArray{T,N}, A::AbstractArray{T,N}, R1,
                      rng::AbstractRange{Int}, R2, k::Int) where {T,N}
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
                       A::AbstractArray{T,N},
                       R1, jmin::Int, jmax::Int, R2,
                       op::Function, kmin::Int, kmax::Int,
                       w::Vector{T}) where {T,N}
    n = jmax - jmin + 1 # length of dimension in A
    p = kmax - kmin + 1 # length of neighborhood
    imin, imax = p, workspacelength(n, p) # range for storing A in W
    length(w) ≥ imax || resize!(w, imax)
    off = imin - jmin + kmax # offset such that W[j-k+off] ≡ A[j-k]
    m = off - kmin # W[j+m] ≡ A[j-kmin]
    pm1 = p - 1

    @inbounds for J2 in R2, J1 in R1
        # Fill the workspace W[imin:imax] with A[jmin-kmax:jmax-kmin] taking
        # care of boundary conditions (here we assume nearest neighbor
        # conditions).
        @simd for i in imin:imax
            j = clamp(i - off, jmin, jmax)
            w[i] = A[J1,j,J2]
        end

        # Process the input by blocks of as much as p elements (less at the end
        # of the range).
        for j = jmin:p:jmax
            # Compute auxilliary array W[1:p-1] ≡ R[1:p-1].
            jpm = j + m
            w[1] = w[jpm-1] # R[1] = A[j-kmin-1]
            @simd for i in 2:pm1
                # R[i] = A[j-kmin-i] ⋄ R[i-1]   for i ∈ [2:p-1]
                w[i] = op(w[jpm-i], w[i-1])
            end

            # Apply the recursion to compute at least 1 and at most p resulting
            # values.
            s = w[jpm] # S[1] = A[j-kmin]
            dst[J1,j,J2] = op(w[pm1], s) # B[j] = R[p-1]⋄S[1]
            @simd for i in 1:min(p-2, jmax-j)
                s = op(s, w[jpm+i]) # S[i+1] = S[i]⋄A[j-kmin+i] for i ∈ [p-1:1]
                dst[J1,j+i,J2] = op(w[pm1-i], s) # B[j+i] = R[p-1-i]⋄S[i+1]
            end
            if j + pm1 ≤ jmax
                # B[j+p-1] = S[p-1]⋄A[j-kmin+p-1]
                dst[J1,j+pm1,J2] = op(s,w[jpm+pm1])
            end
        end
    end
    return nothing
end

# Provide destination.

function localfilter(A::AbstractArray{T,N},
                     dims::Union{Colon, Integer, Tuple{Vararg{Integer}},
                                 AbstractVector{<:Integer}},
                     op::Function, args...) where {T,N}
    return localfilter!(similar(Array{T,N}, axes(A)), A, dims, op, args...)
end

@doc @doc(localfilter!) localfilter

# In-place operation.

function localfilter!(A::AbstractArray{T,N},
                      dims::Union{Colon, Integer, Tuple{Vararg{Integer}},
                                  AbstractVector{<:Integer}},
                      op::Function, args...) where {T,N}
    return localfilter!(A, A, dims, op, args...)
end

# Wrapper methods when destination is specified.

function localfilter!(dst::AbstractArray{T,N},
                      A::AbstractArray{T,N},
                      d::Integer,
                      op::Function,
                      rng::IndexInterval,
                      w::Vector{T} = workspace(A, d, rng)) where {T,N}
    return localfilter!(dst, A, Int(d), op, Int(first(rng)), Int(last(rng)), w)
end

function localfilter!(dst::AbstractArray{T,N},
                      A::AbstractArray{T,N},
                      ::Colon,
                      op::Function,
                      rng::IndexInterval,
                      w::Vector{T} = workspace(A, :, rng)) where {T,N}
    kmin, kmax = Int(first(rng)), Int(last(rng))
    if N ≥ 1
        localfilter!(dst, A, 1, op, kmin, kmax, w)
        for d in 2:N
            localfilter!(dst, d, op, kmin, kmax, w)
        end
    else
        copyto!(dst, A)
    end
    return dst
end

function localfilter!(dst::AbstractArray{T,N},
                      A::AbstractArray{T,N},
                      ::Colon,
                      op::Function,
                      rngs::Union{AbstractVector{<:IndexInterval},
                                  Tuple{Vararg{IndexInterval}}},
                      w::Vector{T} = workspace(A, :, rngs)) where {T,N}
    length(rngs) == N || throw(DimensionMismatch("there must be as many intervals as dimensions"))
    if N ≥ 1
        localfilter!(dst, A, 1, op, rngs[1], w)
        for d in 2:N
            localfilter!(dst, d, op, rngs[d], w)
        end
    else
        copyto!(dst, A)
    end
    return dst
end

function localfilter!(dst::AbstractArray{T,N},
                      A::AbstractArray{T,N},
                      dims::Union{AbstractVector{<:Integer},
                                  Tuple{Vararg{Integer}}},
                      op::Function,
                      rng::IndexInterval,
                      w::Vector{T} = workspace(A, dims, rng)) where {T,N}
    if (m = length(dims)) ≥ 1
        localfilter!(dst, A, dims[1], op, rng, w)
        for d in 2:m
            localfilter!(dst, dims[d], op, rng, w)
        end
    else
        copyto!(dst, A)
    end
    return dst
end

function localfilter!(dst::AbstractArray{T,N},
                      A::AbstractArray{T,N},
                      dims::Union{AbstractVector{<:Integer},
                                  Tuple{Vararg{Integer}}},
                      op::Function,
                      rngs::Union{AbstractVector{<:IndexInterval},
                                  Tuple{Vararg{IndexInterval}}},
                      w::Vector{T} = workspace(A, dims, rngs)) where {T,N}
    (m = length(dims)) == length(rngs) || throw(DimensionMismatch("list of dimensions and list of intervals must have the same length"))
    if m ≥ 1
        localfilter!(dst, A, dims[1], op, rngs[1], w)
        for d in 2:m
            localfilter!(dst, dims[d], op, rngs[d], w)
        end
    else
        copyto!(dst, A)
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
            return localfilter(A, dims, $op, rngs, args...)
        end

        function $fp(A::AbstractArray{T,N}, d::Integer,
                     dims::Union{Colon, Integer, Tuple{Vararg{Integer}},
                                 AbstractVector{<:Integer}},
                     rngs::Union{IndexInterval, Tuple{Vararg{IndexInterval}},
                                 AbstractVector{<:IndexInterval}},
                     args...) where {T,N}
            return localfilter!(A, dims, $op, rngs, args...)
        end

        function $fp(dst::AbstractArray{T,N},
                     A::AbstractArray{T,N}, d::Integer,
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

```julia
workspacelength(n, p)
```

yields the minimal length of the workspace array for applying the van
Herk-Gil-Werman algorithm along a dimension of length `n` with a structuring
element of width `p`.  If `n < 1` or `p ≤ 1`, zero is returned because there is
no needs for a workspace array.

```julia
workspacelength(A, dims, rngs)
```

yields the minimal length of the workspace array for applying the van
Herk-Gil-Werman algorithm along the dimension(s) `dims` of array `A` with a
structuring element defined by the interval(s) `rngs`.  If arguments are not
compatible, ero is returned because there is no needs for a workspace array.

"""
workspacelength(n::Int, p::Int) = (n < 1 || p ≤ 1 ? 0 : n + 2*(p - 1))

function workspacelength(A::AbstractArray{<:Any,N},
                         d::Integer,
                         rng::IndexInterval) where {N}
    if 1 ≤ d ≤ N
        n = length(axes(A, d))
        p = length(rng)
        return workspacelength(n, p)
    end
    return 0
end

function workspacelength(A::AbstractArray{<:Any,N},
                         ::Colon,
                         rng::IndexInterval) where {N}
    if N ≥ 1
        p = length(rng)
        n = reduce(max, map(length, axes(A)))
        return workspacelength(n, p)
    end
    return 0
end

function workspacelength(A::AbstractArray{<:Any,N},
                         ::Colon,
                         rngs::Union{AbstractVector{<:IndexInterval},
                                     Tuple{Vararg{IndexInterval}}}) where {N}
    numb = 0
    if length(rngs) == N
        for d in 1:N
            n = length(axes(A, d))
            p = length(rngs[d])
            numb = max(numb, workspacelength(n, p))
        end
    end
    return numb
end

function workspacelength(A::AbstractArray{<:Any,N},
                         dims::Union{AbstractVector{<:Integer},
                                     Tuple{Vararg{Integer}}},
                         rng::IndexInterval) where {N}
    numb = 0
    p = length(rng)
    for dim in dims
        n = length(axes(A, dim))
        numb = max(numb, workspacelength(n, p))
    end
    return numb
end

function workspacelength(A::AbstractArray{<:Any,N},
                         dims::Union{AbstractVector{<:Integer},
                                     Tuple{Vararg{Integer}}},
                         rngs::Union{AbstractVector{<:IndexInterval},
                                     Tuple{Vararg{IndexInterval}}}) where {N}
    numb = 0
    if (m = length(dims)) == length(rngs)
        for d in 1:m
            n = length(axes(A, dims[d]))
            p = length(rngs[d])
            numb = max(numb, workspacelength(n, p))
        end
    end
    return numb
end
"""

```julia
workspace([T,] A, dims, rngs)
```

yields a workspace array for applying the van Herk-Gil-Werman algorithm along
the dimension(s) `dims` of array `A` with a structuring element defined by the
interval(s) `rngs`.  The element type of the workspace is `T` which is that of
`A` by default.

"""
function workspace(A::AbstractArray{T,N},
                   dims::Union{Colon, Integer, AbstractVector{<:Integer},
                               Tuple{Vararg{Integer}}},
                   rngs::Union{IndexInterval, AbstractVector{<:IndexInterval},
                               Tuple{Vararg{IndexInterval}}}) where {T,N}
    return workspace(T, A, dims, rngs)
end

function workspace(::Type{T},
                   A::AbstractArray{<:Any,N},
                   dims::Union{Colon, Integer, AbstractVector{<:Integer},
                               Tuple{Vararg{Integer}}},
                   rngs::Union{IndexInterval, AbstractVector{<:IndexInterval},
                               Tuple{Vararg{IndexInterval}}}) where {T,N}
    return Array{T}(undef, workspacelength(A, dims, rngs))
end
