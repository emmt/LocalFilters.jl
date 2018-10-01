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



# FIXME: Allocating a vector and then resizing it is almost as fast as directly
#        allocating to the given size, except for very small vectors.  So the
#        default workspace is an empty vector which is automatically resized as
#        needed.

# FIXME: When multiple filtering passes are planned, resize the workspace
#        at most once.

"""
# Local filter by the van Herk-Gil-Werman algorithm

```julia
localfilter!([dst = A,] A, dims, op, rngs [, w])
```

overwrite the contents of `dst` with the result of applying van Herk-Gil-Werman
algorithm to filter array `A` along dimension(s) `dims` with (associative)
binary operation `op` and a contiguous structuring element defined by the
interval(s) `rngs`.  Optional argument `w` is a workspace array which is
automatically allocated if not provided; otherwise, it must be a vector of same
element type as `A` which may be resized (with [`resize!`](@ref) if its length
is insufficient).  The destination `dst` must have the same indices as the
source `A`.  Operation can be done in-place; that is, `dst` and `A` can be the
same.

Argument `dims` specifies along which dimension(s) of `A` the filter is to be
applied, it can be a single integer, several integers or a colon `:` to specify
all dimensions.  Dimensions are processed in the order given by `dims` (the
same dimension may appear several times) and there must be a matching interval
in `rngs` to specify the structuring element (except that if `rngs` is a single
interval, it is used for every dimension in `dims`).  An interval is either an
integer or an integer unit range in the form `kmin:kmax`.

Assuming mono-dimensional arrays `A` and `dst` and a single filtering pass, the
result is:

```
dst[j] = A[j-kmax] ⋄ A[j-kmax+1] ⋄ A[j-kmax+2] ⋄ ... ⋄ A[j-kmin]
```

for all `j ∈ [first(axes(A,d)):last(axes(A,d))]`, with `x ⋄ y = op(x, y)`,
`kmin = first(rng)` and `kmax = last(rng)`.  Note that if `kmin = kmax` (which
occurs if `rng` is a simple integer), the result of the filter is to operate a
simple shift by `kmin` along the dimension `d`.

The in-place *erosion* (local minimum) of the monodimensional array `A` on a
centered structuring element of width 7 can be computed as:

```julia
localfilter!(A, :, min, -3:3)
```

To apply the same filter along all dimensions (with different
index intervals for the structuring element), call:

```julia
localfilter!([dst = A,] A, :, op, rngs [, w])
```

with `inds` a tuple of index intervals to specify the structuring element along
each dimension of `A`.  Specify index interval `0` to do nothing along the
corresponding dimension.  For instance:

```julia
localfilter!(A, :, max, (-3:3, 0, -4:4))
```

will overwrite `A` with the local maxima of the three-dimensional array `A` in
a centered local neighborhood of size `7×1×9` (nothing is done along the second
dimension).  The same result may be obtained with:

```julia
localfilter!(A, (1,3), max, (-3:3, -4:4))
```

The out-place version, allocates the destination array and is called as:

```julia
localfilter(A, dims, op, rngs [, w])
```

For instance, the local average of the two-dimensional array `A` on a centered
structuring element of size 11×11 can be computed as:

```julia
localfilter(A, :, +, (-5:5, -5:5))*(1/11)
```

The van Herk-Gil-Werman algorithm is very fast for large rectangular
structuring elements.  It takes at most 3 operations to filter an element along
a given dimension whatever the width `p` of the considered local neighborhood.
For `N`-dimensional arrays, the algorithm requires only `3N` operations per
element instead of `p^N - 1` operations for a naive implementation.  This
however requires to make a pass along each dimension so memory page faults may
reduce the performances.  This is however attenuated by the fact that the
algorithm can be applied in-place.  For efficient multi-dimensional out of
place filtering, make the first pass with a fresh destination array and then
all other passes in-place on the destination array.

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
    1 ≤ d ≤ N || throw(BoundsError("out of bounds dimension index"))
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
        rng = (k < 0 ? (jmin:jmax) : (jmax:-1:jmin))
        @inbounds for J2 in R2, J1 in R1
            @simd for j in rng
                # FIXME: This loop is (a bit) faster with clamp instead of
                #        min-max and with SIMD.
                jp = clamp(j - k, jmin, jmax)
                dst[J1,j,J2] = A[J1,jp,J2]
            end
        end
        return dst
    end

    # Make sure workspace array is large enough to temporarily store
    # W[1:p-1] ≡ R[1:p-1] and W[j-k+off] ≡ A[j-k] for all possible
    # j ∈ [jmin,jmax] and k ∈ [kmin,kmax] and according to boundary
    # conditions.
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
        for i in imin:imax
            # FIXME: This loop is (a bit) faster with min-max instead of clamp
            #        and no SIMD.
            j = min(max(i - off, jmin), jmax)
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
    return dst
end


# Provide destination.

function localfilter(A::AbstractArray{T,N},
                     dims::Union{Colon, Integer, Tuple{Vararg{Integer}},
                                 AbstractVector{<:Integer}},
                     op::Function, args...) where {T,N}
    return localfilter!(similar(Array{T,N}, axes(A)), A, dims, op, args...)
end

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
                      w::Vector{T} = Array{T}(undef,0)) where {T,N}
    return localfilter!(dst, A, Int(d), op, Int(first(rng)), Int(last(rng)), w)
end

function localfilter!(dst::AbstractArray{T,N},
                      A::AbstractArray{T,N},
                      ::Colon,
                      op::Function,
                      rng::IndexInterval,
                      w::Vector{T} = Array{T}(undef,0)) where {T,N}
    kmin, kmax = Int(first(rng)), Int(last(rng))
    localfilter!(dst, A, 1, op, kmin, kmax, w)
    for d in 2:N
        localfilter!(dst, d, op, kmin, kmax, w)
    end
    return dst
end

function localfilter!(dst::AbstractArray{T,N},
                      A::AbstractArray{T,N},
                      ::Colon,
                      op::Function,
                      rngs::Union{AbstractVector{<:IndexInterval},
                                  Tuple{Vararg{IndexInterval}}},
                      w::Vector{T} = Array{T}(undef,0)) where {T,N}
    length(rngs) == N || throw(DimensionMismatch("there must be as many intervals as dimensions"))
    localfilter!(dst, A, 1, op, rngs[1], w)
    for d in 2:N
        localfilter!(dst, d, op, rngs[d], w)
    end
    return dst
end

function localfilter!(dst::AbstractArray{T,N},
                      A::AbstractArray{T,N},
                      dims::Union{AbstractVector{<:Integer},
                                  Tuple{Vararg{Integer}}},
                      op::Function,
                      rng::IndexInterval,
                      w::Vector{T} = Array{T}(undef,0)) where {T,M,N}
    m = length(dims)
    if m ≥ 1
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
                      w::Vector{T} = Array{T}(undef,0)) where {T,M,N}
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


# Wrapper methods for in-place operation.

function localfilter!(A::AbstractArray{T,N},
                      d::Integer,
                      op::Function,
                      rng::IndexInterval,
                      w::Vector{T} = Array{T}(undef,0)) where {T,N}
    return localfilter!(A, Int(d), op, Int(first(rng)), Int(last(rng)), w)
end

function localfilter!(A::AbstractArray{T,N}, ::Colon,
                      op::Function, rng::IndexInterval,
                      w::Vector{T} = Array{T}(undef,0)) where {T,N}
    kmin, kmax = Int(first(rng)), Int(last(rng))
    for d in 1:N
        localfilter!(A, d, op, kmin, kmax, w)
    end
    return A
end

function localfilter!(A::AbstractArray{T,N},
                      ::Colon,
                      op::Function,
                      rngs::Union{AbstractVector{<:IndexInterval},
                                  Tuple{Vararg{IndexInterval}}},
                      w::Vector{T} = Array{T}(undef,0)) where {T,N}
    length(rngs) == N || throw(DimensionMismatch("there must be as many intervals as dimensions"))
    for d in 1:N
        localfilter!(A, d, op, rngs[d], w)
    end
    return A
end

function localfilter!(A::AbstractArray{T,N},
                      dims::Union{AbstractVector{<:Integer},
                                  Tuple{Vararg{Integer}}},
                      op::Function,
                      rng::IndexInterval,
                      w::Vector{T} = Array{T}(undef,0)) where {T,M,N}
    m = length(dims)
    for d in 1:m
        localfilter!(A, dims[d], op, rng, w)
    end
    return A
end

function localfilter!(A::AbstractArray{T,N},
                      dims::Union{AbstractVector{<:Integer},
                                  Tuple{Vararg{Integer}}},
                      op::Function,
                      rngs::Union{AbstractVector{<:IndexInterval},
                                  Tuple{Vararg{IndexInterval}}},
                      w::Vector{T} = Array{T}(undef,0)) where {T,M,N}
    (m = length(dims)) == length(rngs) || throw(DimensionMismatch("list of dimensions and list of intervals must have the same length"))
    for d in 1:m
        localfilter!(A, dims[d], op, rngs[d], w)
    end
    return A
end



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

`workspacelength(n, p)` yields the minimal length of the workspace array for
applying the van Herk-Gil-Werman algorithm along a dimension of length `n` with
a structuring element of width `p`.  If `n < 1` or `p ≤ 1`, zero is returned
because there is no needs for a workspace array.

"""
workspacelength(n::Int, p::Int) = (n < 1 || p ≤ 1 ? 0 : n + 2*(p - 1))
