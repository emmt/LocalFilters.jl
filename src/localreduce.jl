# Union of for possible types to specify the dimension(s) of interest.
const Dimensions = Union{Colon, Integer, Tuple{Vararg{Integer}},
                         AbstractVector{<:Integer}}

# Union of for possible types to specify the neighborhoods ranges.
const Ranges = Union{Axis, Tuple{Vararg{Axis}}, AbstractVector{<:Axis}}

"""
    WorkVector(buf, len, skip=0) -> A
    WorkVector(buf, rng, skip=0) -> A

yield an abstract vector `A` of length `len` or of indices given by the range `rng` which
share its elements with the buffer `buf`. The first entry of `A` is the `skip+1`-th entry
of `buf`. If `buf` is not large enough, it is automatically resized.

"""
struct WorkVector{T} <: DenseVector{T}
    parent::Vector{T}
    indices::UnitRange{Int}
    offset::Int
    function WorkVector(buf::Vector{T},
                        rng::UnitRange{Int},
                        skip::Int = 0) where {T}
        skip ≥ 0 || throw(ArgumentError("number of elements to skip must be nonnegative"))
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
    localreduce(op, [T=eltype(A),] A, dims, rngs; kwds...) -> dst

yields the local reduction by the associative binary operator `op` of the values of `A`
into contiguous hyper-rectangular neighborhoods defined by the interval(s) `rngs` along
dimension(s) `dims` of `A`. The algorithm of van Herk-Gil-Werman is used to compute the
reduction.

Optional argument `T` is to specify the element type of the result.

Argument `dims` specifies along which dimension(s) of `A` the local reduction is to be
applied, it can be a single integer, a tuple of integers, or a colon `:` to apply the
operation to all dimensions. Dimensions are processed in the order given by `dims` (the
same dimension may appear several times) and there must be a matching interval in `rngs`
to specify the structuring element (except that, if `rngs` is a single interval, it is
used for every dimension in `dims`). An interval is either an integer or an integer valued
unit range in the form `kmin:kmax`. An interval specified as a single integer is treated
as an approximately centered range of this length.

Keyword `order` specifies the filter direction, `FORWARD_FILTER` by default.

Keyword `work` may be used to provide a workspace array of type `Vector{T}` which is
automatically resized as needed.

Assuming a mono-dimensional array `A`, the single reduction pass:

    dst = localreduce(op, A, :, rng)

amounts to computing (assuming forward ordering):

    dst[j] =  A[i+kmin] ⋄ A[i+kmin+1] ⋄ ... ⋄ A[i+kmax-1] ⋄ A[i+kmax]

for all `j ∈ axes(dst,1)`, with `x ⋄ y = op(x, y)`, `kmin = first(rng)` and `kmax =
last(rng)`. Note that if `kmin = kmax = k`, the result of the filter is to operate a
simple shift by `k` along the corresponding dimension and has no effects if `k = 0`. This
can be exploited to not filter some dimension(s).

Flat boundary conditions are assumed for `A[i+k]` in the above formula.

## Examples

The *morphological erosion* (local minimum) of the array `A` on a centered structuring
element of width 7 in every dimension can be obtained by:

    localreduce(min, A, :, -3:3)

or equivalently by:

    localreduce(min, A, :, 7)

Index interval `0:0` may be specified to do nothing along the corresponding dimension. For
instance, assuming `A` is a three-dimensional array:

    localreduce(max, A, :, (-3:3, 0:0, -4:4))

yields the *morphological dilation* (*i.e.* local maximum) of `A` in a centered local
neighborhood of size `7×1×9` (nothing is done along the second dimension). The same result
may be obtained with:

    localreduce(max, A, (1,3), (-3:3, -4:4))

where the second dimension is omitted from the list of dimensions.

The *local average* of the two-dimensional array `A` on a centered sliding window of size
11×11 can be computed as:

    localreduce(+, A, :, (-5:5, -5:5)) ./ 11^2

See [`localreduce!`](@ref) for an in-place version of the method.

"""
localreduce(op, A::AbstractArray, dims::Dimensions, rngs::Ranges; kwds...) =
    localreduce(op, eltype(A), A, dims, rngs; kwds...)

localreduce(op, ::Type{T}, A::AbstractArray, dims::Dimensions, rngs::Ranges; kwds...) where {T} =
    localreduce!(op, similar(A, T), A, dims, rngs; kwds...)

"""
    localreduce!(op, [dst = A,] A, dims, rngs; kwds...)

overwrites the contents of `dst` with the local reduction by the associative binary
operator `op` of the values of `A` into contiguous hyper-rectangular neighborhoods defined
by the interval(s) `rngs` along dimension(s) `dims` of `A`. Except if a single dimension
of interest is specified by `dims`, the destination `dst` must have the same indices as
the source `A` (that is, `axes(dst) == axes(A)`). Operation may be done in-place and `dst`
and `A` can be the same; this is the default behavior if `dst` is not specified. The
algorithm of van Herk-Gil-Werman is used to compute the reduction.

See [`localreduce`](@ref) for a full description of the method and for accepted keywords.

The in-place *morphological erosion* (local minimum) of the array `A` on a centered
structuring element of width 7 in every dimension can be obtained by:

    localreduce!(min, A, :, -3:3)

"""
localreduce!(op, A::AbstractArray, dims::Dimensions, rngs::Ranges; kwds...) =
    localreduce!(op, A, A, dims, rngs; kwds...)

function localreduce!(op, dst::AbstractArray{T,N}, A::AbstractArray{<:Any,N},
                      dims::Dimensions, rngs::Ranges;
                      work::Vector{T} = Vector{T}(undef, workspace_length(A, dims, rngs)),
                      order::FilterOrdering = FORWARD_FILTER) where {T,N}
    _reduce!(op, dst, A, dims, order, rngs, work)
    return dst
end

# Wrapper methods when destination, ordering, and workspace are specified.
#
# The reduction is applied along all chosen dimensions. To avoid page memory faults, the
# operation is performed out-of-place (unless dst and A are the same) for the first chosen
# dimension and the operation is performed in-place for the other chosen dimensions.
#
# Reduce along all dimensions (`dims` is a colon).
function _reduce!(op, dst::AbstractArray{T,N}, A::AbstractArray{<:Any,N}, ::Colon,
                  ord::FilterOrdering, rngs::Ranges, wrk::Vector{T}) where {T,N}
    if rngs isa Union{Integer,AbstractRange{<:Integer}}
        rng = kernel_range(ord, rngs) # optimization: convert once
        if N ≥ 1
            _reduce!(op, dst, A, 1, rng, wrk)
            for dim in 2:N
                _reduce!(op, dst, dst, dim, rng, wrk)
            end
        end
    else
        length(rngs) == N || throw(DimensionMismatch(
            "there must be as many intervals as dimensions"))
        if N ≥ 1
            _reduce!(op, dst, A, 1, kernel_range(ord, first(rngs)), wrk)
            for (dim, rng) in enumerate(rngs)
                dim > 1 && _reduce!(op, dst, dst, dim, kernel_range(ord, rng), wrk)
            end
        end
    end
end
#
# Reduce along a single given dimension (`dims` is an integer).
function _reduce!(op, dst::AbstractArray{T,N}, A::AbstractArray{<:Any,N}, dim::Integer,
                  ord::FilterOrdering, rngs::Ranges, wrk::Vector{T}) where {T,N}
    if rngs isa Union{Integer,AbstractRange{<:Integer}}
        rng = kernel_range(ord, rngs)
    else
        length(rngs) == 1 || throw(DimensionMismatch(
            "there must be as many intervals as dimensions to filter"))
        rng = kernel_range(ord, first(rngs))
    end
    _reduce!(op, dst, A, dim, rng, wrk)
end
#
# Reduce along multiple given dimensions (`dims` is a list of integers).
function _reduce!(op, dst::AbstractArray{T,N}, A::AbstractArray{<:Any,N}, dims::Dimensions,
                  ord::FilterOrdering, rngs::Ranges, wrk::Vector{T}) where {T,N}
    i, j = firstindex(dims), lastindex(dims)
    if rngs isa Union{Integer,AbstractRange{<:Integer}}
        if i ≤ j
            rng = kernel_range(ord, rngs) # optimization: convert once
            _reduce!(op, dst, A, @inbounds(dims[i]), rng, wrk)
            for k in i+1:j
                _reduce!(op, dst, dst, @inbounds(dims[k]), rng, wrk)
            end
        end
    else
        length(dims) == length(rngs) || throw(DimensionMismatch(
            "there must be as many intervals as dimensions to filter"))
        if i ≤ j
            off = firstindex(rngs) - i
            _reduce!(op, dst, A, @inbounds(dims[i]), kernel_range(ord, @inbounds(rngs[off+i])), wrk)
            for k in i+1:j
                _reduce!(op, dst, dst, @inbounds(dims[k]), kernel_range(ord, @inbounds(rngs[off+k])), wrk)
            end
        end
    end
end

# This is the most basic version which is called by other versions.
function _reduce!(op, dst::AbstractArray{T,N}, A::AbstractArray{<:Any,N}, dim::Integer,
                  rng::AbstractUnitRange{Int}, wrk::Vector{T}) where {T,N}
    1 ≤ dim ≤ N || throw(ArgumentError("out of bounds dimension"))
    isempty(rng) && throw(ArgumentError("invalid filter size"))
    _reduce!(op, dst, A, Val{Int(dim)}(), rng, wrk)
end

# This version is to break the type instability related to the variable dimension of
# interest.
function _reduce!(op, dst::AbstractArray{T,N}, A::AbstractArray{<:Any,N}, ::Val{D},
                  K::AbstractUnitRange{Int}, wrk::Vector{T}) where {T,N,D}
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
        _shiftarray!(dst, I1, I, I2, A, B, first(K))
    else
        # Apply van Herk-Gil-Werman algorithm.
        _reduce!(op, dst, I1, I, I2, A, B, K, wrk)
    end
end

function _shiftarray!(dst::AbstractArray{<:Any,N},
                      I1, I::AbstractUnitRange{Int}, I2,
                      src::AbstractArray{<:Any,N},
                      B::BoundaryConditions,
                      k::Int) where {N}
    @inbounds for i2 ∈ I2, i1 ∈ I1
        # To allow for in-place operation without using any temporaries, we walk along the
        # dimension in the forward direction if the shift is nonnegative and in the
        # reverse direction otherwise.
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

function _reduce!(op, dst::AbstractArray{T,N},
                  I1, # range of leading indices
                  I::AbstractUnitRange{Int}, # index range in dst
                  I2, # range of trailing indices
                  src::AbstractArray{<:Any,N},
                  B::BoundaryConditions, # boundary conditions in src
                  K::AbstractUnitRange{Int},
                  R::Vector{T}) where {T,N}
    # The code assumes that the workspace R has 1-based indices, all other arrays may have
    # any indices.
    @assert firstindex(R) == 1

    # Get ranges for indices.
    imin, imax = first(I), last(I) # destination index range
    kmin, kmax = first(K), last(K) # neighborhood index range
    n = length(K) # length of neighborhood

    # Make a work vector for storing values from the source accounting for boundary
    # conditions and leaving n values for the array R.
    L = (imin + kmin):(imax + kmax)
    A = WorkVector(R, L, n)

    # The van Herk-Gil-Werman algorithm
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #
    # A monodimensional local filter involving binary operation ⋄ on index range K =
    # kmin:kmax yields the following array C when filtering the array A (with forward
    # ordering):
    #
    #     C[i] = A[i+kmin] ⋄ A[i+kmin+1] ⋄ ... ⋄ A[i+kmax-1] ⋄ A[i+kmax]
    #
    # Assuming ⋄ is an associative binary operation, van Herk-Gil-Werman (HGW) algorithm
    # consists in building 2 auxiliary arrays, R and S, as follows:
    #
    #     R[1] = A[i+kmax-1]
    #     R[j] = A[i+kmax-j] ⋄ R[j-1]   ∀ j ∈ 2:n-1
    #
    # and:
    #
    #     S[1]   = A[i+kmax]
    #     S[j+1] = S[j] ⋄ A[i+kmax+j]      ∀ j ∈ 1:n-1
    #
    # with n = length(K) the length of the structuring element. Graphically (for n = 7):
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
    # where each □ represents a cell in A. Thanks to the recursion, this only takes 2n - 3
    # operations ⋄. Then, assuming the result is stored in array C, we have:
    #
    #     C[i+j-1] = R[n-j]⋄S[j]   ∀ j ∈ 1:n-1
    #     C[i+n-1] = S[n]
    #
    # which requires n - 1 more operations ⋄. So the total number of operations ⋄ is 3n -
    # 4 for computing n values in C, hence 3 - 4/n operations per cell on average.
    #
    # The following implementation exploits a single workspace array to temporarily store
    # A[i] to allow for in-place application of the filter and with supplementary values
    # to account for boundary conditions. The workspace array is also used to store R[j]
    # while the values of S[j] are computed on the fly.
    #

    @inbounds for i2 ∈ I2, i1 ∈ I1
        # Fill the workspace with the source for all possible indices and taking care of
        # boundary conditions.
        @simd for l ∈ L
            A[l] = src[i1,B(l),i2]
        end

        # Process the input by blocks of as much as n elements (less at the end of the
        # range).
        for i = imin:n:imax
            # Initialize auxiliary array R.
            R[1] = A[i+kmax-1]
            @simd for j ∈ 2:n-1
                R[j] = op(A[i+kmax-j], R[j-1])
            end
            # Apply the recursion to compute at least 1 and at most n resulting values.
            jmax = imax-i+1 # max. value for j in dst[i1,i+j-1,i2]
            # First output (j = 1).
            s = A[i+kmax]                # S[1]
            dst[i1,i,i2] = op(R[n-1], s) # C[i] = R[n-1] ⋄ S[1]
            # Intermediate outputs (j ∈ 2:n-1).
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

        $f(A::AbstractArray, dims::Dimensions, rngs::Ranges; kwds...) =
            localreduce($op, A, dims, rngs; kwds...)

        $f!(A::AbstractArray, dims::Dimensions, rngs::Ranges, kwds...) =
            localreduce!($op, A, dims, rngs; kwds...)

        function $f!(dst::AbstractArray{<:Any,N}, A::AbstractArray{<:Any,N},
                     dims::Dimensions, rngs::Ranges, kwds...) where {N}
            return localreduce!($op, dst, A, dims, rngs; kwds...)
        end

    end
end

"""
    workspace_length(n, p)

yields the minimal length of the workspace array for applying the van Herk-Gil-Werman
algorithm along a dimension of length `n` with a structuring element of width `p`. If `n <
1` or `p ≤ 1`, zero is returned because a workspace array is not needed.

    workspace_length(A, dims, rngs)

yields the minimal length of the workspace array for applying the van Herk-Gil-Werman
algorithm along the dimension(s) `dims` of array `A` with a structuring element defined by
the interval(s) `rngs`. If arguments are not compatible, zero is returned because there is
no needs for a workspace array.

"""
workspace_length(n::Int, p::Int) = ifelse((n < 1)|(p ≤ 1), 0, n + 2*(p - 1))
workspace_length(n::Integer, p::Integer) = workspace_length(Int(n), Int(p))

# `dims` is a colon.
workspace_length(A::AbstractArray, ::Colon, rng::AbstractRange{<:Integer}) =
    workspace_length(A, :, length(rng))
workspace_length(A::AbstractArray, ::Colon, len::Integer) =
    ndims(A) ≥ 1 ? workspace_length(maximum(size(A)), len) : 0
function workspace_length(A::AbstractArray, ::Colon, rngs::Ranges)
    result = 0
    if length(rngs) == ndims(A)
        for (dim, rng) in enumerate(rngs)
            result = max(result, workspace_length(A, dim, rng))
        end
    end
    return result
end

# `dims` is a single dimension.
workspace_length(A::AbstractArray, dim::Integer, rng::AbstractRange{<:Integer}) =
    workspace_length(A, dim, length(rng))
workspace_length(A::AbstractArray, dim::Integer, len::Integer) =
    1 ≤ dim ≤ ndims(A) ? workspace_length(size(A, dim), len) : 0
workspace_length(A::AbstractArray, dim::Integer, rngs::Ranges) =
    length(rngs) == 1 ? workspace_length(A, dim, first(rngs)) : 0

# `dims` is a list of dimensions.
workspace_length(A::AbstractArray, dims::Dimensions, rng::AbstractRange{<:Integer}) =
    workspace_length(A, dims, length(rng))
function workspace_length(A::AbstractArray, dims::Dimensions, len::Integer)
    result = 0
    for dim in dims
        result = max(result, workspace_length(A, dim, len))
    end
    return result
end
function workspace_length(A::AbstractArray, dims::Dimensions, rngs::Ranges)
    result = 0
    if length(rngs) == length(dims)
        for (dim, rng) in zip(dims, rngs)
            result = max(result, workspace_length(A, dim, rng))
        end
    end
    return result
end
