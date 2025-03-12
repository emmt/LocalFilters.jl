module LocalFilters

include("compat.jl")

export
    # Index ordering in filters.
    FORWARD_FILTER,
    REVERSE_FILTER,

    # Shift-invariant linear filters.
    correlate, correlate!,
    convolve, convolve!,
    localmean, localmean!,

    # Mathematical morphology.
    erode, erode!,
    dilate, dilate!,
    localextrema, localextrema!,
    closing, closing!,
    opening, opening!,
    bottom_hat, bottom_hat!,
    top_hat, top_hat!,

    # Other non-linear filters.
    bilateralfilter, bilateralfilter!,

    # Kernels and neighborhoods.
    kernel,
    reverse_kernel,
    strel,

    # Generic filters.
    localfilter, localfilter!,
    localmap, localmap!,
    localreduce, localreduce!

using OffsetArrays, StructuredArrays, EasyRanges, TypeUtils
using EasyRanges: ranges
using Base: @propagate_inbounds, tail, OneTo

function bilateralfilter end
function bilateralfilter! end

include("types.jl")
include("utils.jl")
include("generic.jl")
include("localmap.jl")
include("localreduce.jl")
include("linear.jl")
include("morphology.jl")
include("bilateral.jl")

@deprecate ForwardFilter FORWARD_FILTER false
@deprecate ReverseFilter REVERSE_FILTER false
@deprecate(localfilter(A::AbstractArray, dims::Dimensions, op::Function, rngs::Ranges; kwds...),
           localreduce(op, A, dims, rngs; kwds...), false)
@deprecate(localfilter(T::Type, A::AbstractArray, dims::Dimensions, op::Function, rngs::Ranges; kwds...),
           localreduce(op, T, A, dims, rngs; kwds...), false)
@deprecate(localfilter!(A::AbstractArray, dims::Dimensions, op::Function, rngs::Ranges; kwds...),
           localreduce!(op, A, dims, rngs; kwds...), false)
@deprecate(localfilter!(dst::AbstractArray, A::AbstractArray, dims::Dimensions, op::Function, rngs::Ranges; kwds...),
           localreduce!(op, dst, A, dims, rngs; kwds...), false)

end
