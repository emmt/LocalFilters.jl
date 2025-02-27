#
# LocalFilters.jl --
#
# Local filters for Julia arrays.
#
#-----------------------------------------------------------------------------------------
#
# This file is part of the `LocalFilters.jl` package licensed under the MIT "Expat"
# License.
#
# Copyright (c) 2017-2025, Éric Thiébaut.
#

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
    localmap, localmap!

using OffsetArrays, StructuredArrays, EasyRanges, TypeUtils
using EasyRanges: ranges
using Base: @propagate_inbounds, tail, OneTo

function bilateralfilter end
function bilateralfilter! end

@deprecate ForwardFilter FORWARD_FILTER true
@deprecate ReverseFilter REVERSE_FILTER true

include("types.jl")
include("utils.jl")
include("generic.jl")
include("localmap.jl")
include("linear.jl")
include("morphology.jl")
include("bilateral.jl")
include("separable.jl")

end
