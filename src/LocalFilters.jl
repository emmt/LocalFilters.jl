#
# LocalFilters.jl --
#
# Local filters for Julia arrays.
#
#------------------------------------------------------------------------------
#
# This file is part of the `LocalFilters.jl` package licensed under the MIT
# "Expat" License.
#
# Copyright (C) 2017-2022, Éric Thiébaut.
#

module LocalFilters

include("compat.jl")

export
    ForwardFilter,
    ReverseFilter,
    bilateralfilter!,
    bilateralfilter,
    bottom_hat,
    closing!,
    closing,
    convolve!,
    convolve,
    correlate!,
    correlate,
    dilate!,
    dilate,
    erode!,
    erode,
    kernel,
    localextrema!,
    localextrema,
    localfilter!,
    localfilter,
    localmean!,
    localmean,
    opening!,
    opening,
    strel,
    top_hat

using OffsetArrays, StructuredArrays, EasyRanges
using EasyRanges: ranges
using Base: @propagate_inbounds, tail, OneTo

function localfilter end
function localfilter! end
function erode end
function erode! end
function dilate end
function dilate! end
function bilateralfilter end
function bilateralfilter! end

include("types.jl")
include("utils.jl")
include("generic.jl")
include("linear.jl")
include("separable.jl")
include("morphology.jl")
include("bilateral.jl")

end
