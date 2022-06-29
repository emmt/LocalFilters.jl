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

export
    # Re-exports from Base.Order
    Forward,
    Reverse,

    # Exports from this package.
    convolve!,
    convolve,
    correlate!,
    correlate,
    kernel,
    localfilter!,
    localfilter,
    localmean!,
    localmean,
    strel

    #bilateralfilter!,
    #bilateralfilter,
    #bottom_hat,
    #closing!,
    #closing,
    #dilate!,
    #dilate,
    #erode!,
    #erode,
    #localextrema!,
    #localextrema,
    #opening!,
    #opening,
    #top_hat

using OffsetArrays, StructuredArrays, EasyRanges
using EasyRanges: ranges, to_int
using Base.Order # yields Ordering, Forward, Reverse
using Base: @propagate_inbounds, tail

function localfilter end
function localfilter! end

include("types.jl")
include("basics.jl")
include("generic.jl")
#include("separable.jl")
include("linear.jl")
#include("morphology.jl")
#include("bilateral.jl")
#import .BilateralFilter: bilateralfilter!, bilateralfilter

end
