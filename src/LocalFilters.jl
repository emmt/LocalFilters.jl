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
    ForwardFilter,
    ReverseFilter,
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
using Base: @propagate_inbounds, tail

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
#include("bilateral.jl")
#import .BilateralFilter: bilateralfilter!, bilateralfilter

end
