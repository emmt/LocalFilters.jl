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
# Copyright (C) 2017-2020, Éric Thiébaut.
#

module LocalFilters

export
    Neighborhood,
    RectangularBox,
    bilateralfilter!,
    bilateralfilter,
    bottom_hat,
    closing!,
    closing,
    convolve!,
    convolve,
    dilate!,
    dilate,
    erode!,
    erode,
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

import Base: CartesianIndices,
    axes, eltype, ndims, size, length, first, last, tail,
    getindex, setindex!, convert, reverse

function localfilter end
function localfilter! end

include("types.jl")
include("basics.jl")
include("filters.jl")
include("morphology.jl")
include("separable.jl")
include("bilateral.jl")
import .BilateralFilter: bilateralfilter!, bilateralfilter

end
