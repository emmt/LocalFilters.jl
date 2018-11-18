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
# Copyright (C) 2017-2018, Éric Thiébaut.
#

isdefined(Base, :__precompile__) && __precompile__(true)

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

import Base: eltype, ndims, size, length, first, last, tail,
    getindex, setindex!, convert, reverse

# Deal with compatibility issues.
using Compat
const USE_CARTESIAN_RANGE = !isdefined(Base, :CartesianIndices)
@static if USE_CARTESIAN_RANGE
    import Base: CartesianRange
    import Compat: CartesianIndices
else
    import Base: CartesianIndices
end
@static if isdefined(Base, :axes)
    import Base: axes
else
    import Base: indices
    const axes = indices
end
@static if VERSION < v"0.7.0-alpha"
    Base.Tuple(index::CartesianIndex) = index.I
end

include("types.jl")
include("basics.jl")
include("filters.jl")
include("separable.jl")
include("morphology.jl")
include("bilateral.jl")
import .BilateralFilter: bilateralfilter!, bilateralfilter

end
