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
    bilateralfilter!,
    bilateralfilter,
    localfilter!,
    localmean,
    localmean!,
    convolve,
    convolve!,
    erode,
    erode!,
    dilate,
    dilate!,
    closing,
    closing!,
    opening,
    opening!,
    top_hat,
    bottom_hat,
    localextrema,
    localextrema!

import Base: eltype, ndims, size, length, first, last, tail,
    getindex, setindex!, convert

# Deal with compatibility issues.
using Compat
@static if isdefined(Base, :CartesianIndices)
    import Base: CartesianIndices
else
    import Base: CartesianRange
    import Compat: CartesianIndices
end
@static if isdefined(Base, :axes)
    import Base: axes
else
    import Base: indices
    const axes = indices
end

include("types.jl")
include("basics.jl")
include("filters.jl")
include("morphology.jl")
include("bilateral.jl")

end
