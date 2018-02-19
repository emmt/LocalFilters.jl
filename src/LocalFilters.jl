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
# Copyright (C) 2017, Éric Thiébaut.
#

isdefined(Base, :__precompile__) && __precompile__(true)

module LocalFilters

import Base: CartesianRange, eltype, ndims, size, length, first, last, tail,
    getindex, setindex!, convert

export
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

include("types.jl")
include("basics.jl")
include("filters.jl")
include("morphology.jl")

end
