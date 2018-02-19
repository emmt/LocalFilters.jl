#
# types.jl --
#
# Type definitions for local filters.
#
#------------------------------------------------------------------------------
#
# This file is part of the `LocalFilters.jl` package licensed under the MIT
# "Expat" License.
#
# Copyright (C) 2017-2018, Éric Thiébaut.
#

"""
All neighborhoods are instances of a type derived from `Neighborhood`.
"""
abstract type Neighborhood{N}; end

"""

A centered box is a rectangular neighborhood which is defined by the offsets of
the last element of the neighborhood with respect to the center of the box.

"""
struct CenteredBox{N} <: Neighborhood{N}
    last::CartesianIndex{N}
end

"""

A rectangular (cartesian) box is defined by the bounds of the neighborhood
with respect to the center of the box.

"""
struct CartesianBox{N} <: Neighborhood{N}
    # We have to define this type even though it is nothing more than a
    # CartesianRange because of the heritage.
    bounds::CartesianRange{CartesianIndex{N}}
end

"""

A `Kernel` can be used to define a versatile type of structuring element.  It
is a rectangular array of coefficients over a, possibly off-centered,
rectangular neighborhood.

"""
struct Kernel{T,N} <: Neighborhood{N}
    coefs::Array{T,N}
    anchor::CartesianIndex{N}
end

const RectangularBox{N} = Union{CenteredBox{N}, CartesianBox{N}}
