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
# Copyright (C) 2017-2022, Éric Thiébaut.
#

"""
    LocalFilters.IntegerRange

is the abstract type of integer-valued ranges.

"""
const IntegerRange = OrdinalRange{<:Integer,<:Integer}

"""
    LocalFilters.Axis

is the union of types suitable to specify a neighborhood axis.  It may be an
integer (assumed to be the length of the axis) or an integer-valued range.
This is also the union of types accepted by the
[`LocalFilters.neighborhood_range`](@ref) method.

"""
const Axis = Union{Integer,IntegerRange}

"""
    LocalFilters.Indices(A...)

yields a callable object that can be used to produce ranges of indices for each
of the arrays `A...`.  These ranges are all of the same type: linear index
ranges, if all arrays `A...` are vectors implementing fast linear indexing,
Cartesian index ranges otherwise.

"""
struct Indices{S<:IndexStyle} <: Function end

"""
    LocalFilters.Window{N}

is the union of types of arguments which are not a kernel but which may define
a simple `N`-dimensional hyperrectangular sliding window and that can be
converted into a kernel by the [`LocalFilters.kernel`](@ref) method or by the
[`LocalFilters.Box`](@ref) constructor.

"""
const Window{N} = Union{Axis,NTuple{N,Axis},NTuple{2,CartesianIndex{N}},
                        CartesianIndices{N}}

"""
    LocalFilters.Box{N}(args...)

yields an abstract array whose elements are all `true` and whose axes are
defined by `args...`.  This kind of object is used to represent
hyperrectangular sliding windows in `LocalFilters`.  Type parameter `N` is the
number of dimensions, it may be omitted if it can be deduced from the
arguments.

"""
struct Box{N,R<:CartesianIndices{N}} <: AbstractArray{Bool,N}
    inds::R
end

"""
    LocalFilters.FilterOrdering

is the super-type of the possible ordering of indices in local filtering
operations.

"""
abstract type FilterOrdering end

"""
    LocalFilters.ForwardFilterOrdering

is the singleton type of *forward* ordering of indices in local filter
operations.  The singleton instance [`ForwardFilter`](@ref) of this type is
exported by `LocalFilters`.

"""
struct ForwardFilterOrdering <: FilterOrdering end

"""
    LocalFilters.ForwardFilterOrdering

is the singleton type of *reverse* ordering of indices in local filter
operations.  The singleton instance [`ReverseFilter`](@ref) of this type is
exported by `LocalFilters`.

"""
struct ReverseFilterOrdering <: FilterOrdering end
