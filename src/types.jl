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
    CartesianUnitRange{N}

is an alias for a `N`-dimensional Cartesian index range with unit step.  Since
Julia 1.6, non-unit step Cartesian ranges may be defined.

"""
const CartesianUnitRange{N} = CartesianIndices{
    N,<:NTuple{N,AbstractUnitRange{Int}}}

"""
    LocalFilters.Indices(A...) -> indices

yields a callable object that can be used to produce ranges of indices for each
of the arrays `A...`.  These ranges will all be of the same type: linear index
ranges, if all arrays `A...` are vectors implementing fast linear indexing,
Cartesian index ranges otherwise.

The returned object is similar to the `eachindex` method but specialized for a
style of indexing, it can be used as `indices(B...)` with `B...` any number of
the arrays in `A...` to yield a suitable index range to access all the entries
of array(s) `B...`.  If `B...` consists in several arrays, they must have the
same indices.

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
                        CartesianUnitRange{N}}

"""
    LocalFilters.Box{N}(args...)

yields an abstract array whose elements are all `true` and whose axes are
defined by `args...`.  This kind of object is used to represent
hyperrectangular sliding windows in `LocalFilters`.  Type parameter `N` is the
number of dimensions, it may be omitted if it can be deduced from the
arguments.

"""
struct Box{N,R<:CartesianUnitRange{N}} <: AbstractArray{Bool,N}
    inds::R
    function Box(inds::R) where {N,R<:CartesianIndices{N}}
        R <: CartesianUnitRange{N} && return new{N,R}(inds)
        unit_inds = CartesianIndices(map(unit_range, ranges(inds)))
        return new{N,typeof(unit_inds)}(unit_inds)
    end
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

"""
    LocalFilters.BoundaryConditions

is the super-type of types representing boundary conditions.

"""
abstract type BoundaryConditions end

"""
    LocalFilters.FlatBoundaries(inds)

yields an object representing *flat* boundary conditions for arrays
with index range `inds`.

"""
struct FlatBoundaries{R<:Union{AbstractUnitRange{Int},
                               CartesianUnitRange}} <: BoundaryConditions
    indices::R
    # Inner constructor to refuse to build object is range is empty.
    function FlatBoundaries(indices::R) where {R<:Union{AbstractUnitRange{Int},
                                                        CartesianUnitRange}}
        isempty(indices) && throw(ArgumentError("empty index range"))
        return new{R}(indices)
    end
end
