#
# types.jl --
#
# Type definitions for local filters.
#
#-----------------------------------------------------------------------------------------
#
# This file is part of the `LocalFilters.jl` package licensed under the MIT "Expat"
# License.
#
# Copyright (C) 2017-2025, Éric Thiébaut.
#

"""
    LocalFilters.Axis

is the union of types suitable to specify a neighborhood axis. It may be an integer
(assumed to be the length of the axis) or an integer-valued range. This is also the union
of types accepted by the [`LocalFilters.centered_range`](@ref) method.

"""
const Axis = Union{Integer,AbstractRange{<:Integer}}

"""
    LocalFilters.Indices(A...) -> indices

yields a callable object that can be used to produce ranges of indices for each of the
arrays `A...`. These ranges will all be of the same type: linear index ranges, if all
arrays `A...` are vectors implementing fast linear indexing, Cartesian index ranges
otherwise.

The returned object is similar to the `eachindex` method but specialized for a style of
indexing, it can be used as `indices(B...)` with `B...` any number of the arrays in `A...`
to yield a suitable index range to access all the entries of array(s) `B...`. If `B...`
consists in several arrays, they must have the same indices.

"""
struct Indices{S<:IndexStyle} <: Function end

# FIXME Remove this constant.
const CartesianUnitRange{N} = CartesianIndices{N,<:NTuple{N,AbstractUnitRange{Int}}}

"""
    LocalFilters.Window{N}

is the union of types of arguments which are not a kernel but which may define a simple
`N`-dimensional hyper-rectangular sliding window and that can be converted into a kernel
by the [`LocalFilters.kernel`](@ref) method.

"""
const Window{N} = Union{Axis,NTuple{N,Axis},NTuple{2,CartesianIndex{N}},CartesianIndices{N}}

"""
    const LocalFilters.Box{N} = FastUniformArray{Bool,N,true}

is an alias to an abstract `N`-dimensional array whose elements are all `true`. Instances
of this kind are used to represent hyper-rectangular sliding windows in `LocalFilters`.

"""
const Box{N} = FastUniformArray{Bool,N,true}

"""
    LocalFilters.FilterOrdering

is the super-type of the possible ordering of indices in local filtering operations.

"""
abstract type FilterOrdering end
@public FilterOrdering

"""
    LocalFilters.ForwardFilterOrdering

is the singleton type of *forward* ordering of indices in local filter operations. The
singleton instance [`ForwardFilter`](@ref) of this type is exported by `LocalFilters`.

"""
struct ForwardFilterOrdering <: FilterOrdering end
@public ForwardFilterOrdering

"""
    LocalFilters.ForwardFilterOrdering

is the singleton type of *reverse* ordering of indices in local filter operations. The
singleton instance [`ReverseFilter`](@ref) of this type is exported by `LocalFilters`.

"""
struct ReverseFilterOrdering <: FilterOrdering end
@public ReverseFilterOrdering

"""
    ForwardFilter

is an exported constant object used to indicate *forward* ordering of indices in local
filter operations. It can be called as:

    ForwardFilter(i, j) -> j - i

to yield the index in the filter kernel. See also [`ReverseFilter`](@ref) for *reverse*
ordering and [`LocalFilters.localindices`](@ref) for building a range of valid indices
`j`.

"""
const ForwardFilter = ForwardFilterOrdering()

"""
    ReverseFilter

is an exported constant object used to indicate *reverse* ordering of indices in local
filter operations. It can be called as:

    ReverseFilter(i, j) -> i - j

to yield the index in the filter kernel. See also [`ForwardFilter`](@ref) for *forward*
ordering and [`LocalFilters.localindices`](@ref) for building a range of valid indices
`j`.

"""
const ReverseFilter = ReverseFilterOrdering()

"""
    LocalFilters.BoundaryConditions

is the super-type of types representing boundary conditions.

"""
abstract type BoundaryConditions end
@public BoundaryConditions

"""
    LocalFilters.FlatBoundaries(inds)

yields an object representing *flat* boundary conditions for arrays with index range
`inds`.

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
@public FlatBoundaries
