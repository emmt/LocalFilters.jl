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
# Copyright (c) 2017-2025, Éric Thiébaut.
#

"""
    LocalFilters.Axis

is the union of types suitable to specify a neighborhood axis. It may be an integer
(assumed to be the length of the axis) or an integer-valued range. This is also the union
of types accepted by the [`LocalFilters.kernel_range`](@ref) method.

"""
const Axis = Union{Integer,AbstractRange{<:Integer}}

# Type of the result of `axes(A)`.
const ArrayAxes{N} = NTuple{N,AbstractUnitRange{<:Integer}}

struct Indices{S<:IndexStyle} <: Function end

"""
    LocalFilters.Window{N}

is the union of types of arguments suitable to define a simple `N`-dimensional
hyper-rectangular sliding window and that can be converted into a kernel by the
[`LocalFilters.kernel`](@ref) method.

"""
const Window{N} = Union{Axis,NTuple{N,Axis},NTuple{2,CartesianIndex{N}},
                        CartesianIndices{N}}

"""
    const LocalFilters.Box{N} = FastUniformArray{Bool,N,true}

is an alias to the type of `N`-dimensional arrays whose elements are all `true`. Instances
of this kind are used to represent hyper-rectangular sliding windows in `LocalFilters`.

Method [`LocalFilters.box`](@ref) can be used to build an object of this type.

"""
const Box{N} = FastUniformArray{Bool,N,true}
@public Box

"""
    LocalFilters.FilterOrdering

is the super-type of the possible ordering of indices in local filtering operations.
[`FORWARD_FILTER`](@ref) and [`REVERSE_FILTER`](@ref) are the two concrete singletons that
inherit from this type.

"""
abstract type FilterOrdering end
@public FilterOrdering

"""
    LocalFilters.ForwardFilterOrdering

is the singleton type of *forward* ordering of indices in local filter operations. The
singleton instance [`FORWARD_FILTER`](@ref) of this type is exported by `LocalFilters`.

"""
struct ForwardFilterOrdering <: FilterOrdering end
@public ForwardFilterOrdering

"""
    LocalFilters.ForwardFilterOrdering

is the singleton type of *reverse* ordering of indices in local filter operations. The
singleton instance [`REVERSE_FILTER`](@ref) of this type is exported by `LocalFilters`.

"""
struct ReverseFilterOrdering <: FilterOrdering end
@public ReverseFilterOrdering

"""
    FORWARD_FILTER

is an exported constant object used to indicate *forward* ordering of indices in local
filter operations. It can be called as:

    FORWARD_FILTER(i, j) -> j - i

to yield the index in the filter kernel. See also [`REVERSE_FILTER`](@ref) for *reverse*
ordering and [`LocalFilters.localindices`](@ref) for building a range of valid indices
`j`.

"""
const FORWARD_FILTER = ForwardFilterOrdering()

"""
    REVERSE_FILTER

is an exported constant object used to indicate *reverse* ordering of indices in local
filter operations. It can be called as:

    REVERSE_FILTER(i, j) -> i - j

to yield the index in the filter kernel. See also [`FORWARD_FILTER`](@ref) for *forward*
ordering and [`LocalFilters.localindices`](@ref) for building a range of valid indices
`j`.

"""
const REVERSE_FILTER = ReverseFilterOrdering()

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
                               CartesianIndices}} <: BoundaryConditions
    indices::R
    # Inner constructor to refuse to build object is range is empty.
    function FlatBoundaries(indices::R) where {R<:Union{AbstractUnitRange{Int},
                                                        CartesianIndices}}
        isempty(indices) && throw(ArgumentError("empty index range"))
        return new{R}(indices)
    end
end
@public FlatBoundaries
