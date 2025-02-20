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
    LocalFilters.LocalAxis

is the union of types suitable to specify a neighborhood axis. It may be an integer
(assumed to be the length of the axis) or an integer-valued range. This is also the union
of types accepted by the [`LocalFilters.neighborhood_range`](@ref) method.

"""
const LocalAxis = Union{Integer,AbstractRange{<:Integer}}

const LocalAxes{N} = NTuple{N,LocalAxis}

const Axes{N} = NTuple{N,AbstractUnitRange{Int}}

"""
    LocalFilters.CartesianUnitRange{N}

is an alias for a `N`-dimensional Cartesian index range with unit step. Since Julia 1.6,
non-unit step Cartesian ranges may be defined.

"""
const CartesianUnitRange{N} = CartesianIndices{N,<:NTuple{N,AbstractUnitRange{Int}}}

"""
    LocalFilters.Indices(A...) -> indices

yields a callable object that can be used to produce ranges of indices for each of the
arrays `A...`. These ranges will all be of the same type: linear index ranges, if all
arrays `A...` are vectors implementing fast linear indexing, Cartesian index ranges
otherwise.

The returned object is similar to the `eachindex` method but specialized for a style of
indexing, it can be used as `indices(B...)` to yield a suitable index range to access all
the entries of array(s) `B...` which are any number of the `A...` specified when building
the `indices` object. If `B...` consists in several arrays, they must have the same axes.

Call:

    LocalFilters.Indices{S}()

with `S = IndexLinear` or `S = IndexCartesian` to specifically choose the
indexing style.

"""
struct Indices{S<:IndexStyle} <: Function end

"""
    LocalFilters.Window{N}

is the union of types of arguments suitable to define a simple `N`-dimensional
hyper-rectangular sliding window and that can be converted into a kernel by the
[`LocalFilters.kernel`](@ref) method or by the [`LocalFilters.Box`](@ref) constructor.

"""
const Window{N} = Union{LocalAxis,NTuple{N,LocalAxis},NTuple{2,CartesianIndex{N}},
                        CartesianUnitRange{N}}

"""
    LocalFilters.Box{N,I}

is an alias to the type of `N`-dimensional arrays whose elements are all `true`. Parameter
`I` is the type of the tuple storing the size or axes of the array. This kind of arrays is
used to represent hyper-rectangular sliding windows in `LocalFilters`.

Method [`LocalFilters.box`](@ref) can be used to build such an object with of this type
.

"""
const Box{N,I} = FastUniformArray{Bool,N,true,I}

"""
    LocalFilters.FilterOrdering

is the super-type of the possible ordering of indices in local filtering operations.
[`FORWARD_FILTER`](@ref) and [`REVERSE_FILTER`](@ref) are the two concrete singletons that
inherit from this type.

"""
abstract type FilterOrdering end

"""
    LocalFilters.ForwardFilterOrdering

is the singleton type of *forward* ordering of indices in local filter operations. The
singleton instance [`FORWARD_FILTER`](@ref) of this type is exported by `LocalFilters`.

"""
struct ForwardFilterOrdering <: FilterOrdering end

"""
    LocalFilters.ForwardFilterOrdering

is the singleton type of *reverse* ordering of indices in local filter operations. The
singleton instance [`REVERSE_FILTER`](@ref) of this type is exported by `LocalFilters`.

"""
struct ReverseFilterOrdering <: FilterOrdering end

"""
    LocalFilters.BoundaryConditions

is the super-type of types representing boundary conditions.

"""
abstract type BoundaryConditions end

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

"""
    f = LocalFilters.Yields(value)
    f = LocalFilters.Yields{V}(value)

build a callable object `f` such that `f(args...; kwds...)` yields `value` whatever the
arguments `args...` and the keywords `kwds...`. If type `V` is supplied, `value` is
converted to that type.

"""
struct Yields{V} <: Function
    value::V
    Yields{V}(value) where {V} = new{V}(value)
    Yields(value::V) where {V} = new{V}(value)
end
(obj::Yields)(args...; kwds...) = getfield(obj, 1)
function Base.show(io::IO, obj::Yields)
    show(io, typeof(obj))
    print(io, "(")
    show(io, obj())
    print(io, ")")
end
