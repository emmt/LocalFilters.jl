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
    f = LocalFilters.ConstantProducer(val)

yields a callable object `f` such that `f(args...; kwds...) === val` always
holds.  This is similar to `Returns` which appears in Julia 1.7.

This is useful to avoid closures which are quite inefficient and thus have a
strong impact on performances when called repeatedly in loops.  Typically when
calling [`localfilter!`](@ref) with an initializer whose value is a constant
but given by an anonymous function or a closure depending on a local variable,
the code:

    v0 = ... # initial state variable
    localfilter!(dst, A, B, ConstantProducer(v0), update, store!)

is much faster than:

    v0 = ... # initial state variable
    localfilter!(dst, A, B, a -> v0, update, store!)

""" ConstantProducer

struct ConstantProducer{T} <: Function
    val::T
end

(obj::ConstantProducer)(args...; kwds...) = getfield(obj, :val)
