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
abstract type Neighborhood{N} end

"""

A centered box is a rectangular neighborhood which is defined by the offsets of
the last element of the neighborhood with respect to the center of the box.

"""
struct CenteredBox{N} <: Neighborhood{N}
    # For a CenteredBox, the most used operation is getting its limits, so
    # this is what we store.  The inner constructor enforces the constraints.
    start::CartesianIndex{N}
    stop::CartesianIndex{N}
    CenteredBox{N}(stop::CartesianIndex{N}) where {N} = new{N}(-stop, stop)
end

"""

```julia
CartesianBox(start, stop)
```

yields a rectangular (cartesian) box defined by the bounds of the
multi-dimensional indices in the box.

Another possibility is to specify the dimensions of the box and the offsets of
its central element:

```julia
CartesianBox(dims, offs)
```

with `dims` a `N`-tuple of dimensions and `offs` either a `N`-tuple of indices
of an instance of `CartesianIndex{N}`.

A `CartesianBox` can also be defined by the index ranges along all the
dimensions.  For example:

```julia
CartesianBox(-3:3, 0, -2:1)
CartesianBox((-3:3, 0, -2:1))

```

both yield a 3-dimensional `CartesianBox` of size `7×1×4` and whose first index
varies on `-3:3`, its second index is `0` while its third index varies on
`-2:1`.

Finally, a `CartesianBox` can be defined as:

```julia
CartesianBox(R)
```

where `R` is an instance of `CartesianIndices` or, on version of Julia ≤ 0.6,
`CartesianRange`.

"""
struct CartesianBox{N} <: Neighborhood{N}
    # For a CartesianBox, the most used operation is getting its limits, so
    # this is what we store.
    start::CartesianIndex{N}
    stop::CartesianIndex{N}
end

@static if length(Compat.InteractiveUtils.subtypes(CartesianIndex)) != 0
    error("CartesianIndex is assumed to be a concrete type")
end

"""

A `Kernel` can be used to define a versatile type of structuring element.  It
is a rectangular array of coefficients over a, possibly off-centered,
rectangular neighborhood.  In general, it is sufficient to specify
`::Kernel{T,N}` in the signature of methods, with `T` the type of the
coefficients and `N` the number of dimensions (the third parameter `A` of the
type is to fully qualify the type of the array of coefficients).

A kernel is built as:

```julia
B = Kernel([T,] C, start=defaultstart(C))
```

where `C` is the array of coefficients (which can be retrieved by `coefs(B)`)
and `start` the initial `CartesianIndex` for indexing the kernel (which can be
retrieved by `initialindex(B)`).  The `start` parameter let the caller choose
an arbitrary origin for the kernel coefficients; when a filter is applied, the
following mapping is assumed:

```julia
B[k] ≡ C[k + off]
```

where `off = initialindex(C) - initialindex(B)`.

If `start` is omitted, its value is set so that the *origin* (whose index is
`zero(CartesianIndex{N})` with `N` the number of dimensions) of the kernel
indices is at the geometric center of the array of coefficients (see
[`LocalFilters.defaultstart`](@ref)).  Optional argument `T` is to impose the
type of the coefficients.

To convert the element type of the coefficients of an existing kernel, do:

```julia
Kernel(T, K)
```

which yields a kernel whose coefficients are those of the kernel `K`
converted to type `T`.

It is also possible to convert neighborhoods such as [`CenteredBox`](@ref) and
[`CartesianBox`](@ref) into a kernel with boolean coefficients by calling:

```julia
Kernel(B)
```

where `B` is the neighborhood to convert into an instance of `Kernel`.

"""
struct Kernel{T,N,A<:AbstractArray{T,N}} <: Neighborhood{N}
    coefs::A
    offset::CartesianIndex{N}
    start::CartesianIndex{N}
    stop::CartesianIndex{N}
    function Kernel{T,N,A}(C::A, start::CartesianIndex{N}
                           ) where {T,N,A<:AbstractArray{T,N}}
        # In local filtering operations, the kernel `B` will be used as
        #
        #     A[i-k]⋄B[k]    (∀ k ∈ [kmin, kmax])
        #
        # with `A` the source, `i` the destination index, `⋄` a given binary
        # operation and `k` (here, indices and offsets are multi-dimensional).
        # The offset between the indices in the kernel `B` and those in the
        # array of coefficients `C` is such that:
        #
        #     kmin + off = jmin = initialindex(C)
        #     kmax + off = jmax = finalindex(C)
        #
        # Hence:
        #
        #     kmin = start
        #     off = jmin - kmin
        #     kmax = jmax - off
        #
        # gives the [limits](@ref) and [offset](@ref) for index `k` in the
        # kernel `B`.
        @assert eltype(C) === T
        @assert ndims(C) == N
        jmin, jmax = limits(C)
        off = jmin - start
        return new{T,N,A}(C, off, start, jmax - off)
    end
end

"""
A `RectangularBox{N}` is an union of `CenteredBox{N}` and `CartesianBox{N}`.
"""
const RectangularBox{N} = Union{CenteredBox{N}, CartesianBox{N}}

"""

`IndexInterval` is an union of the types of any argument suitable to specify an
interval of indices along a dimension (that is, an integer or an integer valued
unit range).

"""
const IndexInterval = Union{Integer,AbstractUnitRange{<:Integer}}

"""

`CartesianRegion{N}` is an union of the types of anything suitable to define a
Cartesian region in `N` dimensions.  That is, an interval of Cartesian indices
in `N` dimensions.  Methods [`initialindex`](@ref), [`finalindex`](@ref),
[`limits`](@ref) and [`cartesianregion`](@ref) can be applied to anything
whose type belongs to `CartesianRegion`.

"""
CartesianRegion
@static if isdefined(Base, :CartesianIndices)
    const CartesianRegion{N} = Union{NTuple{N,IndexInterval},
                                     Neighborhood{N},
                                     CartesianIndices{N}}
else
    const CartesianRegion{N} = Union{NTuple{N,IndexInterval},
                                     Neighborhood{N},
                                     CartesianIndices{N},
                                     CartesianRange{CartesianIndex{N}}}
end
