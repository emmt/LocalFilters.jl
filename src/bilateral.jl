#
# bilateral.jl --
#
# Implements the bilateral filter.
#
#------------------------------------------------------------------------------
#
# This file is part of the `LocalFilters.jl` package licensed under the MIT
# "Expat" License.
#
# Copyright (C) 2018-2022, Éric Thiébaut.
#

module BilateralFilter

export bilateralfilter!, bilateralfilter

using ..LocalFilters
using ..LocalFilters: Neighborhood, RectangularBox, Kernel, axes, store!

using Base: @propagate_inbounds

"""
    GaussianWindow{T}(σ) -> f

yields a functor `f` with the shape of a Gaussian of standard deviation `σ` but
with a peak value of one, i.e. `f(0) -> 1`.  The functor `f` can be applied to
a single value `x`, to 2 values `x` and `x0` to yield `f(x - x0)`, or to a
coordinate difference expressed as a Cartesian index.

Type parameter `T` is to specify the numerical type of the parameter of the
functor.  It can be omitted if `σ` is real.

"""
struct GaussianWindow{T} <: Function
    η::T # η = -1/2σ² is the factor in the exponential
    GaussianWindow{T}(σ) where {T} = new{T}(-1/(2*σ^2))
end
GaussianWindow(σ::Real) = GaussianWindow{float(typeof(σ))}(σ)

(f::GaussianWindow)(I::CartesianIndex{N}) where {N} =
    exp(sum((k) -> I[k]^2, 1:N)*f.η) # FIXME: use @generated

(f::GaussianWindow)(x::T, x0::T) where {T} = f(x - x0)
(f::GaussianWindow{T})(x) where {T} = exp(convert(T,x)^2*f.η)


"""
    bilateralfilter([T,] A, F, G, ...)

yields the result of applying the bilateral filter on array `A`.

Argument `F` specifies how to smooth the differences in values.  It may be
function which takes two values from `A` as arguments and returns a nonnegative
weight.  It may be a real which is assumed to be the standard deviation of a
Gaussian.

Arguments `G, ...` specify the settings of the distance filter for smoothing
differences in coordinates.  There are several possibilities:

- `G, ...` can be a [`LocalFilters.Kernel`](@ref) instance (specified as a
  single argument).

- Argument `G` may be a function taking as argument the Cartesian index of the
  coordinate differences and returning a nonnegative weight.  Argument `G` may
  also be a real specifying the standard deviation of the Gaussian used to
  compute weights.  Subsequent arguments `...` are to specify the neighborhood
  where to apply the distance filter function, they can be a
  [`Neighborhood`](@ref) object such as a [`RectangularBox`](@ref) or anything
  that may defined a neighborhood such as an odd integer assumed to be the
  width of the neighborhood along every dimensions of `A`.

Optional argument `T` can be used to force the element type used for (most)
computations.  This argument is needed if the element type of `A` is not a
real.

See [`bilateralfilter!`](@ref) for an in-place version of this function.

See [wikipedia](https://en.wikipedia.org/wiki/Bilateral_filter) for a
description of the bilateral filter.

"""
bilateralfilter(A::AbstractArray{<:Real}, args...) =
    # Provide type for computations and result.
    bilateralfilter(float(eltype(A)), A, args...)

bilateralfilter(T::Type, A::AbstractArray, args...) =
    bilateralfilter!(T, similar(A, T), A, args...)

"""
    bilateralfilter!([T,] dst, A, F, G, ...) -> dst

overwrites `dst` with the result of applying the bilateral filter on array `A`
and returns `dst`.

See [`bilateralfilter`](@ref) for a description of the other arguments than
`dst`.

See [wikipedia](https://en.wikipedia.org/wiki/Bilateral_filter) for a
description of the bilateral filter.

"""
function bilateralfilter!(dst::AbstractArray{<:Real,N},
                          A::AbstractArray{<:Any,N},
                          args...) where {N}
    # Provide type for computations.
    return bilateralfilter!(float(eltype(dst)), dst, A, args...)
end

function bilateralfilter!(::Type{T},
                          dst::AbstractArray{<:Any,N},
                          A::AbstractArray{<:Any,N},
                          F::Function,
                          G::Kernel{<:Any,N}) where {T,N}
    # The state is the tuple: (central_value, numerator, denominator).
    return localfilter!(dst, A, Kernel{T}(G),
                        (val) -> (val, zero(T), zero(T)),
                        (v, val, ker) -> update(v, val, ker,
                                                convert(T, F(val, v[1]))),
                        final!)
end

function bilateralfilter!(::Type{T},
                          dst::AbstractArray{<:Any,N},
                          A::AbstractArray{<:Any,N},
                          F::Function,
                          G::Function,
                          B::RectangularBox{N}
                          ) where {T,N}
    return bilateralfilter!(T, dst, A, F, Kernel{T}(G, B))
end

function bilateralfilter!(::Type{T},
                          dst::AbstractArray{<:Any,N},
                          A::AbstractArray{<:Any,N},
                          F::Function,
                          G::Function,
                          width::Integer
                          ) where {T,N}
    (width > 0 && isodd(width)) || throw(ArgumentError(
        "width of neighborhood must be at least one and odd"))
    h = Int(width) >> 1 # half-width
    I = CartesianIndex(ntuple(i -> h, Val(N)))
    B = RectangularBox{N}(-I, I)
    return bilateralfilter!(T, dst, A, F, G, B)
end

# Range filter specifed by its standard deviation.
function bilateralfilter!(::Type{T},
                          dst::AbstractArray{<:Any,N},
                          A::AbstractArray{<:Any,N},
                          σr::Real,
                          args...) where {T,N}
    (isfinite(σr) && σr > 0) || throw(ArgumentError(
        "standard deviation must be finite and positive"))
    F = GaussianWindow{T}(σr)
    return bilateralfilter!(T, dst, A, F, args...)
end

# Distance filter specifed by its standard deviation and the neighborhood.
function bilateralfilter!(::Type{T},
                          dst::AbstractArray{<:Any,N},
                          A::AbstractArray{<:Any,N},
                          F::Function,
                          σs::Real, B) where {T,N}
    (isfinite(σs) && σs > 0) || throw(ArgumentError(
        "standard deviation must be finite and positive"))
    G = GaussianWindow{T}(σs)
    return bilateralfilter!(T, dst, A, F, G, B)
end

function update(v::Tuple{V,T,T}, val::V, ws::T, wr::T) where {V,T}
    w = wr*ws
    return (v[1], v[2] + convert(T, val)*w, v[3] + w)
end

@inline @propagate_inbounds final!(dst, i, v::Tuple{V,T,T}) where {T,V} =
    store!(dst, i, (v[3] > zero(T) ? v[2]/v[3] : zero(T)))

end # module
