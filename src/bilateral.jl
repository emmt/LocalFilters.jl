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
# Copyright (C) 2018, Éric Thiébaut.
#

module BilateralFilter

export bilateralfilter!, bilateralfilter!

using ..LocalFilters
using ..LocalFilters: Neighborhood, RectangularBox, Kernel, axes, _store!

bilateralfilter(A::AbstractArray{T}, args...) where {T<:Real} =
    # Provide type for computations and result.
    bilateralfilter(float(T), A, args...)

function bilateralfilter(::Type{T}, A::AbstractArray,
                         args...) where {T<:AbstractFloat}
    return bilateralfilter!(T, similar(Array{T}, axes(A)), A, args...)
end

"""
```julia
bilateralfilter([T,] A, Fr, Gs, ...)
```

yields the result of applying the bilateral filter on array `A`.

Argument `Fr` specifies the range kernel for smoothing differences in
intensities, it is a function which takes two values from `A` as arguments and
returns a nonnegative value.

Arguments `Gs, ...` specify the spatial kernel for smoothing differences in
coordinates.

Optional argument `T` can be used to force the floating-point type used for
(most) computations.

The in-place version is:

```julia
bilateralfilter!([T,] dst, A, Fr, Gs, ...)
```

which stores in `dst` the result of applying the bilateral filter on array `A`.

See [wikipedia](https://en.wikipedia.org/wiki/Bilateral_filter).

"""
function bilateralfilter!(dst::AbstractArray{Td,N},
                          A::AbstractArray{Ta,N},
                          args...) where {Td<:Real, Ta, N}
    # Provide type for computations.
    return bilateralfilter!(float(Td), dst, A, args...)
end

@doc @doc(bilateralfilter!) bilateralfilter

function bilateralfilter!(::Type{T},
                          dst::AbstractArray{Td,N},
                          A::AbstractArray{Ta,N},
                          Fr::Function,
                          Gs::Kernel{Tg,N}) where {T<:AbstractFloat, Td, Ta,
                                                   Tg<:Real, N}
    # The state is the tuple: (central_value, numerator, denominator).
    return localfilter!(dst, A, Kernel{T}(Gs),
                        (val) -> (val, zero(T), zero(T)),
                        (v, val, ker) -> _update(v, val, ker,
                                                 convert(T, Fr(val, v[1]))),
                        _final!)
end

function bilateralfilter!(::Type{T},
                          dst::AbstractArray{Td,N},
                          A::AbstractArray{Ta,N},
                          Fr::Function,
                          Gs::Function,
                          B::RectangularBox{N}
                          ) where {T<:AbstractFloat, Td, Ta, N}
    return bilateralfilter!(T, dst, A, Fr, Kernel(T, Gs, B))
end

function bilateralfilter!(::Type{T},
                          dst::AbstractArray{Td,N},
                          A::AbstractArray{Ta,N},
                          Fr::Function,
                          Gs::Function,
                          width::Integer
                          ) where {T<:AbstractFloat, Td<:Real, Ta<:Real, N}
    @assert width > 0 && isodd(width) "width or region of interest must be at least one and odd"
    h = Int(width) >> 1
    I = CartesianIndex(ntuple(i -> h, N))
    B = RectangularBox{N}(-I, I)
    return bilateralfilter!(T, dst, A, Fr, Gs, B)
end

# Range filter specifed by its standard deviation.
function bilateralfilter!(::Type{T},
                          dst::AbstractArray{Td,N},
                          A::AbstractArray{Ta,N},
                          σr::Real,
                          args...) where {T<:AbstractFloat,
                                          Td<:Real, Ta<:Real, N}
    @assert isfinite(σr) && σr > 0
    qr = _gaussfactor(T, σr)
    return bilateralfilter!(T, dst, A,
                            (v, v0) -> _gausswindow(qr, v - v0), args...)
end

# Distance filter specifed by its standard deviation and the size of the ROI.
function bilateralfilter!(::Type{T},
                          dst::AbstractArray{Td,N},
                          A::AbstractArray{Ta,N},
                          Fr::Function,
                          σs::Real, B) where {T<:AbstractFloat,
                                              Td<:Real, Ta<:Real, N}
    @assert isfinite(σs) && σs > 0
    qs = _gaussfactor(T, σs)
    return bilateralfilter!(T, dst, A, Fr, (i) -> _gausswindow(qs, i), B)
end

_gaussfactor(::Type{T}, σ::Real) where {T<:AbstractFloat} =
    -1/(2*convert(T, σ)^2)

# η = -1/2σ²
_gausswindow(η::T, I::CartesianIndex{N}) where {T<:AbstractFloat,N} =
    exp(sum((k) -> I[k]^2, 1:N)*η)

_gausswindow(η::T, x::Real, x0::Real) where {T<:AbstractFloat} =
    _gausswindow(η, x - x0)

_gausswindow(η::T, x::Real) where {T<:AbstractFloat} =
    exp(convert(T, x*x)*η)

function _update(v::Tuple{V,T,T}, val::V,
                 ws::T, wr::T) where {V, T<:AbstractFloat}
    w = wr*ws
    return (v[1], v[2] + convert(T, val)*w, v[3] + w)
end

_final!(dst, i, v::Tuple{V,T,T}) where {T<:AbstractFloat,V} =
    _store!(dst, i, (v[3] > zero(T) ? v[2]/v[3] : zero(T)))

end # module
