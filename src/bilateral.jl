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

function bilateralfilter(::Type{T}, A::AbstractArray, args...) where {T}
    return bilateralfilter!(Array{T}(undef, size(A)), A, args...)
end

"""
```julia
bilateralfilter!([T,] dst, A, Fr, Gs, ...)
```

stores in `dst` the result of applying the bilateral filter on array `A`.

Argument `Fr` specifies the range kernel for smoothing differences in
intensities, it is a function which takes two values from `A` as arguments and
returns a nonnegative value.

Arguments `Gs, ...` specify the spatial kernel for smoothing differences in
coordinates.

Optional argument `T` can be used to force the floating-point type used for
(most) computations.

See [wikipedia](https://en.wikipedia.org/wiki/Bilateral_filter).

"""
function bilateralfilter!(::Type{T},
                          dst::AbstractArray{Td,N},
                          A::AbstractArray{Ta,N},
                          Fr::Function,
                          Gs::Kernel{Tg,N}
                          ) where {T<:AbstractFloat, Tg<:Real, Td, Ta, N}
    return bilateralfilter!(T, dst, A, Fr, T(Gs))
end

function bilateralfilter!(::Type{T},
                          dst::AbstractArray{Td,N},
                          A::AbstractArray{Ta,N},
                          Fr::Function,
                          Gs::Kernel{T,N}) where {T<:AbstractFloat, Td, Ta, N}
    # The state is the tuple: (central_value, numerator, denominator).
    return localfilter!(dst, A, Gs,
                        (val) -> (val, zero(T), zero(T)),
                        (v, val, ker) -> _update(v, val, ker, Fr(val, v[1])),
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
    @assert width > 0 && isodd(width) "width or region of interest must be at least and odd"
    B = CenteredBox{N}(CartesianIndex(ntuple(i -> Int(width), N)))
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
    return bilateralfilter!(dst, A,
                            (v, v0) -> _gausswindow(qr, v - v0), args...)
end

function bilateralfilter!(dst::AbstractArray{Td,N},
                          A::AbstractArray{Ta,N},
                          σr::Real,
                          args...) where {Td<:Real, Ta<:Real, N}
    return bilateralfilter!(Float64, dst, A, σr, args...)
end

# Distance filter specifed by its standard deviation and the size of the ROI.
function bilateralfilter!(dst::AbstractArray{Td,N},
                          A::AbstractArray{Ta,N},
                          Fr::Function,
                          σs::Real, B) where {Td<:Real, Ta<:Real, N}
    return bilateralfilter!(Float64, dst, A, Fr, σs, B)
end

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
    w = convert(T, wr)*ws
    return (v[1], v[2] + convert(T, val)*w, v[3] + w)
end

_store!(dst::AbstractArray{T,N}, i, val) where {T<:AbstractFloat,N} =
    dst[i] = val

_store!(dst::AbstractArray{T,N}, i, val) where {T<:Integer,N} =
    dst[i] = round(T, val)

_final!(dst, i, v::Tuple{V,T,T}) where {T<:AbstractFloat,V} =
    _store!(dst, i, (v[3] > zero(T) ? v[2]/v[3] : zero(T)))
