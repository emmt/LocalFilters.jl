#
# NaiveLocalFilters.jl -
#
# Naive (non-optimized) implementation of local filters. This module provides reference
# implementations of methods for checking results and benchmarking.
#
#------------------------------------------------------------------------------
#
# This file is part of the `LocalFilters.jl` package licensed under the MIT "Expat"
# License.
#
# Copyright (c) 2017-2025, Éric Thiébaut.
#

module NaiveLocalFilters

using LocalFilters
using LocalFilters:
    Kernel,
    Neighborhood,
    RectangularBox,
    store!,
    type_of_sum,
    axes,
    cartesian_region,
    coefs,
    limits,
    offset

import LocalFilters:
    bottom_hat,
    cartesian_region,
    closing,
    convolve!,
    convolve,
    dilate!,
    dilate,
    erode!,
    erode,
    localextrema!,
    localextrema,
    localfilter!,
    localmean,
    localmean!,
    opening,
    top_hat

localmean(variant::Val, A::AbstractArray{T,N}, B=3) where {T,N} =
    localmean(variant, A, Neighborhood{N}(B))

localmean(variant::Val, A::AbstractArray{T}, B::RectangularBox) where {T} =
    localmean!(variant, similar(A, float(T)), A, B)

localmean(variant::Val, A::AbstractArray{T}, B::Kernel{Bool}) where {T} =
    localmean!(variant, similar(A, float(T)), A, B)

localmean(variant::Val, A::AbstractArray{T}, B::Kernel{K}) where {T,K} =
    localmean!(variant, similar(A, float(promote_type(T,K))), A, B)

convolve(variant::Val, A::AbstractArray{T,N}, B=3) where {T,N} =
    convolve(variant, A, Neighborhood{N}(B))

convolve(variant::Val, A::AbstractArray{T}, B::RectangularBox) where {T} =
    convolve!(variant, similar(A, type_of_sum(T)), A, B)

convolve(variant::Val, A::AbstractArray{T}, B::Kernel{Bool}) where {T} =
    convolve!(variant, similar(A, type_of_sum(T)), A, B)

convolve(variant::Val, A::AbstractArray{T}, B::Kernel{K}) where {T,K} =
    convolve!(variant, similar(A, type_of_sum(promote_type(T,K))), A, B)

dilate(variant::Val, A::AbstractArray, args...) =
    dilate!(variant, similar(A), A, args...)

erode(variant::Val, A::AbstractArray, args...) =
    erode!(variant, similar(A), A, args...)

closing(variant::Val, A::AbstractArray{T,N}, B=3) where {T,N} =
    closing(variant, A, Neighborhood{N}(B))

closing(variant::Val, A::AbstractArray, B::Neighborhood) =
    erode(variant, dilate(variant, A, B), B)

opening(variant::Val, A::AbstractArray{T,N}, B=3) where {T,N} =
    opening(variant, A, Neighborhood{N}(B))

opening(variant::Val, A::AbstractArray, B::Neighborhood) =
    dilate(variant, erode(variant, A, B), B)

top_hat(variant::Val, a, r=3) = a .- opening(variant, a, r)
top_hat(variant::Val, a, r, s) =
    top_hat(variant, closing(variant, a, s), r)

bottom_hat(variant::Val, a, r=3) = closing(variant, a, r) .- a
bottom_hat(variant::Val, a, r, s) =
    bottom_hat(variant, opening(variant, a, s), r)

localextrema(variant::Val, A::AbstractArray, args...) =
    localextrema!(variant, similar(A), similar(A), A, args...)

function erode!(::Val{:Naive},
                dst::AbstractArray{T,N},
                A::AbstractArray{T,N},
                B::RectangularBox{N}) where {T,N}
    @assert axes(dst) == axes(A)
    localfilter!(dst, A, B,
                 (a)     -> typemax(T),
                 (v,a,b) -> min(v, a),
                 (d,i,v) -> d[i] = v)
end

function dilate!(::Val{:Naive},
                 dst::AbstractArray{T,N},
                 A::AbstractArray{T,N},
                 B::RectangularBox{N}) where {T,N}
    @assert axes(dst) == axes(A)
    localfilter!(dst, A, B,
                 (a)     -> typemin(T),
                 (v,a,b) -> max(v, a),
                 (d,i,v) -> d[i] = v)
end

#-----------------------------------------------------------------------------------------
# Variants for computing intersecting regions. This is the most critical part for indexing
# a neighborhood (apart from using a non-naive algorithm).

# "Base" variant: use constructors and methods provided by the Base package.
@inline function _cartesianregion(::Val{:Base},
                                  imin::CartesianIndex{N},
                                  imax::CartesianIndex{N},
                                  i::CartesianIndex{N},
                                  off::CartesianIndex{N}) where N
    cartesian_region(max(imin, i - off), min(imax, i + off))
end
@inline function _cartesianregion(::Val{:Base},
                                  imin::CartesianIndex{N},
                                  imax::CartesianIndex{N},
                                  i::CartesianIndex{N},
                                  kmin::CartesianIndex{N},
                                  kmax::CartesianIndex{N}) where N
    cartesian_region(max(imin, i - kmax), min(imax, i - kmin))
end

# "NTuple" variant: use `ntuple()` to expand expressions.
@inline function _cartesianregion(::Val{:NTuple},
                                  imin::CartesianIndex{N},
                                  imax::CartesianIndex{N},
                                  i::CartesianIndex{N},
                                  off::CartesianIndex{N}) where N
    cartesian_region(ntuple(k -> (max(imin[k], i[k] - off[k]) :
                                 min(imax[k], i[k] + off[k])), N))
end
@inline function _cartesianregion(::Val{:NTuple},
                                  imin::CartesianIndex{N},
                                  imax::CartesianIndex{N},
                                  i::CartesianIndex{N},
                                  kmin::CartesianIndex{N},
                                  kmax::CartesianIndex{N}) where N
    cartesian_region(ntuple(k -> (max(imin[k], i[k] - kmax[k]) :
                                 min(imax[k], i[k] - kmin[k])), N))
end

# "NTupleVal" variant: use `ntuple()` to expand expressions and `Val(...)` to force
# specialized versions.
for N in (1,2,3,4)
    @eval begin
        @inline function _cartesianregion(::Val{:NTupleVar},
                                          imin::CartesianIndex{$N},
                                          imax::CartesianIndex{$N},
                                          i::CartesianIndex{$N},
                                          off::CartesianIndex{$N})
            cartesian_region(ntuple(k -> (max(imin[k], i[k] - off[k]) :
                                         min(imax[k], i[k] + off[k])),
                                   Val($N)))
        end
        @inline function _cartesianregion(::Val{:NTupleVar},
                                          imin::CartesianIndex{$N},
                                          imax::CartesianIndex{$N},
                                          i::CartesianIndex{$N},
                                          kmin::CartesianIndex{$N},
                                          kmax::CartesianIndex{$N})
            cartesian_region(ntuple(k -> (max(imin[k], i[k] - kmax[k]) :
                                         min(imax[k], i[k] - kmin[k])),
                                   Val($N)))
        end
    end
end

# "Map" variant: Use `map()` to expand expressions.
@inline _range(imin::Int, imax::Int, i::Int, off::Int) =
    max(imin, i - off) : min(imax, i + off)
@inline _range(imin::Int, imax::Int, i::Int, kmin::Int, kmax::Int) =
    max(imin, i - kmax) : min(imax, i - kmin)
@inline function _cartesianregion(::Val{:Map},
                                  imin::CartesianIndex{N},
                                  imax::CartesianIndex{N},
                                  i::CartesianIndex{N},
                                  off::CartesianIndex{N}) where N
    cartesian_region(map(_range, imin.I, imax.I, i.I, off.I))
end
@inline function _cartesianregion(::Val{:Map},
                                  imin::CartesianIndex{N},
                                  imax::CartesianIndex{N},
                                  i::CartesianIndex{N},
                                  kmin::CartesianIndex{N},
                                  kmax::CartesianIndex{N}) where N
    cartesian_region(map(_range, imin.I, imax.I, i.I, kmin.I, kmax.I))
end

#-----------------------------------------------------------------------------------------
# Methods for rectangular boxes (with optimization for centered boxes).

function localmean!(variant::Val,
                    dst::AbstractArray{<:Any,N},
                    A::AbstractArray{<:Any,N},
                    B::RectangularBox{N}) where {N}
    @assert axes(dst) == axes(A)
    R = cartesian_region(A)
    imin, imax = limits(R)
    kmin, kmax = limits(B)
    T = type_of_sum(eltype(A))
    if kmin === -kmax
        off = kmax
        @inbounds for i in R
            n, s = 0, zero(T)
            for j in _cartesianregion(variant, imin, imax, i, off)
                s += A[j]
                n += 1
            end
            store!(dst, i, s/n)
        end
    else
        @inbounds for i in R
            n, s = 0, zero(T)
            for j in _cartesianregion(variant, imin, imax, i, kmin, kmax)
                s += A[j]
                n += 1
            end
            store!(dst, i, s/n)
        end
    end
    return dst
end

function erode!(variant::Val,
                Amin::AbstractArray{T,N},
                A::AbstractArray{T,N},
                B::RectangularBox{N}) where {T,N}
    @assert axes(Amin) == axes(A)
    R = cartesian_region(A)
    imin, imax = limits(R)
    kmin, kmax = limits(B)
    tmax = typemax(T)
    if kmin === -kmax
        off = kmax
        @inbounds for i in R
            vmin = tmax
            for j in _cartesianregion(variant, imin, imax, i, off)
                vmin = min(vmin, A[j])
            end
            Amin[i] = vmin
        end
    else
        @inbounds for i in R
            vmin = tmax
            for j in _cartesianregion(variant, imin, imax, i, kmin, kmax)
                vmin = min(vmin, A[j])
            end
            Amin[i] = vmin
        end
    end
    return Amin
end

function dilate!(variant::Val,
                 Amax::AbstractArray{T,N},
                 A::AbstractArray{T,N},
                 B::RectangularBox{N}) where {T,N}
    @assert axes(Amax) == axes(A)
    R = cartesian_region(A)
    imin, imax = limits(R)
    kmin, kmax = limits(B)
    tmin = typemin(T)
    if kmin === -kmax
        off = kmax
        @inbounds for i in R
            vmax = tmin
            for j in _cartesianregion(variant, imin, imax, i, off)
                vmax = max(vmax, A[j])
            end
            Amax[i] = vmax
        end
    else
        @inbounds for i in R
            vmax = tmin
            for j in _cartesianregion(variant, imin, imax, i, kmin, kmax)
                vmax = max(vmax, A[j])
            end
            Amax[i] = vmax
        end
    end
    return Amax
end

function localextrema!(variant::Val,
                       Amin::AbstractArray{T,N},
                       Amax::AbstractArray{T,N},
                       A::AbstractArray{T,N},
                       B::RectangularBox{N}) where {T,N}
    @assert axes(Amin) == axes(Amax) == axes(A)
    R = cartesian_region(A)
    imin, imax = limits(R)
    kmin, kmax = limits(B)
    tmin, tmax = limits(T)
    if kmin === -kmax
        off = kmax
        @inbounds for i in R
            vmin, vmax = tmax, tmin
            for j in _cartesianregion(variant, imin, imax, i, off)
                vmin = min(vmin, A[j])
                vmax = max(vmax, A[j])
            end
            Amin[i] = vmin
            Amax[i] = vmax
        end
    else
        @inbounds for i in R
            vmin, vmax = tmax, tmin
            for j in _cartesianregion(variant, imin, imax, i, kmin, kmax)
                vmin = min(vmin, A[j])
                vmax = max(vmax, A[j])
            end
            Amin[i] = vmin
            Amax[i] = vmax
        end
    end
    return Amin, Amax
end

#-----------------------------------------------------------------------------------------
# Methods for kernels of booleans.

function localmean!(variant::Val,
                    dst::AbstractArray{<:Any,N},
                    A::AbstractArray{<:Any,N},
                    B::Kernel{Bool,N}) where {N}
    @assert axes(dst) == axes(A)
    R = cartesian_region(A)
    imin, imax = limits(R)
    kmin, kmax = limits(B)
    ker, off = coefs(B), offset(B)
    T = type_of_sum(eltype(A))
    @inbounds for i in R
        n, s = 0, zero(T)
        k = i + off
        for j in _cartesianregion(variant, imin, imax, i, kmin, kmax)
            if ker[k-j]
                n += 1
                s += A[j]
            end
        end
        store!(dst, i, s/n)
    end
    return dst
end

function erode!(variant::Val,
                Amin::AbstractArray{T,N},
                A::AbstractArray{T,N},
                B::Kernel{Bool,N}) where {T,N}
    @assert axes(Amin) == axes(A)
    R = cartesian_region(A)
    imin, imax = limits(R)
    kmin, kmax = limits(B)
    ker, off = coefs(B), offset(B)
    tmax = typemax(T)
    @inbounds for i in R
        vmin = tmax
        k = i + off
        for j in _cartesianregion(variant, imin, imax, i, kmin, kmax)
            #if ker[k-j] && A[j] < vmin
            #    vmin = A[j]
            #end
            vmin = ker[k-j] && A[j] < vmin ? A[j] : vmin
        end
        Amin[i] = vmin
    end
    return Amin
end

function dilate!(variant::Val,
                 Amax::AbstractArray{T,N},
                 A::AbstractArray{T,N},
                 B::Kernel{Bool,N}) where {T,N}
    @assert axes(Amax) == axes(A)
    R = cartesian_region(A)
    imin, imax = limits(R)
    kmin, kmax = limits(B)
    ker, off = coefs(B), offset(B)
    tmin = typemin(T)
    @inbounds for i in R
        vmax = tmin
        k = i + off
        for j in _cartesianregion(variant, imin, imax, i, kmin, kmax)
            #if ker[k-j] && A[j] > vmax
            #    vmax = A[j]
            #end
            vmax = ker[k-j] && A[j] > vmax ? A[j] : vmax
        end
        Amax[i] = vmax
    end
    return Amax
end

function localextrema!(variant::Val,
                       Amin::AbstractArray{T,N},
                       Amax::AbstractArray{T,N},
                       A::AbstractArray{T,N},
                       B::Kernel{Bool,N}) where {T,N}
    @assert axes(Amin) == axes(Amax) == axes(A)
    R = cartesian_region(A)
    imin, imax = limits(R)
    kmin, kmax = limits(B)
    tmin, tmax = limits(T)
    ker, off = coefs(B), offset(B)
    @inbounds for i in R
        vmin, vmax = tmax, tmin
        k = i + off
        for j in _cartesianregion(variant, imin, imax, i, kmin, kmax)
            #if ker[k-j]
            #    vmin = min(vmin, A[j])
            #    vmax = max(vmax, A[j])
            #end
            vmin = ker[k-j] && A[j] < vmin ? A[j] : vmin
            vmax = ker[k-j] && A[j] > vmax ? A[j] : vmax
        end
        Amin[i] = vmin
        Amax[i] = vmax
    end
    return Amin, Amax
end

#-----------------------------------------------------------------------------------------
# Methods for other kernels.

function localmean!(variant::Val,
                    dst::AbstractArray{<:Any,N},
                    A::AbstractArray{<:Any,N},
                    B::Kernel{<:Any,N}) where {N}
    @assert axes(dst) == axes(A)
    R = cartesian_region(A)
    imin, imax = limits(R)
    kmin, kmax = limits(B)
    ker, off = coefs(B), offset(B)
    T = type_of_sum(promote_type(eltype(A), eltype(B)))
    @inbounds for i in R
        s1, s2 = zero(T), zero(T)
        k = i + off
        for j in _cartesianregion(variant, imin, imax, i, kmin, kmax)
            w = ker[k-j]
            s1 += w*A[j]
            s2 += w
        end
        store!(dst, i, s1/s2)
    end
    return dst
end

function erode!(variant::Val,
                Amin::AbstractArray{T,N},
                A::AbstractArray{T,N},
                B::Kernel{T,N}) where {T<:AbstractFloat,N}
    @assert axes(Amin) == axes(A)
    R = cartesian_region(A)
    imin, imax = limits(R)
    kmin, kmax = limits(B)
    ker, off = coefs(B), offset(B)
    tmax = typemax(T)
    @inbounds for i in R
        vmin = tmax
        k = i + off
        for j in _cartesianregion(variant, imin, imax, i, kmin, kmax)
            vmin = min(vmin, A[j] - ker[k-j])
        end
        Amin[i] = vmin
    end
    return Amin
end

function dilate!(variant::Val,
                 Amax::AbstractArray{T,N},
                 A::AbstractArray{T,N},
                 B::Kernel{T,N}) where {T<:AbstractFloat,N}
    @assert axes(Amax) == axes(A)
    R = cartesian_region(A)
    imin, imax = limits(R)
    kmin, kmax = limits(B)
    ker, off = coefs(B), offset(B)
    tmin = typemin(T)
    @inbounds for i in R
        vmax = tmin
        k = i + off
        for j in _cartesianregion(variant, imin, imax, i, kmin, kmax)
            vmax = max(vmax, A[j] + ker[k-j])
        end
        Amax[i] = vmax
    end
    return Amax
end

function convolve!(variant::Val,
                   dst::AbstractArray{<:Any,N},
                   A::AbstractArray{<:Any,N},
                   B::Kernel{<:Any,N}) where {N}
    @assert axes(dst) == axes(A)
    R = cartesian_region(A)
    imin, imax = limits(R)
    kmin, kmax = limits(B)
    ker, off = coefs(B), offset(B)
    T = type_of_sum(promote_type(eltype(A), eltype(B)))
    @inbounds for i in R
        v = zero(T)
        k = i + off
        for j in _cartesianregion(variant, imin, imax, i, kmin, kmax)
            v += A[j]*ker[k-j]
        end
        store!(dst, i, v)
    end
    return dst
end

function localextrema!(variant::Val,
                       Amin::AbstractArray{T,N},
                       Amax::AbstractArray{T,N},
                       A::AbstractArray{T,N},
                       B::Kernel{T,N}) where {T<:AbstractFloat,N}
    @assert axes(Amin) == axes(Amax) == axes(A)
    R = cartesian_region(A)
    imin, imax = limits(R)
    kmin, kmax = limits(B)
    tmin, tmax = limits(T)
    ker, off = coefs(B), offset(B)
    @inbounds for i in R
        vmin, vmax = tmax, tmin
        k = i + off
        for j in _cartesianregion(variant, imin, imax, i, kmin, kmax)
            vmin = min(vmin, A[j] - ker[k-j])
            vmax = max(vmax, A[j] + ker[k-j])
        end
        Amin[i] = vmin
        Amax[i] = vmax
    end
    return Amin, Amax
end

end # module
