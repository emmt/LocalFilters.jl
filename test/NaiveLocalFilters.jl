#
# NaiveLocalFilters.jl -
#
# Naive (non-optimized) implementation of local filters.  This module provides
# reference implementations of methods for checking results and benchmarking.
#
#------------------------------------------------------------------------------
#
# This file is part of the `LocalFilters.jl` package licensed under the MIT
# "Expat" License.
#
# Copyright (C) 2017-2018, Éric Thiébaut.
#

module NaiveLocalFilters

using Compat
using LocalFilters
using LocalFilters:
    Kernel,
    Neighborhood,
    RectangularBox,
    cartesianregion,
    coefs,
    limits,
    offset

import LocalFilters:
    cartesianregion,
    convolve!,
    dilate!,
    erode!,
    localextrema!,
    localfilter!,
    localmean!

# function localfilter!(dst,
#                       A::AbstractArray{T,N},
#                       B::RectangularBox{N},
#                       initial::Function,
#                       update::Function,
#                       store::Function) where {T,N}
#     R = cartesianregion(A)
#     imin, imax = limits(R)
#     kmin, kmax = limits(B)
#     @inbounds for i in R
#         v = initial(A[i])
#         @simd for j in cartesianregion(max(imin, i - kmax),
#                                        min(imax, i - kmin))
#             v = update(v, A[j], true)
#         end
#         store(dst, i, v)
#     end
#     return dst
# end
#
# function erode!(dst::AbstractArray{T,N},
#                 A::AbstractArray{T,N},
#                 B::RectangularBox{N}) where {T,N}
#     @assert size(dst) == size(A)
#     localfilter!(dst, A, B,
#                  (a)     -> typemax(T),
#                  (v,a,b) -> min(v, a),
#                  (d,i,v) -> d[i] = v)
# end
#
# function dilate!(dst::AbstractArray{T,N},
#                  A::AbstractArray{T,N},
#                  B::RectangularBox{N}) where {T,N}
#     @assert size(dst) == size(A)
#     localfilter!(dst, A, B,
#                  (a)     -> typemin(T),
#                  (v,a,b) -> max(v, a),
#                  (d,i,v) -> d[i] = v)
# end

#------------------------------------------------------------------------------
# Variants for computing interesecting regions.  This is the most critical
# part for indexing a neighborhood (apart from using a non-naive algorithm).

# "Base" variant: use constructors and methods provided by the Base package.
@inline function _cartesianregion(::Type{Val{:Base}},
                         imin::CartesianIndex{N},
                         imax::CartesianIndex{N},
                         i::CartesianIndex{N},
                         off::CartesianIndex{N}) where N
    cartesianregion(max(imin, i - off), min(imax, i + off))
end
@inline function _cartesianregion(::Type{Val{:Base}},
                         imin::CartesianIndex{N},
                         imax::CartesianIndex{N},
                         i::CartesianIndex{N},
                         kmin::CartesianIndex{N},
                         kmax::CartesianIndex{N}) where N
    cartesianregion(max(imin, i - kmax), min(imax, i - kmin))
end

# "NTuple" variant: use `ntuple()` to expand expressions.
@inline function _cartesianregion(::Type{Val{:NTuple}},
                         imin::CartesianIndex{N},
                         imax::CartesianIndex{N},
                         i::CartesianIndex{N},
                         off::CartesianIndex{N}) where N
    cartesianregion(ntuple((k)->(max(imin[k], i[k] - off[k]) :
                        min(imax[k], i[k] + off[k])), N))
end
@inline function _cartesianregion(::Type{Val{:NTuple}},
                         imin::CartesianIndex{N},
                         imax::CartesianIndex{N},
                         i::CartesianIndex{N},
                         kmin::CartesianIndex{N},
                         kmax::CartesianIndex{N}) where N
    cartesianregion(ntuple((k)->(max(imin[k], i[k] - kmax[k]) :
                        min(imax[k], i[k] - kmin[k])), N))
end

# "NTupleVal" variant: use `ntuple()` to expand expressions and `Val(...)` to
# force specialized versions.
for N in (1,2,3,4)
    @eval begin
        @inline function _cartesianregion(::Type{Val{:NTupleVar}},
                                 imin::CartesianIndex{$N},
                                 imax::CartesianIndex{$N},
                                 i::CartesianIndex{$N},
                                 off::CartesianIndex{$N})
            cartesianregion(ntuple((k)->(max(imin[k], i[k] - off[k]) :
                                min(imax[k], i[k] + off[k])), Val($N)))
        end
        @inline function _cartesianregion(::Type{Val{:NTupleVar}},
                                 imin::CartesianIndex{$N},
                                 imax::CartesianIndex{$N},
                                 i::CartesianIndex{$N},
                                 kmin::CartesianIndex{$N},
                                 kmax::CartesianIndex{$N})
            cartesianregion(ntuple((k)->(max(imin[k], i[k] - kmax[k]) :
                                min(imax[k], i[k] - kmin[k])), Val($N)))
        end
    end
end

# "Map" variant: Use `map()` to expand expressions.
@inline _range(imin::Int, imax::Int, i::Int, off::Int) =
    max(imin, i - off) : min(imax, i + off)
@inline _range(imin::Int, imax::Int, i::Int, kmin::Int, kmax::Int) =
    max(imin, i - kmax) : min(imax, i - kmin)
@inline function _cartesianregion(::Type{Val{:Map}},
                                  imin::CartesianIndex{N},
                                  imax::CartesianIndex{N},
                                  i::CartesianIndex{N},
                                  off::CartesianIndex{N}) where N
    cartesianregion(map(_range, imin.I, imax.I, i.I, off.I))
end
@inline function _cartesianregion(::Type{Val{:Map}},
                                  imin::CartesianIndex{N},
                                  imax::CartesianIndex{N},
                                  i::CartesianIndex{N},
                                  kmin::CartesianIndex{N},
                                  kmax::CartesianIndex{N}) where N
    cartesianregion(map(_range, imin.I, imax.I, i.I, kmin.I, kmax.I))
end

#------------------------------------------------------------------------------
# Methods for rectangular boxes (with optimization for centered boxes).

function localmean!(::Type{Op},
                    dst::AbstractArray{T,N},
                    A::AbstractArray{T,N},
                    B::RectangularBox{N}) where {T,N,Op<:Val}
    @assert axes(dst) == axes(A)
    R = cartesianregion(A)
    imin, imax = limits(R)
    kmin, kmax = limits(B)
    if kmin === -kmax
        off = kmax
        @inbounds for i in R
            n, s = 0, zero(T)
            for j in _cartesianregion(Op, imin, imax, i, off)
                s += A[j]
                n += 1
            end
            dst[i] = s/n
        end
    else
        @inbounds for i in R
            n, s = 0, zero(T)
            for j in _cartesianregion(Op, imin, imax, i, kmin, kmax)
                s += A[j]
                n += 1
            end
            dst[i] = s/n
        end
    end
    return dst
end

function erode!(::Type{Op},
                Amin::AbstractArray{T,N},
                A::AbstractArray{T,N},
                B::RectangularBox{N}) where {T,N,Op<:Val}
    @assert size(Amin) == size(A)
    R = cartesianregion(A)
    imin, imax = limits(R)
    kmin, kmax = limits(B)
    tmax = typemax(T)
    if kmin === -kmax
        off = kmax
        @inbounds for i in R
            vmin = tmax
            for j in _cartesianregion(Op, imin, imax, i, off)
                vmin = min(vmin, A[j])
            end
            Amin[i] = vmin
        end
    else
        @inbounds for i in R
            vmin = tmax
            for j in _cartesianregion(Op, imin, imax, i, kmin, kmax)
                vmin = min(vmin, A[j])
            end
            Amin[i] = vmin
        end
    end
    return Amin
end

function dilate!(::Type{Op},
                 Amax::AbstractArray{T,N},
                 A::AbstractArray{T,N},
                 B::RectangularBox{N}) where {T,N,Op<:Val}
    @assert size(Amax) == size(A)
    R = cartesianregion(A)
    imin, imax = limits(R)
    kmin, kmax = limits(B)
    tmin = typemin(T)
    if kmin === -kmax
        off = kmax
        @inbounds for i in R
            vmax = tmin
            for j in _cartesianregion(Op, imin, imax, i, off)
                vmax = max(vmax, A[j])
            end
            Amax[i] = vmax
        end
    else
        @inbounds for i in R
            vmax = tmin
            for j in _cartesianregion(Op, imin, imax, i, kmin, kmax)
                vmax = max(vmax, A[j])
            end
            Amax[i] = vmax
        end
    end
    return Amax
end

function localextrema!(::Type{Op},
                       Amin::AbstractArray{T,N},
                       Amax::AbstractArray{T,N},
                       A::AbstractArray{T,N},
                       B::RectangularBox{N}) where {T,N,Op<:Val}
    @assert size(Amin) == size(Amax) == size(A)
    R = cartesianregion(A)
    imin, imax = limits(R)
    kmin, kmax = limits(B)
    tmin, tmax = limits(T)
    if kmin === -kmax
        off = kmax
        @inbounds for i in R
            vmin, vmax = tmax, tmin
            for j in _cartesianregion(Op, imin, imax, i, off)
                vmin = min(vmin, A[j])
                vmax = max(vmax, A[j])
            end
            Amin[i] = vmin
            Amax[i] = vmax
        end
    else
        @inbounds for i in R
            vmin, vmax = tmax, tmin
            for j in _cartesianregion(Op, imin, imax, i, kmin, kmax)
                vmin = min(vmin, A[j])
                vmax = max(vmax, A[j])
            end
            Amin[i] = vmin
            Amax[i] = vmax
        end
    end
    return Amin, Amax
end

#------------------------------------------------------------------------------
# Nethods for kernels of booleans.

function localmean!(::Type{Op},
                    dst::AbstractArray{T,N},
                    A::AbstractArray{T,N},
                    B::Kernel{Bool,N}) where {T,N,Op<:Val}
    @assert size(dst) == size(A)
    R = cartesianregion(A)
    imin, imax = limits(R)
    kmin, kmax = limits(B)
    ker, off = coefs(B), offset(B)
    @inbounds for i in R
        n, s = 0, zero(T)
        k = i + off
        for j in _cartesianregion(Op, imin, imax, i, kmin, kmax)
            if ker[k-j]
                n += 1
                s += A[j]
            end
        end
        dst[i] = s/n
    end
    return dst
end

function erode!(::Type{Op},
                Amin::AbstractArray{T,N},
                A::AbstractArray{T,N},
                B::Kernel{Bool,N}) where {T,N,Op<:Val}
    @assert size(Amin) == size(A)
    R = cartesianregion(A)
    imin, imax = limits(R)
    kmin, kmax = limits(B)
    ker, off = coefs(B), offset(B)
    tmax = typemax(T)
    @inbounds for i in R
        vmin = tmax
        k = i + off
        for j in _cartesianregion(Op, imin, imax, i, kmin, kmax)
            #if ker[k-j] && A[j] < vmin
            #    vmin = A[j]
            #end
            vmin = ker[k-j] && A[j] < vmin ? A[j] : vmin
        end
        Amin[i] = vmin
    end
    return Amin
end

function dilate!(::Type{Op},
                 Amax::AbstractArray{T,N},
                 A::AbstractArray{T,N},
                 B::Kernel{Bool,N}) where {T,N,Op<:Val}
    @assert size(Amax) == size(A)
    R = cartesianregion(A)
    imin, imax = limits(R)
    kmin, kmax = limits(B)
    ker, off = coefs(B), offset(B)
    tmin = typemin(T)
    @inbounds for i in R
        vmax = tmin
        k = i + off
        for j in _cartesianregion(Op, imin, imax, i, kmin, kmax)
            #if ker[k-j] && A[j] > vmax
            #    vmax = A[j]
            #end
            vmax = ker[k-j] && A[j] > vmax ? A[j] : vmax
        end
        Amax[i] = vmax
    end
    return Amax
end

function localextrema!(::Type{Op},
                       Amin::AbstractArray{T,N},
                       Amax::AbstractArray{T,N},
                       A::AbstractArray{T,N},
                       B::Kernel{Bool,N}) where {T,N,Op<:Val}
    @assert size(Amin) == size(Amax) == size(A)
    R = cartesianregion(A)
    imin, imax = limits(R)
    kmin, kmax = limits(B)
    tmin, tmax = limits(T)
    ker, off = coefs(B), offset(B)
    @inbounds for i in R
        vmin, vmax = tmax, tmin
        k = i + off
        for j in _cartesianregion(Op, imin, imax, i, kmin, kmax)
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

#------------------------------------------------------------------------------
# Methods for other kernels.

function localmean!(::Type{Op},
                    dst::AbstractArray{T,N},
                    A::AbstractArray{T,N},
                    B::Kernel{T,N}) where {T<:AbstractFloat,N,Op<:Val}
    @assert size(dst) == size(A)
    R = cartesianregion(A)
    imin, imax = limits(R)
    kmin, kmax = limits(B)
    ker, off = coefs(B), offset(B)
    @inbounds for i in R
        s1, s2 = zero(T), zero(T)
        k = i + off
        for j in _cartesianregion(Op, imin, imax, i, kmin, kmax)
            w = ker[k-j]
            s1 += w*A[j]
            s2 += w
        end
        dst[i] = s1/s2
    end
    return dst
end

function erode!(::Type{Op},
                Amin::AbstractArray{T,N},
                A::AbstractArray{T,N},
                B::Kernel{T,N}) where {T<:AbstractFloat,N,Op<:Val}
    @assert size(Amin) == size(A)
    R = cartesianregion(A)
    imin, imax = limits(R)
    kmin, kmax = limits(B)
    ker, off = coefs(B), offset(B)
    tmax = typemax(T)
    @inbounds for i in R
        vmin = tmax
        k = i + off
        for j in _cartesianregion(Op, imin, imax, i, kmin, kmax)
            vmin = min(vmin, A[j] - ker[k-j])
        end
        Amin[i] = vmin
    end
    return Amin
end

function dilate!(::Type{Op},
                 Amax::AbstractArray{T,N},
                 A::AbstractArray{T,N},
                 B::Kernel{T,N}) where {T<:AbstractFloat,N,Op<:Val}
    @assert size(Amax) == size(A)
    R = cartesianregion(A)
    imin, imax = limits(R)
    kmin, kmax = limits(B)
    ker, off = coefs(B), offset(B)
    tmin = typemin(T)
    @inbounds for i in R
        vmax = tmin
        k = i + off
        for j in _cartesianregion(Op, imin, imax, i, kmin, kmax)
            vmax = max(vmax, A[j] + ker[k-j])
        end
        Amax[i] = vmax
    end
    return Amax
end

function convolve!(::Type{Op},
                   dst::AbstractArray{T,N},
                   A::AbstractArray{T,N},
                   B::Kernel{T,N}) where {T<:AbstractFloat,N,Op<:Val}
    @assert size(dst) == size(A)
    R = cartesianregion(A)
    imin, imax = limits(R)
    kmin, kmax = limits(B)
    ker, off = coefs(B), offset(B)
    @inbounds for i in R
        v = zero(T)
        k = i + off
        for j in _cartesianregion(Op, imin, imax, i, kmin, kmax)
            v += A[j]*ker[k-j]
        end
        dst[i] = v
    end
    return dst
end

function localextrema!(::Type{Op},
                       Amin::AbstractArray{T,N},
                       Amax::AbstractArray{T,N},
                       A::AbstractArray{T,N},
                       B::Kernel{T,N}) where {T<:AbstractFloat,N,Op<:Val}
    @assert size(Amin) == size(Amax) == size(A)
    R = cartesianregion(A)
    imin, imax = limits(R)
    kmin, kmax = limits(B)
    tmin, tmax = limits(T)
    ker, off = coefs(B), offset(B)
    @inbounds for i in R
        vmin, vmax = tmax, tmin
        k = i + off
        for j in _cartesianregion(Op, imin, imax, i, kmin, kmax)
            vmin = min(vmin, A[j] - ker[k-j])
            vmax = max(vmax, A[j] + ker[k-j])
        end
        Amin[i] = vmin
        Amax[i] = vmax
    end
    return Amin, Amax
end

end # module
