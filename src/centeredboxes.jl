#
# centeredboxes.jl --
#
# Implementation of basic local operations on a neighborhood defined by a
# centered rectangular box.
#
#------------------------------------------------------------------------------
#
# This file is part of the `LocalFilters.jl` package licensed under the MIT
# "Expat" License.
#
# Copyright (C) 2017, Éric Thiébaut.
#

"""
A centered box is a rectangular neighborhood which is defined by the offsets of
the last element of the neighborhood with respect to the center of the box.
"""
struct CenteredBox{N} <: Neighborhood{N}
    last::CartesianIndex{N}
end

@inline function halfdim(n::Integer)
    @assert n ≥ 1 && isodd(n) "dimensions of centered box must be ≥ 1 and odd"
    Int(n)>>1
end

CenteredBox(B::CenteredBox) = B

CenteredBox(siz::Integer...) =
    (N = length(siz);
     CenteredBox{N}(CartesianIndex(ntuple(i -> halfdim(siz[i]), N))))

CenteredBox(siz::Vector{Integer}) =
    (N = length(siz);
     CenteredBox{N}(CartesianIndex(ntuple(i -> halfdim(siz[i]), N))))

CenteredBox(siz::NTuple{N,Integer}) where {N} =
    CenteredBox{N}(CartesianIndex(ntuple(i -> halfdim(siz[i]), N)))

eltype(B::CenteredBox) = Bool
size(B::CenteredBox, i) = 2*last(B)[i] + 1
first(B::CenteredBox) = -last(B)
last(B::CenteredBox) = B.last
getindex(B::CenteredBox, i::CartesianIndex) = true
getindex(B::CenteredBox, i::Integer...) = true

function localfilter!(dst,
                      A::AbstractArray{T,N},
                      B::CenteredBox{N},
                      initial::Function,
                      update::Function,
                      store::Function) where {T,N}
    R = CartesianRange(size(A))
    imin, imax = limits(R)
    off = last(B)
    @inbounds for i in R
        v = initial()
        for j in CartesianRange(max(imin, i - off), min(imax, i + off))
            v = update(v, A[j], true)
        end
        store(dst, i, v)
    end
    return dst
end

function localmean!(dst::AbstractArray{T,N},
                    A::AbstractArray{T,N},
                    B::CenteredBox{N}) where {T,N}
    @assert size(dst) == size(A)
    localfilter!(dst, A, B,
                 ()      -> (zero(T), 0),
                 (v,a,b) -> (v[1] + a, v[2] + 1),
                 (d,i,v) -> d[i] = v[1]/v[2])
end

function erode!(dst::AbstractArray{T,N},
                A::AbstractArray{T,N},
                B::CenteredBox{N}) where {T,N}
    @assert size(dst) == size(A)
    localfilter!(dst, A, B,
                 ()      -> typemax(T),
                 (v,a,b) -> min(v, a),
                 (d,i,v) -> d[i] = v)
end

function dilate!(dst::AbstractArray{T,N},
                 A::AbstractArray{T,N},
                 B::CenteredBox{N}) where {T,N}
    @assert size(dst) == size(A)
    localfilter!(dst, A, B,
                 ()      -> typemin(T),
                 (v,a,b) -> max(v, a),
                 (d,i,v) -> d[i] = v)
end

function localextrema!(Amin::AbstractArray{T,N},
                       Amax::AbstractArray{T,N},
                       A::AbstractArray{T,N},
                       B::CenteredBox{N}) where {T,N}
    @assert size(Amin) == size(Amax) == size(A)
    localfilter!((Amin, Amax), A, B,
                 ()      -> (typemax(T),
                             typemin(T)),
                 (v,a,b) -> (min(v[1], a),
                             max(v[2], a)),
                 (d,i,v) -> (Amin[i], Amax[i]) = v)
end

function convolve!(dst::AbstractArray{S,N}, A::AbstractArray{T,N},
                   B::CenteredBox{N}) where {S,T,N}
    @assert size(dst) == size(A)
    localfilter!(dst, A, B,
                 ()      -> zero(S),
                 (v,a,b) -> v + S(a),
                 (d,i,v) -> d[i] = v)
end
