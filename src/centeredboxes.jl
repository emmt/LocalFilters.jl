# centeredboxes.jl --
#
# Implementation of basic local operations on a neighborhood defined by a
# centered rectangular box.

"""
A centered box is a rectangular neighborhood which is defined by the offsets of
the last element of the neighborhood with respect to the center of the box.
"""
immutable CenteredBox{N} <: Neighborhood{N}
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

CenteredBox{N}(siz::NTuple{N,Integer}) =
    CenteredBox{N}(CartesianIndex(ntuple(i -> halfdim(siz[i]), N)))

size(B::CenteredBox, i) = 2*last(B)[i] + 1
first(B::CenteredBox) = -last(B)
last(B::CenteredBox) = B.last
getindex(B::CenteredBox, i::CartesianIndex) = true
getindex(B::CenteredBox, i::Integer...) = true

function localfilter!{T,N}(dst::AbstractArray{T,N}, A::AbstractArray{T,N},
                           B::CenteredBox{N}, initial::Function,
                           update::Function, final::Function)
    @assert size(dst) == size(A)
    R = CartesianRange(size(A))
    imin, imax = limits(R)
    off = last(B)
    @inbounds for i in R
        v = initial()
        for j in CartesianRange(max(imin, i - off), min(imax, i + off))
            v = update(v, A[j], true)
        end
        dst[i] = final(v)
    end
    return dst
end

function localmean!{T,N}(dst::AbstractArray{T,N}, A::AbstractArray{T,N},
                         B::CenteredBox{N})
    localfilter!(dst, A, B,
                 ()      -> (zero(T), 0),
                 (v,a,b) -> (v[1] + a, v[2] + 1),
                 (v)     -> v[1]/v[2])
end

function erode!{T,N}(dst::AbstractArray{T,N}, A::AbstractArray{T,N},
                     B::CenteredBox{N})
    localfilter!(dst, A, B,
                 ()      -> typemax(T),
                 (v,a,b) -> min(v, a),
                 (v)     -> v)
end

function dilate!{T,N}(dst::AbstractArray{T,N},
                      A::AbstractArray{T,N},
                      B::CenteredBox{N})
    localfilter!(dst, A, B,
                 ()      -> typemin(T),
                 (v,a,b) -> max(v, a),
                 (v)     -> v)
end

function localextrema!{T,N}(Amin::AbstractArray{T,N},
                            Amax::AbstractArray{T,N},
                            A::AbstractArray{T,N},
                            B::CenteredBox{N})
    @assert size(Amin) == size(Amax) == size(A)
    R = CartesianRange(size(A))
    imin, imax = limits(R)
    tmin, tmax = limits(T)
    off = last(B)
    @inbounds for i in R
        vmin, vmax = tmax, tmin
        for j in CartesianRange(max(imin, i - off), min(imax, i + off))
            vmin = min(vmin, A[j])
            vmax = max(vmax, A[j])
        end
        Amin[i] = vmin
        Amax[i] = vmax
    end
    return Amin, Amax
end

function convolve!{S,T,N}(dst::AbstractArray{S,N}, A::AbstractArray{T,N},
                          B::CenteredBox{N})
    localfilter!(dst, A, B,
                 ()      -> zero(S),
                 (v,a,b) -> v + S(a),
                 (v)     -> v)
end
