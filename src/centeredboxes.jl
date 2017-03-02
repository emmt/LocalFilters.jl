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

function localmean!{T,N}(dst::AbstractArray{T,N},
                         A::AbstractArray{T,N},
                         B::CenteredBox{N})
    @assert size(dst) == size(A)
    R = CartesianRange(size(A))
    imin, imax = limits(R)
    off = last(B)
    @inbounds for i in R
        n, s = 0, zero(T)
        for j in CartesianRange(max(imin, i - off), min(imax, i + off))
            s += A[j]
            n += 1
        end
        dst[i] = s/n
    end
    return dst
end

function erode!{T,N}(Amin::AbstractArray{T,N},
                     A::AbstractArray{T,N},
                     B::CenteredBox{N})
    @assert size(Amin) == size(A)
    R = CartesianRange(size(A))
    imin, imax = limits(R)
    tmax = typemax(T)
    off = last(B)
    @inbounds for i in R
        vmin = tmax
        for j in CartesianRange(max(imin, i - off), min(imax, i + off))
            vmin = min(vmin, A[j])
        end
        Amin[i] = vmin
    end
    return Amin
end

function dilate!{T,N}(Amax::AbstractArray{T,N},
                      A::AbstractArray{T,N},
                      B::CenteredBox{N})
    @assert size(Amax) == size(A)
    R = CartesianRange(size(A))
    imin, imax = limits(R)
    tmin = typemin(T)
    off = last(B)
    @inbounds for i in R
        vmax = tmin
        for j in CartesianRange(max(imin, i - off), min(imax, i + off))
            vmax = max(vmax, A[j])
        end
        Amax[i] = vmax
    end
    return Amax
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
