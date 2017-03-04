# cartesianboxes.jl --
#
# Implementation of basic local on a neighborhood defined by a rectangular box,
# possibly off-centered.

"""
A rectangular (cartesian) box is defined by the bounds of the neighborhood
with respect to the center of the box.
"""
immutable CartesianBox{N} <: Neighborhood{N}
    # We have to define this type even though it is nothing more than a
    # CartesianRange because of the heritage.
    bounds::CartesianRange{CartesianIndex{N}}
end

CartesianBox(B::CartesianBox) = B

CartesianBox(B::CenteredBox) = CartesianBox(CartesianRange(B))

CartesianBox{N}(I0::CartesianIndex{N}, I1::CartesianIndex{N}) =
    CartesianBox(CartesianRange(I0, I1))

CartesianBox{N}(dims::NTuple{N,Integer}, offs::NTuple{N,Integer}) =
    (I = CartesianIndex(offs);
     CartesianBox(one(I) - I, CartesianIndex(dims) - I))

CartesianBox{T<:Integer}(inds::AbstractUnitRange{T}...) =
    CartesianBox(inds)

CartesianBox{N,T<:Integer}(inds::NTuple{N,AbstractUnitRange{T}}) =
    CartesianBox(CartesianIndex(map(first, inds)),
                 CartesianIndex(map(last, inds)))

eltype(B::CartesianBox) = Bool
size(B::CartesianBox, i) = max(last(B)[i] - first(B)[i] + 1, 0)
first(B::CartesianBox) = first(B.bounds)
last(B::CartesianBox) = last(B.bounds)
limits(B::CartesianBox) = first(B), last(B)
getindex(B::CartesianBox, I::CartesianIndex) = true
getindex(B::CartesianBox, inds::Integer...) = true

#
# With `dst` the destination, `A` the source, and `B` the structuring
# element the pseudo-code to implement a local operation writes:
#
#     for i ∈ Sup(A)
#         v = initial()
#         for k ∈ Sup(B) and j = i - k ∈ Sup(A)
#             v = update(v, A[j], B[k])
#         end
#         dst[i] = final(v)
#     end
#
# where `A` `Sup(A)` yields the support of `A` (that is the set of indices in
# `A`) and likely `Sub(B)` for `B`.
#
# Equivalent form:
#
#     for i ∈ Sup(A)
#         v = initial()
#         for j ∈ Sup(A) and k = i - j ∈ Sup(B)
#             v = update(v, A[j], B[k])
#         end
#         dst[i] = final(v)
#     end
#
# In the second form, the bounds for `j` are:
#
#    imin ≤ j ≤ imax   and   kmin ≤ k = i - j ≤ kmax
#
# where `imin` and `imax` are the bounds for `A` while `kmin` and `kmax` are
# the bounds for `B`.  The above constraints are identical to:
#
#    max(imin, i - kmax) ≤ j ≤ min(imax, i - kmin)
#

function localfilter!{T,N}(dst, A::AbstractArray{T,N},
                           B::CartesianBox{N}, initial::Function,
                           update::Function, store::Function)
    R = CartesianRange(size(A))
    imin, imax = limits(R)
    kmin, kmax = limits(B)
    @inbounds for i in R
        v = initial()
        for j in CartesianRange(max(imin, i - kmax), min(imax, i - kmin))
            v = update(v, A[j], true)
        end
        store(dst, i, v)
    end
    return dst
end

function localmean!{T,N}(dst::AbstractArray{T,N}, A::AbstractArray{T,N},
                         B::CartesianBox{N})
    @assert size(dst) == size(A)
    localfilter!(dst, A, B,
                 ()      -> (zero(T), 0),
                 (v,a,b) -> (v[1] + a, v[2] + 1),
                 (d,i,v) -> d[i] = v[1]/v[2])
end

function erode!{T,N}(dst::AbstractArray{T,N}, A::AbstractArray{T,N},
                     B::CartesianBox{N})
    @assert size(dst) == size(A)
    localfilter!(dst, A, B,
                 ()      -> typemax(T),
                 (v,a,b) -> min(v, a),
                 (d,i,v) -> d[i] = v)
end

function dilate!{T,N}(dst::AbstractArray{T,N},
                      A::AbstractArray{T,N},
                      B::CartesianBox{N})
    @assert size(dst) == size(A)
    localfilter!(dst, A, B,
                 ()      -> typemin(T),
                 (v,a,b) -> max(v, a),
                 (d,i,v) -> d[i] = v)
end

function localextrema!{T,N}(Amin::AbstractArray{T,N},
                            Amax::AbstractArray{T,N},
                            A::AbstractArray{T,N},
                            B::CartesianBox{N})
    @assert size(Amin) == size(Amax) == size(A)
    localfilter!((Amin, Amax), A, B,
                 ()      -> (typemax(T),
                             typemin(T)),
                 (v,a,b) -> (min(v[1], a),
                             max(v[2], a)),
                 (d,i,v) -> (Amin[i], Amax[i]) = v)
end

function convolve!{S,T,N}(dst::AbstractArray{S,N}, A::AbstractArray{T,N},
                          B::CartesianBox{N})
    @assert size(dst) == size(A)
    localfilter!(dst, A, B,
                 ()      -> zero(S),
                 (v,a,b) -> v + S(a),
                 (d,i,v) -> d[i] = v)
end
