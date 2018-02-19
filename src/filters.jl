#
# filters.jl --
#
# Implementation of basic filters.
#
#------------------------------------------------------------------------------
#
# This file is part of the `LocalFilters.jl` package licensed under the MIT
# "Expat" License.
#
# Copyright (C) 2017-2018, Éric Thiébaut.
#

"""

    localmean(A, B)

yields the local mean of `A` in a neighborhood defined by `B`.  The result is
an array similar to `A`.

The in-place version is:

    localmean!(dst, A, B) -> dst

"""
localmean(A::AbstractArray, args...) = localmean!(similar(A), A, args...)

localmean!(dst, src::AbstractArray{T,N}, B=3) where {T,N} =
    localmean!(dst, src, convert(Neighborhood{N}, B))

@doc @doc(localmean) localmean!

function localmean!(dst::AbstractArray{T,N},
                    A::AbstractArray{T,N},
                    B::RectangularBox{N}) where {T,N}
    @assert size(dst) == size(A)
    localfilter!(dst, A, B,
                 (a)     -> (zero(T), 0),
                 (v,a,b) -> (v[1] + a, v[2] + 1),
                 (d,i,v) -> d[i] = v[1]/v[2])
end

function localmean!(dst::AbstractArray{T,N},
                    A::AbstractArray{T,N},
                    B::Kernel{Bool,N}) where {T,N}
    @assert size(dst) == size(A)
    localfilter!(dst, A, B,
                 (a)     -> (zero(T), 0),
                 (v,a,b) -> b ? (v[1] + a, v[2] + 1) : v,
                 (d,i,v) -> d[i] = v[1]/v[2])
end

function localmean!(dst::AbstractArray{T,N},
                    A::AbstractArray{T,N},
                    B::Kernel{T,N}) where {T,N}
    @assert size(dst) == size(A)
    localfilter!(dst, A, B,
                 (a)     -> (zero(T), zero(T)),
                 (v,a,b) -> (v[1] + a*b, v[2] + b),
                 (d,i,v) -> d[i] = v[1]/v[2])
end

"""

    convolve(A, B)

yields the convolution of `A` by the support of the neighborhood defined by
`B` of by the kernel `B` if it is an instance of `LocalFilters.Kernel`
with numerical coefficients.  The result is an array similar to `A`.

The in-place version is:

    convolve!(dst, A, B) -> dst

"""
convolve(A::AbstractArray, args...) = convolve!(similar(A), A, args...)

convolve!(dst, src::AbstractArray{T,N}, B=3) where {T,N} =
    convolve!(dst, src, convert(Neighborhood{N}, B))

@doc @doc(convolve) convolve!

function convolve!(dst::AbstractArray{S,N},
                   A::AbstractArray{T,N},
                   B::RectangularBox{N}) where {S,T,N}
    @assert size(dst) == size(A)
    localfilter!(dst, A, B,
                 (a)     -> zero(S),
                 (v,a,b) -> v + S(a),
                 (d,i,v) -> d[i] = v)
end

function convolve!(dst::AbstractArray{T,N},
                   A::AbstractArray{T,N},
                   B::Kernel{Bool,N}) where {T,N}
    @assert size(dst) == size(A)
    localfilter!(dst, A, B,
                 (a)     -> zero(T),
                 (v,a,b) -> b ? v + a : v,
                 (d,i,v) -> d[i] = v)
end

function convolve!(dst::AbstractArray{T,N},
                   A::AbstractArray{T,N},
                   B::Kernel{T,N}) where {T,N}
    @assert size(dst) == size(A)
    localfilter!(dst, A, B,
                 (a)     -> zero(T),
                 (v,a,b) -> v + a*b,
                 (d,i,v) -> d[i] = v)
end

#
# With `dst` the destination, `A` the source, and `B` the structuring
# element the pseudo-code to implement a local operation writes:
#
#     for i ∈ Sup(A)
#         v = initial(A[i])
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
#         v = initial(A[i])
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

"""
A local filtering operation can be performed by calling:

    localfilter!(dst, A, B, initial, update, store) -> dst

where `dst` is the destination, `A` is the source, `B` defines the
neighborhood, `initial`, `update` and `store` are three functions whose
purposes are explained by the following pseudo-code to implement the local
operation:

    for i ∈ Sup(A)
        v = initial(A[i])
        for j ∈ Sup(A) and i - j ∈ Sup(B)
            v = update(v, A[j], B[i-j])
        end
        store(dst, i, v)
    end

where `A` `Sup(A)` yields the support of `A` (that is the set of indices in
`A`) and likely `Sub(B)` for `B`.

For instance, to compute a local minimum (that is an erosion):

    localfilter!(dst, A, B,
                 (a)     -> typemax(T),
                 (v,a,b) -> min(v,a),
                 (d,i,v) -> d[i] = v)

**Important:** For efficiency reasons, the loop(s) in `localfilter!` are
perfomed without bound checking and it is the caller's responsability to insure
that the arguments have the correct sizes.

"""
function localfilter!(dst, A::AbstractArray{T,N}, B, initial::Function,
                      update::Function, store::Function) where {T,N}
    # Notes: The signature of this method is intentionally as little
    #        specialized as possible to avoid confusing the dispatcher.  The
    #        prupose of this method is just to convert `B ` into a neighborhood
    #        suitable for `A`.
    localfilter!(dst, A, convert(Neighborhood{N}, B), initial, update, store)
end

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
        v = initial(A[i])
        @simd for j in CartesianRange(max(imin, i - off), min(imax, i + off))
            v = update(v, A[j], true)
        end
        store(dst, i, v)
    end
    return dst
end

function localfilter!(dst,
                      A::AbstractArray{T,N},
                      B::CartesianBox{N},
                      initial::Function,
                      update::Function,
                      store::Function) where {T,N}
    R = CartesianRange(size(A))
    imin, imax = limits(R)
    kmin, kmax = limits(B)
    @inbounds for i in R
        v = initial(A[i])
        @simd for j in CartesianRange(max(imin, i - kmax), min(imax, i - kmin))
            v = update(v, A[j], true)
        end
        store(dst, i, v)
    end
    return dst
end

#
# Pseudo-code for a local operation on `A` in a neighborhood `B` is:
#
#     for i ∈ Sup(A)
#         v = initial(A[i])
#         for j ∈ Sup(A) and i - j ∈ Sup(B)
#             v = update(v, A[j], kernel[i-j+off])
#         end
#         dst[i] = final(v)
#     end
#
# where `off` is the anchor offset; the bounds for `j` are:
#
#    imin ≤ j ≤ imax   and   kmin ≤ i - j ≤ kmax
#
# where `imin` and `imax` are the bounds for `A` while `kmin` and `kmax` are
# the bounds for `B`.  The above constraints are identical to:
#
#    max(imin, i - kmax) ≤ j ≤ min(imax, i - kmin)
#

function localfilter!(dst,
                      A::AbstractArray{T,N},
                      B::Kernel{K,N},
                      initial::Function,
                      update::Function,
                      store::Function) where {T,K,N}
    R = CartesianRange(size(A))
    imin, imax = limits(R)
    kmin, kmax = limits(B)
    ker, off = coefs(B), anchor(B)
    @inbounds for i in R
        v = initial(A[i])
        k = i + off
        @simd for j in CartesianRange(max(imin, i - kmax), min(imax, i - kmin))
            v = update(v, A[j], ker[k-j])
        end
        store(dst, i, v)
    end
    return dst
end
