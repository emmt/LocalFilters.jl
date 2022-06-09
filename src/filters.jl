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
```julia
localmean(A, B)
```

yields the local mean of `A` in a neighborhood defined by `B`.  The result is
an array similar to `A`.

The in-place version is:

```julia
localmean!(dst, A, B) -> dst
```

See also [`localfilter!`](@ref).

"""
localmean(A::AbstractArray{T,N}, B=3) where {T,N} =
    localmean(A, Neighborhood{N}(B))

localmean(A::AbstractArray{T}, B::RectangularBox) where {T} =
    localmean!(similar(Array{float(T)}, axes(A)), A, B)

localmean(A::AbstractArray{T}, B::Kernel{Bool}) where {T} =
    localmean!(similar(Array{float(T)}, axes(A)), A, B)

localmean(A::AbstractArray{T}, B::Kernel{K}) where {T,K} =
    localmean!(similar(Array{float(promote_type(T,K))}, axes(A)), A, B)

function localmean!(dst::AbstractArray{Td,N},
                    A::AbstractArray{Ts,N}, B=3) where {Td,Ts,N}
    localmean!(dst, A, Neighborhood{N}(B))
end

@doc @doc(localmean) localmean!

function localmean!(dst::AbstractArray{Td,N},
                    A::AbstractArray{Ts,N},
                    B::RectangularBox{N}) where {Td,Ts,N}
    @assert axes(dst) == axes(A)
    T = _typeofsum(Ts)
    localfilter!(dst, A, B,
                 (a)     -> (zero(T), 0),
                 (v,a,b) -> (v[1] + a, v[2] + 1),
                 (d,i,v) -> _store!(d, i, v[1]/v[2]))
end

function localmean!(dst::AbstractArray{Td,N},
                    A::AbstractArray{Ts,N},
                    B::Kernel{Bool,N}) where {Td,Ts,N}
    @assert axes(dst) == axes(A)
    T = _typeofsum(Ts)
    localfilter!(dst, A, B,
                 (a)     -> (zero(T), 0),
                 (v,a,b) -> b ? (v[1] + a, v[2] + 1) : v,
                 (d,i,v) -> _store!(d, i, v[1]/v[2]))
end

function localmean!(dst::AbstractArray{Td,N},
                    A::AbstractArray{Ts,N},
                    B::Kernel{Tk,N}) where {Td,Ts,Tk,N}
    @assert axes(dst) == axes(A)
    T = _typeofsum(promote_type(Ts, Tk))
    localfilter!(dst, A, B,
                 (a)     -> (zero(T), zero(T)),
                 (v,a,b) -> (v[1] + a*b, v[2] + b),
                 (d,i,v) -> _store!(d, i, v[1]/v[2]))
end

"""

`_store!(arr, idx, val)` stores value `val` in array `arr` at index `idx`,
taking care of rounding `val` if it is of floating-point type while the
elements of `arr` are integers.

"""
@inline function _store!(arr::AbstractArray{T}, idx,
                         val::AbstractFloat) where {T<:Integer}
    arr[idx] = round(T, val)
end

@inline function _store!(arr::AbstractArray, idx, val)
    arr[idx] = val
end


"""

`_typeofsum(T)` yields a numerical type suitable for storing a sum of elements
of type `T`.

"""
_typeofsum(::Type{T}) where {T} = T
_typeofsum(::Type{T}) where {T<:Integer} =
    (sizeof(T) < sizeof(Int) ? widen(T) : T)

"""
```julia
convolve(A, B)
```

yields the convolution of `A` by the support of the neighborhood defined by
`B` of by the kernel `B` if it is an instance of `LocalFilters.Kernel` with
numerical coefficients.  The result is an array similar to `A`.

The in-place version is:

```julia
convolve!(dst, A, B) -> dst
```

See also [`localfilter!`](@ref).

"""
convolve(A::AbstractArray{T,N}, B=3) where {T,N} =
    convolve(A, Neighborhood{N}(B))

convolve(A::AbstractArray{T}, B::RectangularBox) where {T} =
    convolve!(similar(Array{_typeofsum(T)}, axes(A)), A, B)

convolve(A::AbstractArray{T}, B::Kernel{Bool}) where {T} =
    convolve!(similar(Array{_typeofsum(T)}, axes(A)), A, B)

convolve(A::AbstractArray{T}, B::Kernel{K}) where {T,K} =
    convolve!(similar(Array{_typeofsum(promote_type(T,K))}, axes(A)), A, B)

function convolve!(dst::AbstractArray{Td,N},
                   A::AbstractArray{Ts,N}, B=3) where {Td,Ts,N}
    convolve!(dst, A, Neighborhood{N}(B))
end

@doc @doc(convolve) convolve!

function convolve!(dst::AbstractArray{Td,N},
                   A::AbstractArray{Ts,N},
                   B::RectangularBox{N}) where {Td,Ts,N}
    @assert axes(dst) == axes(A)
    T = _typeofsum(Ts)
    localfilter!(dst, A, B,
                 (a)     -> zero(T),
                 (v,a,b) -> v + a,
                 _store!)
end

function convolve!(dst::AbstractArray{Td,N},
                   A::AbstractArray{Ts,N},
                   B::Kernel{Bool,N}) where {Td,Ts,N}
    @assert axes(dst) == axes(A)
    T = _typeofsum(Ts)
    localfilter!(dst, A, B,
                 (a)     -> zero(T),
                 (v,a,b) -> b ? v + a : v,
                 _store!)
end

function convolve!(dst::AbstractArray{Td,N},
                   A::AbstractArray{Ts,N},
                   B::Kernel{Tk,N}) where {Td,Ts,Tk,N}
    @assert axes(dst) == axes(A)
    T = _typeofsum(promote_type(Ts, Tk))
    localfilter!(dst, A, B,
                 (a)     -> zero(T),
                 (v,a,b) -> v + a*b,
                 _store!)
end

"""
# General local filters

A local filtering operation can be performed by calling:

```julia
localfilter!(dst, A, B, initial, update, store) -> dst
```

where `dst` is the destination, `A` is the source, `B` defines the
neighborhood, `initial`, `update` and `store` are three functions whose
purposes are explained by the following pseudo-code to implement the local
operation:

```
for i ∈ Sup(A)
    v = initial(A[i])
    for j ∈ Sup(A) and i - j ∈ Sup(B)
        v = update(v, A[j], B[i-j])
    end
    store(dst, i, v)
end
```

where `A` `Sup(A)` yields the support of `A` (that is the set of indices in
`A`) and likely `Sup(B)` for `B`.

For instance, to compute a local minimum (that is, an *erosion*):

```julia
localfilter!(dst, A, B,
             (a)     -> typemax(T),
             (v,a,b) -> min(v,a),
             (d,i,v) -> d[i] = v)
```

**Important:** Because the result of an elementary operation can be something
else than just a scalar, the loop(s) in `localfilter!` are performed without
bound checking of the destination and it is the caller's responsability to
insure that the destination have the correct sizes.

"""
function localfilter!(dst, A::AbstractArray{T,N}, B, initial::Function,
                      update::Function, store::Function) where {T,N}
    # Notes: The signature of this method is intentionally as little
    #        specialized as possible to avoid confusing the dispatcher.  The
    #        purpose of this method is just to convert `B ` into a neighborhood
    #        suitable for `A`.
    localfilter!(dst, A, Neighborhood{N}(B), initial, update, store)
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
# `A`) and likely `Sup(B)` for `B`.  Note that, in this example, destination
# `dst` and source `A` must have the same support.
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
# where `imin = first_cartesian_index(A)` and `imax = last_cartesian_index(A)` are the bounds for
# `A` while `kmin = first_cartesian_index(B)` and `kmax = last_cartesian_index(B)` are the bounds
# for `B`.  The above constraints are identical to:
#
#    max(imin, i - kmax) ≤ j ≤ min(imax, i - kmin)
#
function localfilter!(dst,
                      A::AbstractArray{T,N},
                      B::RectangularBox{N},
                      initial::Function,
                      update::Function,
                      store::Function) where {T,N}
    R = cartesian_region(A)
    imin, imax = limits(R)
    kmin, kmax = limits(B)
    @inbounds for i in R
        v = initial(A[i])
        @simd for j in cartesian_region(max(imin, i - kmax),
                                       min(imax, i - kmin))
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
# where `off` is the offset; the bounds for `j` are:
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
    R = cartesian_region(A)
    imin, imax = limits(R)
    kmin, kmax = limits(B)
    ker, off = coefs(B), offset(B)
    @inbounds for i in R
        v = initial(A[i])
        k = i + off
        @simd for j in cartesian_region(max(imin, i - kmax),
                                       min(imax, i - kmin))
            v = update(v, A[j], ker[k-j])
        end
        store(dst, i, v)
    end
    return dst
end
