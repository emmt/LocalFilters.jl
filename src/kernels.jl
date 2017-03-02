# kernels.jl --
#
# Implementation of local operations with a general purpose kernel which
# is a rectangular array of coefficients over a, possibly off-centered,
# rectangular neighborhood.

"""
A kernel can be used to define a versatile type of structuring elements.
"""
immutable Kernel{T,N} <: Neighborhood{N}
    coefs::Array{T,N}
    anchor::CartesianIndex{N}
    Kernel{T,N}(coefs::Array{T,N}, anchor::CartesianIndex{N}) =
        new(coefs, anchor)
end

# The index in the array of kernel coefficients is `k + anchor` hence:
#
#     1 ≤ k + anchor ≤ dim
#     1 - anchor ≤ k ≤ dim - anchor
#
# thus `first = 1 - anchor` and `last = dim - anchor`.

length(B::Kernel) = length(coefs(B))
size{N}(B::Kernel{N}) = size(coefs(B))
size(B::Kernel, i) = size(coefs(B), i)
first(B::Kernel) = (I = anchor(B); one(I) - I)
last(B::Kernel) = CartesianIndex(size(coefs(B))) - anchor(B)
getindex(B::Kernel, I::CartesianIndex) = getindex(coefs(B), I + anchor(B))
getindex(B::Kernel, inds::Integer...) = getindex(B, CartesianIndex(inds))

coefs(B::Kernel) = B.coefs
anchor(B::Kernel) = B.anchor

Kernel(B::CartesianBox) = Kernel(ones(Bool, size(B)), anchor(B))
Kernel(B::CenteredBox) = Kernel(ones(Bool, size(B)))

# Wrap an array into a kernel (call copy if you do not want to share).
function Kernel{T,N}(arr::Array{T,N},
                     off::CartesianIndex{N}=anchor(arr))
    Kernel{T,N}(arr, off)
end

function Kernel{T,N}(arr::AbstractArray{T,N},
                     off::CartesianIndex{N}=anchor(arr))
    Kernel{T,N}(copy!(Array(T, size(arr)), arr), off)
end

function Kernel{T,N}(::Type{T},
                     arr::AbstractArray{T,N},
                     off::CartesianIndex{N}=anchor(arr))
    Kernel(arr, off)
end

function Kernel{T,N}(tup::Tuple{T,T},
                     msk::AbstractArray{Bool,N},
                     off::CartesianIndex{N}=anchor(msk))
    arr = Array(T, size(msk))
    vtrue, vfalse = tup[1], tup[2]
    @inbounds for i in eachindex(arr, msk)
        arr[i] = msk[i] ? vtrue : vfalse
    end
    Kernel{T,N}(arr, off)
end

Kernel{T,N}(tup::Tuple{T,T}, B::Kernel{Bool,N}) =
    Kernel(tup, coefs(B), anchor(B))

# Make a flat structuring element from a boolean kernel.
function Kernel{T<:AbstractFloat,N}(::Type{T},
                                    msk::AbstractArray{Bool,N},
                                    off::CartesianIndex{N}=anchor(msk))
    Kernel((zero(T), -T(Inf)), msk, off)
end

Kernel{T,N}(::Type{T}, B::Kernel{Bool,N}) =
    Kernel(T, coefs(B), anchor(B))


Kernel{T<:AbstractFloat,N}(::Type{T}, B::Kernel{Bool,N}) =
    Kernel(T, coefs(B), anchor(B))

Kernel{T1,T2,N}(::Type{T1}, msk::AbstractArray{T2,N}) =
    Kernel(T1, msk, anchor(msk))

Kernel(B::Kernel) = B

Kernel{N}(::Type{Bool}, B::Kernel{Bool,N}) = B

function strictfloor{T}(::Type{T}, x)
    n = floor(T, x)
    (n < x ? n : n - one(T)) :: T
end

function ball(rank::Integer, radius::Real)
    b = radius + 1/2
    r = strictfloor(Int, b)
    dim = 2*r + 1
    dims = ntuple(d->dim, rank)
    arr = Array(Bool, dims)
    qmax = strictfloor(Int, b^2)
    _ball!(arr, 0, qmax, r, 1:dim, tail(dims))
    arr
end

@inline function _ball!{N}(arr::AbstractArray{Bool,N},
                           q::Int, qmax::Int, r::Int,
                           range::UnitRange{Int},
                           dims::Tuple{Int}, I::Int...)
    nextdims = tail(dims)
    x = -r
    for i in range
        _ball!(arr, q + x*x, qmax, r, range, nextdims, I..., i)
        x += 1
    end
end

@inline function _ball!{N}(arr::AbstractArray{Bool,N},
                           q::Int, qmax::Int, r::Int,
                           range::UnitRange{Int},
                           ::Tuple{}, I::Int...)
    x = -r
    for i in range
        arr[I...,i] = (q + x*x ≤ qmax)
        x += 1
    end
end

#
# Pseudo-code for a local operation on `A` in a neighborhood `B` is:
#
#     for i ∈ Sup(A)
#         v = initial()
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

function localfilter!{T1,T2,T3,N}(dst::AbstractArray{T1,N},
                                  A::AbstractArray{T2,N}, B::Kernel{T3,N},
                                  initial::Function, update::Function,
                                  final::Function)
    @assert size(dst) == size(A)
    R = CartesianRange(size(A))
    imin, imax = limits(R)
    kmin, kmax = limits(B)
    ker, off = coefs(B), anchor(B)
    @inbounds for i in R
        v = initial()
        k = i + off
        for j in CartesianRange(max(imin, i - kmax), min(imax, i - kmin))
            v = update(v, A[j], ker[k-j])
        end
        dst[i] = final(v)
    end
    return dst
end

function localmean!{T,N}(dst::AbstractArray{T,N}, A::AbstractArray{T,N},
                         B::Kernel{Bool,N})
    localfilter!(dst, A, B,
                 ()      -> (zero(T), 0),
                 (v,a,b) -> b ? (v[1] + a, v[2] + 1) : v,
                 (v)     -> v[1]/v[2])
end

function erode!{T,N}(dst::AbstractArray{T,N}, A::AbstractArray{T,N},
                     B::Kernel{Bool,N})
    localfilter!(dst, A, B,
                 ()      -> typemax(T),
                 (v,a,b) -> b && a < v ? a : v,
                 (v)     -> v)
end

function dilate!{T,N}(dst::AbstractArray{T,N}, A::AbstractArray{T,N},
                      B::Kernel{Bool,N})
    localfilter!(dst, A, B,
                 ()      -> typemin(T),
                 (v,a,b) -> b && a > v ? a : v,
                 (v)     -> v)
end

function localextrema!{T,N}(Amin::AbstractArray{T,N},
                            Amax::AbstractArray{T,N},
                            A::AbstractArray{T,N},
                            B::Kernel{Bool,N})
    @assert size(Amin) == size(Amax) == size(A)
    R = CartesianRange(size(A))
    imin, imax = limits(R)
    kmin, kmax = limits(B)
    tmin, tmax = limits(T)
    @inbounds for i in R
        vmin, vmax = tmax, tmin
        k = i + off
        for j in CartesianRange(max(imin, i - kmax), min(imax, i - kmin))
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

# Erosion and dilation with a shaped structuring element
# (FIXME: for integers satured addition/subtraction would be needed)

function localmean!{T,N}(dst::AbstractArray{T,N}, A::AbstractArray{T,N},
                         B::Kernel{T,N})
    localfilter!(dst, A, B,
                 ()      -> (zero(T), zero(T)),
                 (v,a,b) -> (v[1] + a*b, v[2] + b),
                 (v)     -> v[1]/v[2])
end

function erode!{T,N}(dst::AbstractArray{T,N}, A::AbstractArray{T,N},
                     B::Kernel{T,N})
    localfilter!(dst, A, B,
                 ()      -> typemax(T),
                 (v,a,b) -> min(v, a - b),
                 (v)     -> v)
end

function dilate!{T,N}(dst::AbstractArray{T,N}, A::AbstractArray{T,N},
                      B::Kernel{T,N})
    localfilter!(dst, A, B,
                 ()      -> typemin(T),
                 (v,a,b) -> max(v, a + b),
                 (v)     -> v)
end

function localextrema!{T<:AbstractFloat,N}(Amin::AbstractArray{T,N},
                                           Amax::AbstractArray{T,N},
                                           A::AbstractArray{T,N},
                                           B::Kernel{T,N})
    @assert size(Amin) == size(Amax) == size(A)
    R = CartesianRange(size(A))
    imin, imax = limits(R)
    kmin, kmax = limits(B)
    tmin, tmax = limits(T)
    @inbounds for i in R
        vmin, vmax = tmax, tmin
        k = i + off
        for j in CartesianRange(max(imin, i - kmax), min(imax, i - kmin))
            vmin = min(vmin, A[j] - ker[k-j])
            vmax = max(vmax, A[j] + ker[k-j])
        end
        Amin[i] = vmin
        Amax[i] = vmax
    end
    return Amin, Amax
end

function convolve!{T<:AbstractFloat,N}(dst::AbstractArray{T,N},
                                       A::AbstractArray{T,N},
                                       B::Kernel{T,N})
    @assert size(dst) == size(A)
    R = CartesianRange(size(A))
    imin, imax = limits(R)
    kmin, kmax = limits(B)
    ker, off = coefs(B), anchor(B)
    @inbounds for i in R
        v = zero(T)
        k = i + off
        for j in CartesianRange(max(imin, i - kmax), min(imax, i - kmin))
            v += A[j]*ker[k-j]
        end
        dst[i] = v
    end
    return dst
end
