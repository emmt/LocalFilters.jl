#
# bilateral.jl --
#
# Implements the bilateral filter.
#
#-----------------------------------------------------------------------------------------
#
# This file is part of the `LocalFilters.jl` package licensed under the MIT "Expat"
# License.
#
# Copyright (c) 2018-2025, Éric Thiébaut.
#

module BilateralFilter

using ..LocalFilters
using ..LocalFilters:
    Box, FilterOrdering, Indices, BoundaryConditions, FlatBoundaries,
    ranges, Window, localindices, kernel, nearest

import ..LocalFilters: bilateralfilter!, bilateralfilter

using OffsetArrays, TypeUtils
using Base: @propagate_inbounds

"""
    LocalFilters.BilateralFilter.GaussianWindow{T}(σ) -> f

yields a callable object `f` with the shape of a Gaussian of standard deviation `σ` but
with a peak value of one, i.e. `f(0) -> 1`. The functor `f` can be applied to a single
value, to 2 values, say `x` and `y`, to yield `f(y - x)`, or to a coordinate difference
expressed as a Cartesian index.

Type parameter `T` is to specify the numerical type of the parameter of the callable
object. It can be omitted if `σ` is real.

"""
struct GaussianWindow{T<:AbstractFloat,S} <: AbstractTypeStableFunction{T}
    η::T # η = -1/2σ² is the factor in the exponential
    σ::S # keep in original precision for lossless conversion
    GaussianWindow{T}(σ::S) where {T,S} = new{T,S}(-1/(2*σ^2), σ)
end

GaussianWindow(σ::Real) = GaussianWindow{float(typeof(σ))}(σ)
GaussianWindow(f::GaussianWindow) = f
GaussianWindow{T}(f::GaussianWindow{T}) where {T} = f
GaussianWindow{T}(f::GaussianWindow) where {T} = GaussianWindow{T}(f.σ)

AbstractTypeStableFunction{T}(f::GaussianWindow) where {T} = GaussianWindow{T}(f)

(f::GaussianWindow{T})(I::CartesianIndex{N}) where {T,N} =
    exp(sum((k) -> I[k]^2, 1:N)*f.η) # FIXME: use @generated

(f::GaussianWindow)(x::T, y::T) where {T} = f(y - x)
(f::GaussianWindow{T})(x) where {T} = exp(convert(T,x)^2*f.η)

# Default window width for a Gaussian distance_filter with given standard
# deviation σ.
default_width(σ::Real) = 2*round(Int, 3σ) + 1

# Extend convert method.
Base.convert(::Type{T}, f::T) where {T<:GaussianWindow} = f
Base.convert(::Type{GaussianWindow{T}}, f) where {T} = GaussianWindow{T}(f)

"""
    bilateralfilter([T=float(eltype(A)),] A, F, [ord=FORWARD_FILTER,] G...=3)

yields the result of applying the bilateral filter on array `A`.

Argument `F` specifies how to smooth the differences in values. It can be:

- a function, say `f`, which is called as `f(A[i],A[j])` to yield a nonnegative weight for
  `i` the central index and `j` the index in a nearby position;

- a positive real, say `σ`, which is assumed to be the standard deviation of a Gaussian.

Arguments `G, ...` specify the settings of the distance filter for smoothing differences
in coordinates. There are several possibilities:

- `G... = wgt` an array of nonnegative weights or of Booleans. The axes of `wgt` must have
  offsets so that the zero index is part of the indices of `wgt`.

- `G... = f, w` with `f` a function and `w` any kind of argument that can be used to build
  a window `win` specifying the extension of the neighborhood. The value of the distance
  filter will be `max(f(i),0)` for all Cartesian index `i` of `win` such that `win[i]` is
  true. See [`kernel`](@ref) for the different ways to specify a window.

- `G... = σ` or , `G... = σ, w` with `σ` a positive real assumed to be the standard
  deviation of a Gaussian function and `w` any kind of argument that can be used to build
  a window `win` specifying the extension of the neighborhood. If `w` is not specified, a
  default window of size `±3σ` is assumed.

Optional argument `T` can be used to force the element type of the result. This argument
is needed if the element type of `A` is not a real.

See [`bilateralfilter!`](@ref) for an in-place version of this function and see
[Wikipedia](https://en.wikipedia.org/wiki/Bilateral_filter) for a description of the
bilateral filter.

"""
bilateralfilter(A::AbstractArray{<:Real}, args...) =
    # Provide type for computations and result.
    bilateralfilter(float(eltype(A)), A, args...)

bilateralfilter(T::Type, A::AbstractArray, args...) =
    bilateralfilter!(similar(A, T), A, args...)

"""
    bilateralfilter!(dst, A, F, [ord=FORWARD_FILTER,] G...) -> dst

overwrites `dst` with the result of applying the bilateral filter on array `A` and returns
`dst`.

See [`bilateralfilter`](@ref) for a description of the other arguments than `dst` and see
[Wikipedia](https://en.wikipedia.org/wiki/Bilateral_filter) for a description of the
bilateral filter.

""" bilateralfilter!

# Provide ordering to use with the distance filter.
function bilateralfilter!(dst::AbstractArray{<:Any,N},
                          A::AbstractArray{<:Any,N}, F, G...) where {N}
    return bilateralfilter!(dst, A, F, FORWARD_FILTER, G...)
end

# Provide the value and distance filters.
function bilateralfilter!(dst::AbstractArray{<:Any,N},
                          A::AbstractArray{<:Any,N}, F,
                          ord::FilterOrdering, G...) where {N}
    # Get the (unconverted) type returned by the value filter.
    Tf = value_filter_type(eltype(A), F)

    # Get the (unconverted) element type of the distance filter and simplify trailing
    # arguments.
    Tg, Gp = distance_filter(Dims{N}, G...)

    # Determine the resulting weights type.
    Tw = weight_type(Tf, Tg)
    F2 = value_filter(Tw, F)

    # Call the main function with filters of suitable types.
    return bilateralfilter!(dst, A, value_filter(Tw, F),
                            ord, distance_filter(Tw, Gp))
end

# Yield the type returned by default by the value filter (at least single precision
# floating-point for a Gaussian window).
value_filter_type(T::Type, f::Function) = Base.promote_op(f, T)
function value_filter_type(T::Type, σ::Real)
    (isfinite(σ) && σ > 0) || throw(ArgumentError(
        "standard deviation of the value filter must be finite and positive"))
    return promote_type(real(T), Float32)
end

# First pass to determine the distance filter. Yield a 2-tuple: an element type and
# anything else that may be used in the second pass to effectively build the filter as an
# array.
distance_filter(::Type{Dims{N}}, wgt::AbstractArray{T,N}) where {T,N} = (T, wgt)
distance_filter(::Type{Dims{N}}, win::Window{N}) where {N} =
    (Bool, kernel(Dims{N}, win))
function distance_filter(::Type{Dims{N}}, σ::Real = 3,
                         win::Window{N} = default_width(σ)) where {N}
    (isfinite(σ) && σ > 0) || throw(ArgumentError(
        "standard deviation of the distance filter must be finite and positive"))
    return (Float32, (σ,  kernel(Dims{N}, win)))
end
distance_filter(::Type{Dims{N}}, f::Function, win::Window{N}) where {N} =
    (Base.promote_op(f, CartesianIndex{N}), (f, kernel(Dims{N}, win)))

# Yield the type of the weights given that of the value and distance filters.
weight_type(Tf::Type{<:Real}, Tg::Type{Bool}) = Tf
weight_type(Tf::Type{<:Real}, Tg::Type{<:Real}) = promote_type(Tf, Tg)
weight_type(Tf::Type{<:Complex}, Tg::Type) =
    throw(ArgumentError("value filter must not yield complex type"))

# Yield a callable object to be used as the *value filter* in the bilateral filter.
value_filter(T::Type, f::Function) = AbstractTypeStableFunction{T}(f)
value_filter(T::Type, σ::Real) = GaussianWindow{T}(σ)

# Second pass to build/convert the distance filter. Argument T is the type of the weights.
distance_filter(::Type{T}, win::AbstractArray{Bool}) where {T} = win
distance_filter(::Type{T}, wgt::AbstractArray{T}) where {T} = wgt
distance_filter(::Type{T}, wgt::AbstractArray) where {T} = AbstractArray{T}(wgt)
function distance_filter(::Type{T}, G::Tuple{Function,
                                             AbstractArray{Bool}}) where {T}
    f, win = G
    return distance_filter!(
        OffsetArray(Array{T}(undef, size(win)), axes(win)),
        AbstractTypeStableFunction{T}(f), win)
end
function distance_filter(::Type{T}, G::Tuple{Real,
                                             AbstractArray{Bool}}) where {T}
    σ, win = G
    return distance_filter!(
        OffsetArray(Array{T}(undef, size(win)), axes(win)),
        GaussianWindow{T}(σ), win)
end
function distance_filter!(wgt::AbstractArray{T,N},
                          f::AbstractTypeStableFunction{T},
                          win::Box{N}) where {T,N}
    @inbounds for i in eachindex(IndexCartesian(), wgt, win)
        wgt[i] = max(f(i), zero(T))
    end
    return wgt
end
function distance_filter!(wgt::AbstractArray{T,N},
                          f::AbstractTypeStableFunction{T},
                          win::AbstractArray{Bool,N}) where {T,N}
    @inbounds for i in eachindex(IndexCartesian(), wgt, win)
        if win[i]
            wgt[i] = max(f(i), zero(T))
        else
            wgt[i] = zero(T)
        end
    end
    return wgt
end

# Distance filter is a simple sliding window.
function bilateralfilter!(dst::AbstractArray{<:Any,N},
                          A::AbstractArray{<:Any,N},
                          F::AbstractTypeStableFunction{T},
                          ord::FilterOrdering,
                          G::Box{N}) where {T,N}
    indices = Indices(dst, A, G)
    B = FlatBoundaries(indices(A))
    @inbounds for i in indices(dst)
        Ai = A[B(i)]
        den = zero(T)
        num = zero(promote_type(T, eltype(A)))
        @simd for j in localindices(indices(A), ord, indices(G), i)
            Aj = A[j]
            w = F(Ai, Aj)
            den += w
            num += w*Aj
        end
        if den > zero(den)
            dst[i] = nearest(eltype(dst), num/den)
        else
            dst[i] = nearest(eltype(dst), Ai)
        end
    end
    return dst
end

# Distance filter is an array of booleans.
function bilateralfilter!(dst::AbstractArray{<:Any,N},
                          A::AbstractArray{<:Any,N},
                          F::AbstractTypeStableFunction{T},
                          ord::FilterOrdering,
                          G::AbstractArray{Bool,N}) where {T,N}
    indices = Indices(dst, A, G)
    B = FlatBoundaries(indices(A))
    @inbounds for i in indices(dst)
        Ai = A[B(i)]
        den = zero(T)
        num = zero(promote_type(T, eltype(A)))
        for j in localindices(indices(A), ord, indices(G), i)
            if G[ord(i,j)]
                Aj = A[j]
                w = F(Ai, Aj)
                den += w
                num += w*Aj
            end
        end
        if den > zero(den)
            dst[i] = nearest(eltype(dst), num/den)
        else
            dst[i] = nearest(eltype(dst), Ai)
        end
    end
    return dst
end

# Distance filter is an array of weights.
function bilateralfilter!(dst::AbstractArray{<:Any,N},
                          A::AbstractArray{<:Any,N},
                          F::AbstractTypeStableFunction{T},
                          ord::FilterOrdering,
                          G::AbstractArray{T,N}) where {T,N}
    indices = Indices(dst, A, G)
    B = FlatBoundaries(indices(A))
    @inbounds for i in indices(dst)
        Ai = A[B(i)]
        den = zero(T)
        num = zero(promote_type(T, eltype(A)))
        @simd for j in localindices(indices(A), ord, indices(G), i)
            Aj = A[j]
            w = F(Ai, Aj)*G[ord(i,j)]
            den += w
            num += w*Aj
        end
        if den > zero(den)
            dst[i] = nearest(eltype(dst), num/den)
        else
            dst[i] = nearest(eltype(dst), Ai)
        end
    end
    return dst
end

end # module
