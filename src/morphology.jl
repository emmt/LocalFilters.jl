#
# morphology.jl --
#
# Implementation of non-linear morphological filters.
#
#------------------------------------------------------------------------------
#
# This file is part of the `LocalFilters.jl` package licensed under the MIT
# "Expat" License.
#
# Copyright (C) 2017-2018, Éric Thiébaut.
#

"""
Basic operations of mathematical morphology are:

```julia
erode(A, B) -> Amin
dilate(A, B) -> Amax
```

which respectively return the local minima `Amin` and the local maxima `Amax`
of argument `A` in a neighborhood defined by `B`.  The returned result is
similar to `A` (same size and type).

The two operations can be combined in one call:

```julia
localextrema(A, B) -> Amin, Amax
```

The in-place versions:

```julia
erode!(Amin, A, B) -> Amin
dilate!(Amax, A, B) -> Amax
localextrema!(Amin, Amax, A, B) -> Amin, Amax
```

apply the operation to `A` with structuring element `B` and store the
result in the provided arrays `Amin` and/or `Amax`.

See also [`localmean`](@ref), [`opening`](@ref), [`closing`](@ref),
[`top_hat`](@ref) and [`bottom_hat`](@ref).

"""
erode(A::AbstractArray, args...) = erode!(similar(A), A, args...)

erode!(dst, src::AbstractArray{T,N}, B=3) where {T,N} =
    erode!(dst, src, Neighborhood{N}(B))

@doc @doc(erode) erode!


function erode!(dst::AbstractArray{T,N},
                A::AbstractArray{T,N},
                B::RectangularBox{N}) where {T,N}
    localfilter!(dst, A, :, min, axes(B))
end

function erode!(A::AbstractArray{T,N},
                B::RectangularBox{N}) where {T,N}
    localfilter!(A, :, min, axes(B))
end

function erode!(dst::AbstractArray{T,N},
                A::AbstractArray{T,N},
                B::Kernel{Bool,N}) where {T,N}
    if all(identity, coefs(B))
        localfilter!(dst, A, :, min, axes(B))
    else
        @assert axes(dst) == axes(A)
        localfilter!(dst, A, B,
                     (a)     -> typemax(T),
                     (v,a,b) -> b && a < v ? a : v,
                     (d,i,v) -> d[i] = v)
    end
    return dst
end

function erode!(dst::AbstractArray{T,N},
                A::AbstractArray{T,N},
                B::Kernel{K,N}) where {T<:AbstractFloat,
                                       K<:AbstractFloat,N}
    if all(x -> x == zero(K), coefs(B))
        localfilter!(dst, A, :, min, axes(B))
    else
        @assert axes(dst) == axes(A)
        localfilter!(dst, A, Kernel{T}(B),
                     (a)     -> typemax(T),
                     (v,a,b) -> min(v, a - b),
                     (d,i,v) -> d[i] = v)
    end
    return dst
end

dilate(A::AbstractArray, args...) = dilate!(similar(A), A, args...)

@doc @doc(erode) dilate

dilate!(dst, src::AbstractArray{T,N}, B=3) where {T,N} =
    dilate!(dst, src, Neighborhood{N}(B))

@doc @doc(dilate) dilate!

function dilate!(dst::AbstractArray{T,N},
                 A::AbstractArray{T,N},
                 B::RectangularBox{N}) where {T,N}
    localfilter!(dst, A, :, max, axes(B))
end

function dilate!(A::AbstractArray{T,N},
                 B::RectangularBox{N}) where {T,N}
    localfilter!(A, :, max, axes(B))
end

function dilate!(dst::AbstractArray{T,N},
                 A::AbstractArray{T,N},
                 B::Kernel{Bool,N}) where {T,N}
    if all(identity, coefs(B))
        localfilter!(dst, A, :, max, axes(B))
    else
        @assert axes(dst) == axes(A)
        localfilter!(dst, A, B,
                     (a)     -> typemin(T),
                     (v,a,b) -> b && a > v ? a : v,
                     (d,i,v) -> d[i] = v)
    end
    return dst
end

function dilate!(dst::AbstractArray{T,N},
                 A::AbstractArray{T,N},
                 B::Kernel{K,N}) where {T<:AbstractFloat,
                                        K<:AbstractFloat,N}
    if all(x -> x == zero(K), coefs(B))
        localfilter!(dst, A, :, max, axes(B))
    else
        @assert axes(dst) == axes(A)
        localfilter!(dst, A, Kernel{T}(B),
                     (a)     -> typemin(T),
                     (v,a,b) -> max(v, a + b),
                     (d,i,v) -> d[i] = v)
    end
    return dst
end

localextrema(A::AbstractArray, args...) =
    localextrema!(similar(A), similar(A), A, args...)

@doc @doc(erode) localextrema

function localextrema!(Amin::AbstractArray{T,N},
                       Amax::AbstractArray{T,N},
                       A::AbstractArray{T,N}, B=3) where {T,N}
    localextrema!(Amin, Amax, A, Neighborhood{N}(B))
end

@doc @doc(localextrema) localextrema!

function localextrema!(Amin::AbstractArray{T,N},
                       Amax::AbstractArray{T,N},
                       A::AbstractArray{T,N},
                       B::RectangularBox{N}) where {T,N}
    @assert axes(Amin) == axes(Amax) == axes(A)
    localfilter!((Amin, Amax), A, B,
                 (a)     -> (typemax(T),
                             typemin(T)),
                 (v,a,b) -> (min(v[1], a),
                             max(v[2], a)),
                 (d,i,v) -> (Amin[i], Amax[i]) = v)
end

function localextrema!(Amin::AbstractArray{T,N},
                       Amax::AbstractArray{T,N},
                       A::AbstractArray{T,N},
                       B::Kernel{Bool,N}) where {T,N}
    @assert axes(Amin) == axes(Amax) == axes(A)
    localfilter!((Amin, Amax), A, B,
                 (a)     -> (typemax(T),
                             typemin(T)),
                 (v,a,b) -> (b && a < v[1] ? a : v[1],
                             b && a > v[2] ? a : v[2]),
                 (d,i,v) -> (Amin[i], Amax[i]) = v)
end

function localextrema!(Amin::AbstractArray{T,N},
                       Amax::AbstractArray{T,N},
                       A::AbstractArray{T,N},
                       B::Kernel{K,N}) where {T<:AbstractFloat,
                                              K<:AbstractFloat,N}
    @assert axes(Amin) == axes(Amax) == axes(A)
    localfilter!((Amin, Amax), A, Kernel{T}(B),
                 (a)     -> (typemax(T),
                             typemin(T)),
                 (v,a,b) -> (min(v[1], a - b),
                             max(v[2], a + b)),
                 (d,i,v) -> (Amin[i], Amax[i]) = v)
end

#------------------------------------------------------------------------------
# Higher level operators.

"""
```julia
closing(A, R)
opening(A, R)
```

respectively perform a closing or an opening of array `A` by the structuring
element `R`.  If not specified, `R` is a box of size 3 along all the dimensions
of `A`.  A closing is a dilation followed by an erosion, whereas an opening is
an erosion followed by a dilation.

The in-place versions are:

```julia
closing!(dst, wrk, src, R)
opening!(dst, wrk, src, R)
```

which perform the operation on the source `src` and store the result in
destination `dst` using `wrk` as a workspace array.  These 3 arguments must be
similar arrays, `dst` and `src` may be identical, but `wrk` must not be the
same array as `src` or `dst`.  The destination `dst` is returned.

See [`erode`](@ref) or [`dilate`](@ref) for the meaning of the arguments.

"""
closing(A::AbstractArray, args...) =
    closing!(similar(A), similar(A), A, args...)

function closing!(dst::AbstractArray{T,N},
                  wrk::AbstractArray{T,N},
                  src::AbstractArray{T,N}, B=3) where {T,N}
    closing!(dst, wrk, src, Neighborhood{N}(B))
end

function closing!(dst::AbstractArray{T,N},
                  wrk::AbstractArray{T,N},
                  src::AbstractArray{T,N},
                  B::Neighborhood{N}) where {T,N}
    erode!(dst, dilate!(wrk, src, B), B)
end

opening(A::AbstractArray, args...) =
    opening!(similar(A), similar(A), A, args...)

function opening!(dst::AbstractArray{T,N},
                  wrk::AbstractArray{T,N},
                  src::AbstractArray{T,N}, B=3) where {T,N}
    opening!(dst, wrk, src, Neighborhood{N}(B))
end

function opening!(dst::AbstractArray{T,N},
                  wrk::AbstractArray{T,N},
                  src::AbstractArray{T,N},
                  B::Neighborhood{N}) where {T,N}
    dilate!(dst, erode!(wrk, src, B), B)
end

@doc @doc(closing) closing!
@doc @doc(closing) opening
@doc @doc(closing) opening!

# Out-of-place top hat filter requires 2 allocations without a
# pre-filtering, 3 allocations with a pre-filtering.

"""
```julia
top_hat(A, R)
top_hat(A, R, S)
bottom_hat(A, R)
bottom_hat(A, R, S)
```

Perform A summit/valley detection by applying a top-hat filter to array
`A`.  Argument `R` defines the structuring element for the feature
detection.  Optional argument `S` specifies the structuring element used to
apply a smoothing to `A` prior to the top-hat filter.  If `R` and `S` are
specified as the radii of the structuring elements, then `S` should be
smaller than `R`.  For instance:

```julia
top_hat(bitmap, 3, 1)
```

may be used to detect text or lines in a bimap image.

The in-place versions:

```julia
top_hat!(dst, wrk, src, R)
bottom_hat!(dst, wrk, src, R)
```

apply the top-hat filter on the source `src` and store the result in the
destination `dst` using `wrk` as a workspace array.  These 3 arguments must
be similar but different arrays.  The destination `dst` is returned.

See also [`dilate`](@ref), [`closing`](@ref) and [`morph_enhance`](@ref).

"""
top_hat(a, r=3) = top_hat!(similar(a), similar(a), a, r)

function top_hat(a, r, s)
    wrk = similar(a)
    top_hat!(similar(a), wrk, closing!(similar(a), wrk, a, s), r)
end

function top_hat!(dst::AbstractArray{T,N},
                  wrk::AbstractArray{T,N},
                  src::AbstractArray{T,N}, r=3) where {T,N}
    opening!(dst, wrk, src, r)
    @inbounds for i in eachindex(dst, src)
        dst[i] = src[i] - dst[i]
    end
    return dst
end

bottom_hat(a, r=3) = bottom_hat!(similar(a), similar(a), a, r)

function bottom_hat(a, r, s)
    wrk = similar(a)
    bottom_hat!(similar(a), wrk, opening!(similar(a), wrk, a, s), r)
end

function bottom_hat!(dst::AbstractArray{T,N},
                     wrk::AbstractArray{T,N},
                     src::AbstractArray{T,N}, r=3) where {T,N}
    closing!(dst, wrk, src, r)
    @inbounds for i in eachindex(dst, src)
        dst[i] -= src[i]
    end
    return dst
end

@doc @doc(top_hat)    top_hat!
@doc @doc(top_hat) bottom_hat
@doc @doc(top_hat) bottom_hat!
