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

    erode(A, B) -> Amin
    dilate(A, B) -> Amax

which return the local minima `Amin` and the local maxima `Amax` of argument
`A` in a neighborhood defined by `B`.  The returned result is similar to `A`
(same size and type).

The two operations can be combined in one call:

    localextrema(A, B) -> Amin, Amax

The in-place versions:

    erode!(Amin, A, B) -> Amin
    dilate!(Amax, A, B) -> Amax
    localextrema!(Amin, Amax, A, B) -> Amin, Amax

apply the operation to `A` with structuring element `B` and store the
result in the provided arrays `Amin` and/or `Amax`.


## See also:
localmean, opening, closing, top_hat, bottom_hat

"""
erode(A::AbstractArray, args...) = erode!(similar(A), A, args...)

erode!(dst, src::AbstractArray{T,N}, B=3) where {T,N} =
    erode!(dst, src, convert(Neighborhood{N}, B))

@doc @doc(erode) erode!

function erode!(dst::AbstractArray{T,N},
                A::AbstractArray{T,N},
                B::RectangularBox{N}) where {T,N}
    @assert size(dst) == size(A)
    localfilter!(dst, A, B,
                 (a)     -> typemax(T),
                 (v,a,b) -> min(v, a),
                 (d,i,v) -> d[i] = v)
end

function erode!(dst::AbstractArray{T,N},
                A::AbstractArray{T,N},
                B::Kernel{Bool,N}) where {T,N}
    @assert size(dst) == size(A)
    localfilter!(dst, A, B,
                 (a)     -> typemax(T),
                 (v,a,b) -> b && a < v ? a : v,
                 (d,i,v) -> d[i] = v)
end

function erode!(dst::AbstractArray{T,N},
                A::AbstractArray{T,N},
                B::Kernel{T,N}) where {T,N}
    localfilter!(dst, A, B,
                 (a)     -> typemax(T),
                 (v,a,b) -> min(v, a - b),
                 (d,i,v) -> d[i] = v)
end

dilate(A::AbstractArray, args...) = dilate!(similar(A), A, args...)

@doc @doc(erode) dilate

dilate!(dst, src::AbstractArray{T,N}, B=3) where {T,N} =
    dilate!(dst, src, convert(Neighborhood{N}, B))

@doc @doc(dilate) dilate!

function dilate!(dst::AbstractArray{T,N},
                 A::AbstractArray{T,N},
                 B::RectangularBox{N}) where {T,N}
    @assert size(dst) == size(A)
    localfilter!(dst, A, B,
                 (a)     -> typemin(T),
                 (v,a,b) -> max(v, a),
                 (d,i,v) -> d[i] = v)
end

function dilate!(dst::AbstractArray{T,N},
                 A::AbstractArray{T,N},
                 B::Kernel{Bool,N}) where {T,N}
    @assert size(dst) == size(A)
    localfilter!(dst, A, B,
                 (a)     -> typemin(T),
                 (v,a,b) -> b && a > v ? a : v,
                 (d,i,v) -> d[i] = v)
end

function dilate!(dst::AbstractArray{T,N},
                 A::AbstractArray{T,N},
                 B::Kernel{T,N}) where {T,N}
    @assert size(dst) == size(A)
    localfilter!(dst, A, B,
                 (a)     -> typemin(T),
                 (v,a,b) -> max(v, a + b),
                 (d,i,v) -> d[i] = v)
end

localextrema(A::AbstractArray, args...) =
    localextrema!(similar(A), similar(A), A, args...)

@doc @doc(erode) localextrema

function localextrema!(Amin::AbstractArray{T,N},
                       Amax::AbstractArray{T,N},
                       A::AbstractArray{T,N}, B=3) where {T,N}
    localextrema!(Amin, Amax, A, convert(Neighborhood{N}, B))
end

@doc @doc(localextrema) localextrema!

function localextrema!(Amin::AbstractArray{T,N},
                       Amax::AbstractArray{T,N},
                       A::AbstractArray{T,N},
                       B::RectangularBox{N}) where {T,N}
    @assert size(Amin) == size(Amax) == size(A)
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
    @assert size(Amin) == size(Amax) == size(A)
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
                       B::Kernel{T,N}) where {T,N}
    @assert size(Amin) == size(Amax) == size(A)
    localfilter!((Amin, Amax), A, B,
                 (a)     -> (typemax(T),
                             typemin(T)),
                 (v,a,b) -> (min(v[1], a - b),
                             max(v[2], a + b)),
                 (d,i,v) -> (Amin[i], Amax[i]) = v)
end

#------------------------------------------------------------------------------
# Higher level operators.

"""

    closing(arr, r)
    opening(arr, r)

perform a closing or an opening of array `arr` by the structuring element
`r`.  If not specified, `r` is a box of size 3 along all the dimensions of
`arr`.  A closing is a dilation followed by an erosion, whereas an opening
is an erosion followed by a dilation.

The in-place versions are:

    closing!(dst, wrk, src, r)
    opening!(dst, wrk, src, r)

which perform the operation on the source `src` and store the result in
destination `dst` using `wrk` as a workspace array.  These 3 arguments must
be similar arrays, `dst` and `src` may be identical, but `wrk` must not be
the same array as `src` or `dst`.  The destination `dst` is returned.

See `erode` or `dilate` for the meaning of the arguments.

"""
closing(A::AbstractArray, args...) =
    closing!(similar(A), similar(A), A, args...)

function closing!(dst::AbstractArray{T,N},
                  wrk::AbstractArray{T,N},
                  src::AbstractArray{T,N}, B=3) where {T,N}
    closing!(dst, wrk, src, convert(Neighborhood{N}, B))
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
    opening!(dst, wrk, src, convert(Neighborhood{N}, B))
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

    top_hat(a, r)
    top_hat(a, r, s)
    bottom_hat(a, r)
    bottom_hat(a, r, s)

Perform a summit/valley detection by applying a top-hat filter to array
`a`.  Argument `r` defines the structuring element for the feature
detection.  Optional argument `s` specifies the structuring element used to
apply a smoothing to `a` prior to the top-hat filter.  If `r` and `s` are
specified as the radii of the structuring elements, then `s` should be
smaller than `r`.  For instance:

     top_hat(bitmap, 3, 1)

may be used to detect text or lines in a bimap image.

The in-place versions:

     top_hat!(dst, wrk, src, r)
     bottom_hat!(dst, wrk, src, r)

apply the top-hat filter on the source `src` and store the result in the
destination `dst` using `wrk` as a workspace array.  These 3 arguments must
be similar but different arrays.  The destination `dst` is returned.

See also: dilate, closing, morph_enhance.

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
