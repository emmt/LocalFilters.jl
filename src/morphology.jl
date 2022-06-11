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
# Copyright (C) 2017-2022, Éric Thiébaut.
#

"""
    erode(A, R=3) -> Amin

yields the array of local minima `Amin` of argument `A` with a structuring
element defined by `R`.  The returned result `Amin` is similar to `A` (same
size and type).

If the structuring element `R` is a simple hyperrectangular moving window, the
much faster van Herk-Gil-Werman algorithm is used.  If specified as an odd
integer (as is assumed by default), the structuring element is a
hyperrectangular moving window of size `R` along every dimension of `A`.

An erosion is one of the most basic operations of mathematical morphology.  See
[`erode!`](@ref) for an in-place version of the method, [`dilate`](@ref) for
retrieving the local maxima, and [`localextrema`](@ref) for performing an
erosion and a dilation in a single pass.

"""
erode(A::AbstractArray, args...) = erode!(similar(A), A, args...)

"""
    erode!(Amin, A, R=3) -> Amin

overwrites `Amin` with an erosion of the array `A` by the structuring element
defined by `R` and returns `Amin`.

If the structuring element `R` is a simple hyperrectangular moving window, the
much faster van Herk-Gil-Werman algorithm is used and the operation can be done
in-place.  That is `A` and `Amin` can be the same arrays.  In that case, the
following syntax is allowed:

    erode!(A, R=3) -> A

See [`erode`](@ref) for an out-of-place version and for more information.

"""
erode!(dst, A::AbstractArray{T,N}, R=3) where {T,N} =
    erode!(dst, A, Neighborhood{N}(R))

function erode!(dst::AbstractArray{T,N},
                A::AbstractArray{T,N},
                R::RectangularBox{N}) where {T,N}
    localfilter!(dst, A, :, min, axes(R))
end

function erode!(A::AbstractArray{T,N},
                R::RectangularBox{N}) where {T,N}
    localfilter!(A, :, min, axes(R))
end

function erode!(dst::AbstractArray{T,N},
                A::AbstractArray{T,N},
                R::Kernel{Bool,N}) where {T,N}
    if ismmbox(R)
        localfilter!(dst, A, :, min, axes(R))
    else
        check_indices(dst, A)
        localfilter!(dst, A, R,
                     (a)     -> typemax(T),
                     (v,a,b) -> b && a < v ? a : v,
                     (d,i,v) -> d[i] = v)
    end
    return dst
end

function erode!(dst::AbstractArray{T,N},
                A::AbstractArray{T,N},
                R::Kernel{K,N}) where {T<:AbstractFloat,
                                       K<:AbstractFloat,N}
    if ismmbox(R)
        localfilter!(dst, A, :, min, axes(R))
    else
        check_indices(dst, A)
        localfilter!(dst, A, Kernel{T}(R),
                     (a)     -> typemax(T),
                     (v,a,b) -> min(v, a - b),
                     (d,i,v) -> d[i] = v)
    end
    return dst
end

"""
    dilate(A, R=3) -> Amax

yields the array of local maxima `Amax` of argument `A` with a structuring
element defined by `R`.  The returned result `Amax` is similar to `A` (same
size and type).

If the structuring element `R` is a simple hyperrectangular moving window, the
much faster van Herk-Gil-Werman algorithm is used.  If specified as an odd
integer (as is assumed by default), the structuring element is a
hyperrectangular moving window of size `R` along every dimension of `A`.

A dilation is one of the most basic operations of mathematical morphology.  See
[`dilate!`](@ref) for an in-place version of the method, [`erode`](@ref) for
retrieving the local minima, and [`localextrema`](@ref) for performing an
erosion and a dilation in a single pass.

"""
dilate(A::AbstractArray, args...) = dilate!(similar(A), A, args...)

"""
    dilate!(Amax, A, R=3) -> Amax

overwrites `Amax` with a dilation of the array `A` by the structuring element
defined by `R` and returns `Amax`.

If the structuring element `R` is a simple hyperrectangular moving window, the
much faster van Herk-Gil-Werman algorithm is used and the operation can be done
in-place.  That is `A` and `Amin` can be the same arrays.  In that case, the
following syntax is allowed:

    dilate!(A, R=3) -> A

See [`dilate`](@ref) for an out-of-place version and for more information.

"""
dilate!(dst, A::AbstractArray{T,N}, R=3) where {T,N} =
    dilate!(dst, A, Neighborhood{N}(R))

function dilate!(dst::AbstractArray{T,N},
                 A::AbstractArray{T,N},
                 R::RectangularBox{N}) where {T,N}
    localfilter!(dst, A, :, max, axes(R))
end

function dilate!(A::AbstractArray{T,N},
                 R::RectangularBox{N}) where {T,N}
    localfilter!(A, :, max, axes(R))
end

function dilate!(dst::AbstractArray{T,N},
                 A::AbstractArray{T,N},
                 R::Kernel{Bool,N}) where {T,N}
    if ismmbox(R)
        localfilter!(dst, A, :, max, axes(R))
    else
        check_indices(dst, A)
        localfilter!(dst, A, R,
                     (a)     -> typemin(T),
                     (v,a,b) -> b && a > v ? a : v,
                     (d,i,v) -> d[i] = v)
    end
    return dst
end

function dilate!(dst::AbstractArray{T,N},
                 A::AbstractArray{T,N},
                 R::Kernel{K,N}) where {T<:AbstractFloat,
                                        K<:AbstractFloat,N}
    if ismmbox(R)
        localfilter!(dst, A, :, max, axes(R))
    else
        check_indices(dst, A)
        localfilter!(dst, A, Kernel{T}(R),
                     (a)     -> typemin(T),
                     (v,a,b) -> max(v, a + b),
                     (d,i,v) -> d[i] = v)
    end
    return dst
end

"""
    localextrema(A, R=3) -> Amin, Amax

yields the results of performing an erosion and a dilation of `A` by the
structuring element defined by `R` in a single pass.  Calling this method is
usually almost twice as fast as calling [`erode`](@ref) and [`dilate`](@ref).

See [`localextrema!`](@ref) for an in-place version of the method, and
[`erode`](@ref) or [`dilate`](@ref) for a description of these operations.

"""
localextrema(A::AbstractArray, args...) =
    localextrema!(similar(A), similar(A), A, args...)

"""
    localextrema!(Amin, Amax, A, R=3) -> Amin, Amax

overwrites `Amin` and `Amax` with, respectively, an erosion and a dilation of
the array `A` by by the structuring element defined by `R` in a single pass.

See [`localextrema`](@ref) for an out-of-place version for more information.

"""
function localextrema!(Amin::AbstractArray{T,N},
                       Amax::AbstractArray{T,N},
                       A::AbstractArray{T,N}, R=3) where {T,N}
    localextrema!(Amin, Amax, A, Neighborhood{N}(R))
end

function localextrema!(Amin::AbstractArray{T,N},
                       Amax::AbstractArray{T,N},
                       A::AbstractArray{T,N},
                       R::RectangularBox{N}) where {T,N}
    check_indices(Amin, Amax, A)
    localfilter!((Amin, Amax), A, R,
                 (a)     -> (typemax(T),
                             typemin(T)),
                 (v,a,b) -> (min(v[1], a),
                             max(v[2], a)),
                 (d,i,v) -> (Amin[i], Amax[i]) = v)
end

function localextrema!(Amin::AbstractArray{T,N},
                       Amax::AbstractArray{T,N},
                       A::AbstractArray{T,N},
                       R::Kernel{Bool,N}) where {T,N}
    check_indices(Amin, Amax, A)
    localfilter!((Amin, Amax), A, R,
                 (a)     -> (typemax(T),
                             typemin(T)),
                 (v,a,b) -> (b && a < v[1] ? a : v[1],
                             b && a > v[2] ? a : v[2]),
                 (d,i,v) -> (Amin[i], Amax[i]) = v)
end

function localextrema!(Amin::AbstractArray{T,N},
                       Amax::AbstractArray{T,N},
                       A::AbstractArray{T,N},
                       R::Kernel{K,N}) where {T<:AbstractFloat,
                                              K<:AbstractFloat,N}
    check_indices(Amin, Amax, A)
    localfilter!((Amin, Amax), A, Kernel{T}(R),
                 (a)     -> (typemax(T),
                             typemin(T)),
                 (v,a,b) -> (min(v[1], a - b),
                             max(v[2], a + b)),
                 (d,i,v) -> (Amin[i], Amax[i]) = v)
end

#------------------------------------------------------------------------------
# Higher level operators.

"""
    closing(A, R=3) -> dst

yields a closing of array `A` by the structuring element `R`.  A closing is a
dilation followed by an erosion.  The result `dst` is an array similar to `A`.

See [`closing!`](@ref) for an in-place version of the method, [`opening`](@ref)
for a related filter, and [`erode`](@ref) or [`dilate`](@ref) for a description
of these operations.

"""
closing(A::AbstractArray, args...) =
    closing!(similar(A), similar(A), A, args...)

"""
    closing!(dst, wrk, A, R=3) -> dst

overwrites destination `dst` with the result of a closing of the source `A` by
a structuring element specified by `R` using `wrk` as a workspace array.  The 3
arguments `dst`, `wrk`, and `A` must be similar arrays, `dst` and `A` may be
identical, but `wrk` must not be the same array as `A` or `dst`.  The
destination `dst` is returned.

See [`closing`](@ref) for a description of this kind of filter and for the
meaning of the arguments.

"""
function closing!(dst::AbstractArray{T,N},
                  wrk::AbstractArray{T,N},
                  A::AbstractArray{T,N}, R=3) where {T,N}
    closing!(dst, wrk, A, Neighborhood{N}(R))
end

function closing!(dst::AbstractArray{T,N},
                  wrk::AbstractArray{T,N},
                  A::AbstractArray{T,N},
                  R::Neighborhood{N}) where {T,N}
    erode!(dst, dilate!(wrk, A, R), R)
end

"""
    opening(A, R=3) -> dst

yields an opening of array `A` by the structuring element `R`.  An opening is
an erosion followed by a dilation.  The result `dst` is an array similar to
`A`.

See [`opening!`](@ref) for an in-place version of the method, [`closing`](@ref)
for a related filter, and [`erode`](@ref) or [`dilate`](@ref) for a description
of these operations.

"""
opening(A::AbstractArray, args...) =
    opening!(similar(A), similar(A), A, args...)

"""
    opening!(dst, wrk, A, R=3) -> dst

overwrites destination `dst` with the result of an opening of the source `A` by
a structuring element specified by `R` using `wrk` as a workspace array.  The 3
arguments `dst`, `wrk`, and `A` must be similar arrays, `dst` and `A` may be
identical, but `wrk` must not be the same array as `A` or `dst`.  The
destination `dst` is returned.

See [`opening`](@ref) for a description of this kind of filter and for the
meaning of the arguments.

"""
function opening!(dst::AbstractArray{T,N},
                  wrk::AbstractArray{T,N},
                  A::AbstractArray{T,N}, R=3) where {T,N}
    opening!(dst, wrk, A, Neighborhood{N}(R))
end

function opening!(dst::AbstractArray{T,N},
                  wrk::AbstractArray{T,N},
                  A::AbstractArray{T,N},
                  R::Neighborhood{N}) where {T,N}
    dilate!(dst, erode!(wrk, A, R), R)
end

# Out-of-place top hat filter requires 2 allocations without a
# pre-filtering, 3 allocations with a pre-filtering.

"""
    top_hat(A, R[, S]) -> dst

performs a *summit detection* by applying a top-hat filter to array `A`.
Argument `R` defines the structuring element for the feature detection.
Top-hat filtering is equivalent to:

    dst = A .- opening(A, R)

Optional argument `S` specifies the structuring element for smoothing `A` prior
to the top-hat filter.  If `R` and `S` are specified as the radii of the
structuring elements, then `S` should be smaller than `R`.  For instance:

    top_hat(bitmap, 3, 1)

may be used to detect text or lines in a bitmap image.

See [`bottom_hat`](@ref) for a related operation,
[`LocalFilters.top_hat!`](@ref) for an in-place version.

"""
top_hat(A, R=3) = top_hat!(similar(A), similar(A), A, R)

function top_hat(A, R, S)
    wrk = similar(A)
    top_hat!(similar(A), wrk, closing!(similar(A), wrk, A, S), R)
end

"""
    LocalFilters.top_hat!(dst, wrk, A, R[, S]) -> dst

overwrites `dst` with the result of a top-hat filter applied to `A` with
structuring element `R` and optional smoothing element `S`.  Argument `wrk` is
a workspace array whose contents is not preserved.  The 3 arguments `A`, `dst`,
and `wrk` must be similar but different arrays.  The destination `dst` is
returned.

See also [`top_hat`](@ref) for more details.

"""
function top_hat!(dst::AbstractArray{T,N},
                  wrk::AbstractArray{T,N},
                  A::AbstractArray{T,N}, R=3) where {T,N}
    opening!(dst, wrk, A, R)
    @inbounds for i in eachindex(dst, A)
        dst[i] = A[i] - dst[i]
    end
    return dst
end

"""
    bottom_hat(A, R[, S]) -> dst

performs a *valley detection* by applying a bottom-hat filter to array `A`.
Argument `R` defines the structuring element for the feature detection.
Bottom-hat filtering is equivalent to:

    dst = closing(A, R) .- A

Optional argument `S` specifies the structuring element for smoothing `A` prior
to the top-hat filter.  If `R` and `S` are specified as the radii of the
structuring elements, then `S` should be smaller than `R`.

See [`top_hat`](@ref) for a related operation,
[`LocalFilters.bottom_hat!`](@ref) for an in-place version.

"""
bottom_hat(A, R=3) = bottom_hat!(similar(A), similar(A), A, R)

function bottom_hat(A, R, S)
    wrk = similar(A)
    return bottom_hat!(similar(A), wrk, opening!(similar(A), wrk, A, S), R)
end

"""
    LocalFilters.bottom_hat!(dst, wrk, A, R[, S]) -> dst

overwrites `dst` with the result of a bottom-hat filter applied to `A` with
structuring element `R` and optional smoothing element `S`.  Argument `wrk` is
a workspace array whose contents is not preserved.  The 3 arguments `A`, `dst`,
and `wrk` must be similar but different arrays.  The destination `dst` is
returned.

See also [`bottom_hat`](@ref) for more details.

"""
function bottom_hat!(dst::AbstractArray{T,N},
                     wrk::AbstractArray{T,N},
                     A::AbstractArray{T,N},
                     R=3) where {T,N}
    closing!(dst, wrk, A, R)
    @inbounds for i in eachindex(dst, A)
        dst[i] -= A[i]
    end
    return dst
end
