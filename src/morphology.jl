#
# morphology.jl --
#
# Implementation of non-linear morphological filters.
#
#-----------------------------------------------------------------------------------------
#
# This file is part of the `LocalFilters.jl` package licensed under the MIT "Expat"
# License.
#
# Copyright (C) 2017-2025, Éric Thiébaut.
#

"""
    erode(A, [ord=ForwardFilter,] B=3; slow=false) -> Amin

yields the erosion of `A` by the structuring element defined by `B`. The returned result,
`Amin`, is similar to `A` (same size and type) and its values are the local minima of `A`
in the neighborhood defined by `B`.

If `B` is not a kernel (that is, if `B` is not an array or is an instance of
`CartesianIndices`), [`kernel(Dims{N},B)`](@ref) is called to build a kernel with `N` the
number of dimensions of `A`.

If the structuring element `B` is equivalent to a simple hyper-rectangular sliding window
(which is the case by default), the much faster van Herk-Gil-Werman algorithm is used.

An erosion is one of the most basic operations of mathematical morphology. See
[`erode!`](@ref) for an in-place version of the method, [`dilate`](@ref) for retrieving
the local maxima, and [`localextrema`](@ref) for performing an erosion and a dilation in a
single pass.

""" erode

"""
    erode!(Amin, A, [ord=ForwardFilter,] B=3; slow=false) -> Amin

overwrites `Amin` with the erosion of the array `A` by the structuring element defined by
`B` and returns `Amin`.

If the structuring element `B` is equivalent to a simple hyper-rectangular sliding window
(which is the case by default) and unless keyword `slow` is true, the much faster van
Herk-Gil-Werman algorithm is used and the operation can be done in-place. That is, `A` and
`Amin` can be the same arrays. In that case, the following syntax is allowed:

    erode!(A, [ord=ForwardFilter,] B=3) -> A

See [`erode`](@ref) for an out-of-place version and for more information.

""" erode!

"""
    dilate(A, [ord=ForwardFilter,] B=3; slow=false) -> Amax

yields the dilation of `A` by the structuring element defined by `B`. The returned result,
`Amax`, is similar to `A` (same size and type) and its values are the local maxima of `A`
in the neighborhood defined by `B`.

If `B` is not a kernel (that is, if `B` is not an array or is an instance of
`CartesianIndices`), [`kernel(Dims{N},B)`](@ref) is called to build a kernel with `N` the
number of dimensions of `A`.

If the structuring element `B` is equivalent to a simple hyper-rectangular sliding window
(which is the case by default), the much faster van Herk-Gil-Werman algorithm is used.

A dilation is one of the most basic operations of mathematical morphology. See
[`dilate!`](@ref) for an in-place version of the method, [`erode`](@ref) for retrieving
the local minima, and [`localextrema`](@ref) for performing an erosion and a dilation in a
single pass.

""" dilate

"""
    dilate!(Amax, A, [ord=ForwardFilter,] B=3; slow=false) -> Amax

overwrites `Amax` with a dilation of the array `A` by the structuring element defined by
`B` and returns `Amax`.

If the structuring element `B` is equivalent to a simple hyper-rectangular sliding window
(which is the case by default) and unless keyword `slow` is true, the much faster van
Herk-Gil-Werman algorithm is used and the operation can be done in-place. That is, `A` and
`Amin` can be the same arrays. In that case, the following syntax is allowed:

    dilate!(A, [ord=ForwardFilter,] B=3) -> A

See [`dilate`](@ref) for an out-of-place version and for more information.

""" dilate!

for (f, op) in ((:erode, :min), (:dilate, :max))
    f! = Symbol(f,:!)
    slow_f! = Symbol(:slow_,f!)
    fast_f! = Symbol(:fast_,f!)
    @eval begin
        # Provide destination array.
        $f(A::AbstractArray, args...; kwds...) = $f!(similar(A), A, args...; kwds...)

        # Provide default ordering and default structuring element.
        function $f!(dst::AbstractArray{<:Any,N},
                     A::AbstractArray{<:Any,N},
                     B::Union{Window{N},AbstractArray{<:Any,N}} = 3;
                     kwds...) where {N}
            return $f!(dst, A, ForwardFilter, B; kwds...)
        end

        # Build structuring element.
        function $f!(dst::AbstractArray{<:Any,N},
                     A::AbstractArray{<:Any,N},
                     ord::FilterOrdering,
                     B::Window{N} = 3;
                     kwds...) where {N}
            return $f!(dst, A, ord, kernel(Dims{N}, B); kwds...)
        end

        # Fast separable filter (out-of-place).
        function $fast_f!(dst::AbstractArray{<:Any,N},
                          A::AbstractArray{<:Any,N},
                          ord::FilterOrdering,
                          B::Box{N}) where {N}
            localfilter!(dst, A, :, $op, ord, axes(B))
        end
        function $fast_f!(dst::AbstractArray{T,N},
                          A::AbstractArray{<:Any,N},
                          ord::FilterOrdering,
                          B::Box{N},
                          wrk::Vector{T}) where {T,N}
            localfilter!(dst, A, :, $op, ord, axes(B), wrk)
        end

        # Fast separable filter (in-place).
        function $f!(A::AbstractArray{<:Any,N},
                     ord::FilterOrdering,
                     B::Box{N}) where {N}
            localfilter!(A, :, $op, ord, axes(B))
        end
        function $f!(A::AbstractArray{<:Any,N},
                     ord::FilterOrdering,
                     B::Box{N},
                     wrk::Vector) where {N}
            localfilter!(A, :, $op, ord, axes(B), wrk)
        end

        # General case.
        function $f!(dst::AbstractArray{<:Any,N},
                     A::AbstractArray{<:Any,N},
                     ord::FilterOrdering,
                     B::AbstractArray{<:Any,N};
                     slow::Bool = false) where {N}
            if !slow && is_morpho_math_box(B)
                # Use fast separable filter.
                $fast_f!(dst, A, ord, morpho_math_box(B))
            else
                # Use slow filter.
                $slow_f!(dst, A, ord, B)
            end
            return dst
        end
    end
end

# This version is for a simple hyper-rectangular structuring element. By default, the fast
# version of this is always used.
function slow_erode!(dst::AbstractArray{<:Any,N},
                     A::AbstractArray{T,N},
                     ord::FilterOrdering,
                     B::Box{N}) where {T,N}
    indices = Indices(dst, A, B)
    @inbounds for i in indices(dst)
        v = typemax(T)
        @simd for j in localindices(indices(A), ord, indices(B), i)
            v = min(v, A[j])
        end
        dst[i] = v
    end
    return dst
end

# This version is for a boolean structuring element representing a shaped neighborhood.
function slow_erode!(dst::AbstractArray{<:Any,N},
                     A::AbstractArray{T,N},
                     ord::FilterOrdering,
                     B::AbstractArray{Bool,N}) where {T,N}
    indices = Indices(dst, A, B)
    @inbounds for i in indices(dst)
        v = typemax(T)
        @simd for j in localindices(indices(A), ord, indices(B), i)
            v = ifelse(B[ord(i,j)], min(v, A[j]), v)
        end
        dst[i] = v
    end
    return dst
end

# This version is for a "flat" structuring element or for grayscale morphology.
function slow_erode!(dst::AbstractArray{<:Any,N},
                     A::AbstractArray{<:Any,N},
                     ord::FilterOrdering,
                     B::AbstractArray{<:Any,N}) where {N}
    T = promote_type(eltype(A), eltype(B))
    indices = Indices(dst, A, B)
    @inbounds for i in indices(dst)
        v = typemax(T)
        for j in localindices(indices(A), ord, indices(B), i)
            v = min(v, A[j] - B[ord(i,j)])
        end
        dst[i] = v
    end
    return dst
end

# This version is for a simple hyper-rectangular structuring element. By default, the fast
# version of this is always used.
function slow_dilate!(dst::AbstractArray{<:Any,N},
                      A::AbstractArray{T,N},
                      ord::FilterOrdering,
                      B::Box{N}) where {T,N}
    indices = Indices(dst, A, B)
    @inbounds for i in indices(dst)
        v = typemin(T)
        @simd for j in localindices(indices(A), ord, indices(B), i)
            v = max(v, A[j])
        end
        dst[i] = v
    end
    return dst
end

# This version is for a boolean structuring element representing a shaped neighborhood.
function slow_dilate!(dst::AbstractArray{<:Any,N},
                      A::AbstractArray{T,N},
                      ord::FilterOrdering,
                      B::AbstractArray{Bool,N}) where {T,N}
    indices = Indices(dst, A, B)
    @inbounds for i in indices(dst)
        v = typemin(T)
        @simd for j in localindices(indices(A), ord, indices(B), i)
            v = ifelse(B[ord(i,j)], max(v, A[j]), v)
        end
        dst[i] = v
    end
    return dst
end

# This version is for a "flat" structuring element or for grayscale morphology.
function slow_dilate!(dst::AbstractArray{<:Any,N},
                      A::AbstractArray{<:Any,N},
                      ord::FilterOrdering,
                      B::AbstractArray{<:Any,N}) where {N}
    T = promote_type(eltype(A), eltype(B))
    indices = Indices(dst, A, B)
    @inbounds for i in indices(dst)
        v = typemin(T)
        @simd for j in localindices(indices(A), ord, indices(B), i)
            v = max(v, A[j] + B[ord(i,j)])
        end
        dst[i] = v
    end
    return dst
end

"""
    localextrema(A, [ord=ForwardFilter,] B=3) -> Amin, Amax

yields the results of performing an erosion and a dilation of `A` by the structuring
element defined by `B` in a single pass. Calling this method is usually almost twice as
fast as calling [`erode`](@ref) and [`dilate`](@ref).

See [`localextrema!`](@ref) for an in-place version of the method, and [`erode`](@ref) or
[`dilate`](@ref) for a description of these operations.

"""
localextrema(A::AbstractArray, args...) = localextrema!(similar(A), similar(A), A, args...)

"""
    localextrema!(Amin, Amax, A, [ord=ForwardFilter,] B=3) -> Amin, Amax

overwrites `Amin` and `Amax` with, respectively, an erosion and a dilation of the array
`A` by the structuring element defined by `B` in a single pass.

See [`localextrema`](@ref) for an out-of-place version for more information.

"""
function localextrema!(Amin::AbstractArray{<:Any,N},
                       Amax::AbstractArray{<:Any,N},
                       A::AbstractArray{<:Any,N},
                       B::Union{Window{N},AbstractArray{<:Any,N}} = 3) where {N}
    return localextrema!(Amin, Amax, A, ForwardFilter, B)
end

function localextrema!(Amin::AbstractArray{<:Any,N},
                       Amax::AbstractArray{<:Any,N},
                       A::AbstractArray{<:Any,N},
                       ord::FilterOrdering,
                       B::Window{N} = 3) where {N}
    return localextrema!(Amin, Amax, A, ord, kernel(Dims{N}, B))
end

# This version is for a simple hyper-rectangular structuring element.
function localextrema!(Amin::AbstractArray{<:Any,N},
                       Amax::AbstractArray{<:Any,N},
                       A::AbstractArray{T,N},
                       ord::FilterOrdering,
                       B::Box{N}) where {T,N}
    indices = Indices(Amin, Amax, A, B)
    @inbounds for i in indices(Amin, Amax)
        vmin, vmax = typemax(T), typemin(T)
        @simd for j in localindices(indices(A), ord, indices(B), i)
            a = A[j]
            vmin = min(vmin, a)
            vmax = max(vmax, a)
        end
        Amin[i] = vmin
        Amax[i] = vmax
    end
    return Amin, Amax
end

# This version is for a boolean structuring element representing a shaped neighborhood.
function localextrema!(Amin::AbstractArray{<:Any,N},
                       Amax::AbstractArray{<:Any,N},
                       A::AbstractArray{T,N},
                       ord::FilterOrdering,
                       B::AbstractArray{Bool,N}) where {T,N}
    indices = Indices(Amin, Amax, A, B)
    @inbounds for i in indices(Amin, Amax)
        vmin, vmax = typemax(T), typemin(T)
        @simd for j in localindices(indices(A), ord, indices(B), i)
            a = A[j]
            b = B[ord(i,j)]
            vmin = ifelse(b, min(vmin, a), vmin)
            vmax = ifelse(b, max(vmax, a), vmax)
        end
        Amin[i] = vmin
        Amax[i] = vmax
    end
    return Amin, Amax
end

# This version is for a "flat" structuring element or for grayscale morphology.
function localextrema!(Amin::AbstractArray{<:Any,N},
                       Amax::AbstractArray{<:Any,N},
                       A::AbstractArray{<:Any,N},
                       ord::FilterOrdering,
                       B::AbstractArray{<:Any,N}) where {N}
    T = promote_type(eltype(A), eltype(B))
    indices = Indices(Amin, Amax, A, B)
    @inbounds for i in indices(Amin, Amax)
        vmin, vmax = typemax(T), typemin(T)
        @simd for j in localindices(indices(A), ord, indices(B), i)
            a = A[j]
            b = B[ord(i,j)]
            vmin = min(vmin, a - b)
            vmax = max(vmax, a + b)
        end
        Amin[i] = vmin
        Amax[i] = vmax
    end
    return Amin, Amax
end

#------------------------------------------------------------------------------
# Higher level operators.

"""
    closing(A, [ord=ForwardFilter,] B=3; slow=false) -> dst

yields a closing of array `A` by the structuring element defined by `B`. A closing is a
dilation followed by an erosion. The result `dst` is an array similar to `A`.

See [`closing!`](@ref) for an in-place version of the method, [`opening`](@ref) for a
related filter, and [`erode`](@ref) or [`dilate`](@ref) for a description of these
operations.

""" closing

"""
    closing!(dst, wrk, A, [ord=ForwardFilter,] B=3; slow=false) -> dst

overwrites `dst` with the result of a closing of `A` by the structuring element defined by
`B` using `wrk` as a workspace array. The arguments `dst`, `wrk`, and `A` must be similar
arrays, `dst` and `A` may be identical, but `wrk` must not be the same array as `A` or
`dst`. The destination `dst` is returned.

See [`closing`](@ref) for a description of this kind of filter and for the meaning of the
arguments.

""" closing!

"""
    opening(A, [ord=ForwardFilter,] B=3; slow=false) -> dst

yields an opening of array `A` by the structuring element defined by `B`. An opening is an
erosion followed by a dilation. The result `dst` is an array similar to `A`.

See [`opening!`](@ref) for an in-place version of the method, [`closing`](@ref) for a
related filter, and [`erode`](@ref) or [`dilate`](@ref) for a description of these
operations.

""" opening

"""
    opening!(dst, wrk, A, [ord=ForwardFilter,] B=3; slow=false) -> dst

overwrites `dst` with the result of an opening of `A` by the structuring element defined
by `B` using `wrk` as a workspace array. The arguments `dst`, `wrk`, and `A` must be
similar arrays, `dst` and `A` may be identical, but `wrk` must not be the same array as
`A` or `dst`. The destination `dst` is returned.

See [`opening`](@ref) for a description of this kind of filter and for the meaning of the
arguments.

""" opening!

for f in (:closing, :opening)
    f! = Symbol(f,:!)
    @eval begin
        # Provide destination and workspace.
        $f(A::AbstractArray, args...; kwds...) = $f!(similar(A), similar(A), A, args...; kwds...)

        # Provide default ordering and default structuring element.
        function $f!(dst::AbstractArray{<:Any,N},
                     wrk::AbstractArray{<:Any,N},
                     A::AbstractArray{<:Any,N},
                     B::Union{Window{N},AbstractArray{<:Any,N}} = 3; kwds...) where {N}
            $f!(dst, wrk, A, ForwardFilter, B; kwds...)
        end
        function $f!(dst::AbstractArray{<:Any,N},
                     wrk::AbstractArray{<:Any,N},
                     A::AbstractArray{<:Any,N},
                     ord::FilterOrdering; kwds...) where {N}
            $f!(dst, wrk, A, ord, 3; kwds...)
        end

        # Build structuring element.
        function $f!(dst::AbstractArray{<:Any,N},
                     wrk::AbstractArray{<:Any,N},
                     A::AbstractArray{<:Any,N},
                     ord::FilterOrdering,
                     B::Window{N}; kwds...) where {N}
            return $f!(dst, wrk, A, ord, kernel(Dims{N}, B); kwds...)
        end
    end
end

function closing!(dst::AbstractArray{<:Any,N},
                  wrk::AbstractArray{<:Any,N},
                  A::AbstractArray{<:Any,N},
                  ord::FilterOrdering,
                  B::AbstractArray{<:Any,N}; kwds...) where {N}
    erode!(dst, dilate!(wrk, A, ord, B; kwds...), ord, B; kwds...)
end

function opening!(dst::AbstractArray{<:Any,N},
                  wrk::AbstractArray{<:Any,N},
                  A::AbstractArray{<:Any,N},
                  ord::FilterOrdering,
                  B::AbstractArray{<:Any,N}; kwds...) where {N}
    dilate!(dst, erode!(wrk, A, ord, B; kwds...), ord, B; kwds...)
end

"""
    top_hat(A, [B_ord=ForwardFilter,] B=3 [, [C_ord=B_ord,] C]; slow=false) -> dst

performs a *summit detection* by applying a top-hat filter to array `A` using the
structuring element defined by `B` for the feature detection. Top-hat filtering is
equivalent to:

    dst = A .- opening(A, B)

Optional argument `C` specifies the structuring element for smoothing `A` prior to top-hat
filtering. If `B` and `C` are specified as the radii of the structuring elements, then `C`
should be smaller than `B`. For instance:

    top_hat(bitmap, 3, 1)

may be used to detect text or lines in a bitmap image.

See [`bottom_hat`](@ref) for a related operation, [`LocalFilters.top_hat!`](@ref) for an
in-place version.

""" top_hat

"""
    LocalFilters.top_hat!(dst, wrk, A, [ord=ForwardFilter,] B=3; slow=false) -> dst

overwrites `dst` with the result of a top-hat filter applied to `A` with structuring
element `B`, and using `wrk` as a workspace whose contents is not preserved. The arguments
`A`, `dst`, and `wrk` must be similar but different arrays. The destination `dst` is
returned.

See also [`top_hat`](@ref) for more details.

""" top_hat!
@public top_hat!

"""
    bottom_hat(A, [B_ord=ForwardFilter,] B=3 [, [C_ord=B_ord,] C]; slow=false) -> dst

performs a *valley detection* by applying a bottom-hat filter to array `A` using the
structuring element defined by `B` for the feature detection. Bottom-hat filtering is
equivalent to:

    dst = closing(A, B) .- A

Optional argument `C` specifies the structuring element for smoothing `A` prior to
bottom-hat filtering. If `B` and `C` are specified as the radii of the structuring
elements, then `C` should be smaller than `B`.

See [`top_hat`](@ref) for a related operation, [`LocalFilters.bottom_hat!`](@ref) for an
in-place version.

""" bottom_hat

"""
    LocalFilters.bottom_hat!(dst, wrk, A, [ord=ForwardFilter,] B=3; slow=false) -> dst

overwrites `dst` with the result of a bottom-hat filter applied to `A` with structuring
element `B` and optional smoothing element `C`. Argument `wrk` is a workspace array whose
contents is not preserved. The arguments `A`, `dst`, and `wrk` must be similar but
different arrays. The destination `dst` is returned.

See also [`bottom_hat`](@ref) for more details.

""" bottom_hat!
@public bottom_hat!

for (f, pf) in ((:top_hat,    :closing),
                (:bottom_hat, :opening))
    f! = Symbol(f,:!)
    pf! = Symbol(pf,:!) # pre-filter
    @eval begin
        # Provide default ordering and default structuring element.
        $f(A::AbstractArray, B=3; kwds...) = $f(A, ForwardFilter, B; kwds...)
        $f(A::AbstractArray, B, C; kwds...) = $f(A, ForwardFilter, B, C; kwds...)
        $f(A::AbstractArray, ord::FilterOrdering, B, C; kwds...) = $f(A, ord, B, ord, C; kwds...)

        # Provide destination and workspace. Out-of-place top/bottom hat
        # filters require 2 allocations without a pre-filtering, 3 allocations
        # with a pre-filtering.
        $f(A::AbstractArray, ord::FilterOrdering, B=3; kwds...) =
            $f!(similar(A), similar(A), A, ord, B; kwds...)
        function $f(A::AbstractArray,
                    B_ord::FilterOrdering, B,
                    C_ord::FilterOrdering, C; kwds...)
            wrk = similar(A)
            $f!(similar(A), wrk, $pf!(similar(A), wrk, A, C_ord, C; kwds...), B_ord, B; kwds...)
        end

        # Provide default ordering and default structuring element.
        function $f!(dst::AbstractArray{<:Any,N},
                     wrk::AbstractArray{<:Any,N},
                     A::AbstractArray{<:Any,N},
                     B::Union{Window{N},AbstractArray{<:Any,N}} = 3; kwds...) where {N}
            return $f!(dst, wrk, A, ForwardFilter, B; kwds...)
        end

        # Build structuring element.
        function $f!(dst::AbstractArray{<:Any,N},
                     wrk::AbstractArray{<:Any,N},
                     A::AbstractArray{<:Any,N},
                     ord::FilterOrdering,
                     B::Window{N} = 3; kwds...) where {N}
            return $f!(dst, wrk, A, ord, kernel(Dims{N}, B); kwds...)
        end
    end
end

function top_hat!(dst::AbstractArray{<:Any,N},
                  wrk::AbstractArray{<:Any,N},
                  A::AbstractArray{<:Any,N},
                  ord::FilterOrdering,
                  B::AbstractArray{<:Any,N}; kwds...) where {N}
    opening!(dst, wrk, A, ord, B; kwds...)
    @inbounds for i in eachindex(dst, A)
        dst[i] = A[i] - dst[i]
    end
    return dst
end

function bottom_hat!(dst::AbstractArray{<:Any,N},
                     wrk::AbstractArray{<:Any,N},
                     A::AbstractArray{<:Any,N},
                     ord::FilterOrdering,
                     B::AbstractArray{<:Any,N}; kwds...) where {N}
    closing!(dst, wrk, A, B; kwds...)
    @inbounds for i in eachindex(dst, A)
        dst[i] -= A[i]
    end
    return dst
end
