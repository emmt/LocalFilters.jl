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
# Copyright (c) 2017-2025, Éric Thiébaut.
#

"""
    erode(A, B=3; order=FORWARD_FILTER, slow=false) -> Amin

yields the erosion of `A` by the structuring element defined by `B`. The returned result,
`Amin`, is similar to `A` (same size and type) and its values are the local minima of `A`
in the neighborhood defined by `B`.

Keyword `order` specifies the filter direction, `FORWARD_FILTER` by default.

If `B` is not an `N`-dimensional array, [`kernel(Dims{N},B)`](@ref) is called to build a
kernel with `N = ndims(A)` the number of dimensions of `A`.

If the structuring element `B` is equivalent to a simple hyper-rectangular sliding window
(which is the case by default) and unless keyword `slow` is true, the much faster van
Herk-Gil-Werman algorithm is used.

An erosion is one of the most basic operations of mathematical morphology. See
[`erode!`](@ref) for an in-place version of the method, [`dilate`](@ref) for retrieving
the local maxima, and [`localextrema`](@ref) for performing an erosion and a dilation in a
single pass.

""" erode

"""
    erode!(Amin, A, B=3; order=FORWARD_FILTER, slow=false) -> Amin

overwrites `Amin` with the erosion of the array `A` by the structuring element defined by
`B` and returns `Amin`.

Keyword `order` specifies the filter direction, `FORWARD_FILTER` by default.

If the structuring element `B` is equivalent to a simple hyper-rectangular sliding window
(which is the case by default) and unless keyword `slow` is true, the much faster van
Herk-Gil-Werman algorithm is used and the operation can be done in-place. That is, `A` and
`Amin` can be the same arrays. In that case, the following syntax is allowed:

    erode!(A, B=3; order=FORWARD_FILTER, ) -> A

See [`erode`](@ref) for an out-of-place version and for more information.

""" erode!

"""
    dilate(A, B=3; order=FORWARD_FILTER, slow=false) -> Amax

yields the dilation of `A` by the structuring element defined by `B`. The returned result,
`Amax`, is similar to `A` (same size and type) and its values are the local maxima of `A`
in the neighborhood defined by `B`.

Keyword `order` specifies the filter direction, `FORWARD_FILTER` by default.

If `B` is not an `N`-dimensional array, [`kernel(Dims{N},B)`](@ref) is called to build a
kernel with `N = ndims(A)` the number of dimensions of `A`.

If the structuring element `B` is equivalent to a simple hyper-rectangular sliding window
(which is the case by default) and unless keyword `slow` is true, the much faster van
Herk-Gil-Werman algorithm is used.

A dilation is one of the most basic operations of mathematical morphology. See
[`dilate!`](@ref) for an in-place version of the method, [`erode`](@ref) for retrieving
the local minima, and [`localextrema`](@ref) for performing an erosion and a dilation in a
single pass.

""" dilate

"""
    dilate!(Amax, A, B=3; order=FORWARD_FILTER, slow=false) -> Amax

overwrites `Amax` with a dilation of the array `A` by the structuring element defined by
`B` and returns `Amax`.

Keyword `order` specifies the filter direction, `FORWARD_FILTER` by default.

If the structuring element `B` is equivalent to a simple hyper-rectangular sliding window
(which is the case by default) and unless keyword `slow` is true, the much faster van
Herk-Gil-Werman algorithm is used and the operation can be done in-place. That is, `A` and
`Amin` can be the same arrays. In that case, the following syntax is allowed:

    dilate!(A, B=3; order=FORWARD_FILTER) -> A

See [`dilate`](@ref) for an out-of-place version and for more information.

""" dilate!

for (f, op) in ((:erode, :min), (:dilate, :max))
    f! = Symbol(f,:!)
    slow_f! = Symbol(:slow_,f!)
    @eval begin
        # Provide destination array.
        $f(A::AbstractArray, args...; kwds...) = $f!(similar(A), A, args...; kwds...)

        # Build structuring element.
        function $f!(dst::AbstractArray{<:Any,N}, A::AbstractArray{<:Any,N},
                     B::Window{N} = 3; kwds...) where {N}
            return $f!(dst, A, kernel(Dims{N}, B); kwds...)
        end

        # General case (out-of-place).
        function $f!(dst::AbstractArray{<:Any,N}, A::AbstractArray{<:Any,N},
                     B::AbstractArray{<:Any,N}; slow::Bool = false, kwds...) where {N}
            if !slow && is_morpho_math_box(B)
                # Use fast separable filter.
                localfilter!(dst, A, :, $op, axes(B); kwds...)
            else
                # Use slow filter.
                $slow_f!(dst, A, B; kwds...)
            end
            return dst
        end

        # Fast separable filter (in-place).
        function $f!(A::AbstractArray{<:Any,N}, B::Window{N} = 3; kwds...) where {N}
            _B = kernel(Dims{N}, B)
            is_morpho_math_box(_B) || throw(ArgumentError(
                "for in-place operation kernel must be a simple box"))
            return localfilter!(A, A, :, $op, axes(_B); kwds...)
        end
    end
end

function slow_erode!(dst::AbstractArray{<:Any,N}, A::AbstractArray{<:Any,N},
                     B::AbstractArray{<:Any,N};
                     order::FilterOrdering = FORWARD_FILTER) where {N}
    indices = Indices(dst, A, B)
    if B isa Box{N}
        # `B` is a simple hyper-rectangular structuring element. By default, the fast
        # version of this is always used.
        T = eltype(A)
        @inbounds for i in indices(dst)
            v = typemax(T)
            @simd for j in localindices(indices(A), order, indices(B), i)
                v = min(v, A[j])
            end
            dst[i] = v
        end
    elseif B isa AbstractArray{Bool,N}
        # `B` is a Boolean structuring element representing a shaped neighborhood.
        T = eltype(A)
        @inbounds for i in indices(dst)
            v = typemax(T)
            @simd for j in localindices(indices(A), order, indices(B), i)
                v = ifelse(B[order(i,j)], min(v, A[j]), v)
            end
            dst[i] = v
        end
    else
        # `B` is a "flat" structuring element or for grayscale morphology.
        T = promote_type(eltype(A), eltype(B)) # FIXME may be widen is small integers
        @inbounds for i in indices(dst)
            v = typemax(T)
            for j in localindices(indices(A), order, indices(B), i)
                v = min(v, A[j] - B[order(i,j)])
            end
            dst[i] = v
        end
    end
    return dst
end

function slow_dilate!(dst::AbstractArray{<:Any,N}, A::AbstractArray{<:Any,N},
                      B::AbstractArray{<:Any,N};
                      order::FilterOrdering = FORWARD_FILTER) where {N}
    indices = Indices(dst, A, B)
    if B isa Box{N}
        # `B` is a simple hyper-rectangular structuring element. By default, the fast
        # version of this is always used.
        T = eltype(A)
        @inbounds for i in indices(dst)
            v = typemin(T)
            @simd for j in localindices(indices(A), order, indices(B), i)
                v = max(v, A[j])
            end
            dst[i] = v
        end
    elseif B isa AbstractArray{Bool,N}
        # `B` is a Boolean structuring element representing a shaped neighborhood.
        T = eltype(A)
        @inbounds for i in indices(dst)
            v = typemin(T)
            @simd for j in localindices(indices(A), order, indices(B), i)
                v = ifelse(B[order(i,j)], max(v, A[j]), v)
            end
            dst[i] = v
        end
    else
        # `B` is a "flat" structuring element or for grayscale morphology.
        T = promote_type(eltype(A), eltype(B)) # FIXME may be widen is small integers
        @inbounds for i in indices(dst)
            v = typemin(T)
            @simd for j in localindices(indices(A), order, indices(B), i)
                v = max(v, A[j] + B[order(i,j)])
            end
            dst[i] = v
        end
    end
    return dst
end

"""
    localextrema(A, B=3; order=FORWARD_FILTER) -> Amin, Amax

yields the results of performing an erosion and a dilation of `A` by the structuring
element defined by `B` in a single pass. Calling this method is usually almost twice as
fast as calling [`erode`](@ref) and [`dilate`](@ref).

Keyword `order` specifies the filter direction, `FORWARD_FILTER` by default.

See [`localextrema!`](@ref) for an in-place version of the method, and [`erode`](@ref) or
[`dilate`](@ref) for a description of these operations.

"""
localextrema(A::AbstractArray, args...; kwds...) =
    localextrema!(similar(A), similar(A), A, args...; kwds...)

"""
    localextrema!(Amin, Amax, A, B=3; order=FORWARD_FILTER) -> Amin, Amax

overwrites `Amin` and `Amax` with, respectively, an erosion and a dilation of the array
`A` by the structuring element defined by `B` in a single pass.

Keyword `order` specifies the filter direction, `FORWARD_FILTER` by default.

See [`localextrema`](@ref) for an out-of-place version for more information.

"""
function localextrema!(Amin::AbstractArray{<:Any,N},
                       Amax::AbstractArray{<:Any,N},
                       A::AbstractArray{<:Any,N},
                       B::Window{N} = 3; kwds...) where {N}
    return localextrema!(Amin, Amax, A, kernel(Dims{N}, B); kwds...)
end

function localextrema!(Amin::AbstractArray{<:Any,N}, Amax::AbstractArray{<:Any,N},
                       A::AbstractArray{<:Any,N}, B::AbstractArray{<:Any,N};
                       order::FilterOrdering = FORWARD_FILTER) where {N}
    indices = Indices(Amin, Amax, A, B)
    if B isa Box{N}
        # `B` is a simple hyper-rectangular structuring element.
        T = eltype(A)
        @inbounds for i in indices(Amin, Amax)
            vmin, vmax = typemax(T), typemin(T)
            @simd for j in localindices(indices(A), order, indices(B), i)
                a = A[j]
                vmin = min(vmin, a)
                vmax = max(vmax, a)
            end
            Amin[i] = vmin
            Amax[i] = vmax
        end
    elseif B isa AbstractArray{Bool,N}
        # `B` is a Boolean structuring element representing a shaped neighborhood.
        T = eltype(A)
        @inbounds for i in indices(Amin, Amax)
            vmin, vmax = typemax(T), typemin(T)
            @simd for j in localindices(indices(A), order, indices(B), i)
                a = A[j]
                b = B[order(i,j)]
                vmin = ifelse(b, min(vmin, a), vmin)
                vmax = ifelse(b, max(vmax, a), vmax)
            end
            Amin[i] = vmin
            Amax[i] = vmax
        end
    else
        # `B` is a "flat" structuring element or for grayscale morphology.
        T = promote_type(eltype(A), eltype(B)) # FIXME may be widen is small integers
        @inbounds for i in indices(Amin, Amax)
            vmin, vmax = typemax(T), typemin(T)
            @simd for j in localindices(indices(A), order, indices(B), i)
                a = A[j]
                b = B[order(i,j)]
                vmin = min(vmin, a - b)
                vmax = max(vmax, a + b)
            end
            Amin[i] = vmin
            Amax[i] = vmax
        end
    end
    return Amin, Amax
end

#------------------------------------------------------------------------------
# Higher level operators.

"""
    closing(A, B=3; order=FORWARD_FILTER, slow=false) -> dst

yields a closing of array `A` by the structuring element defined by `B`. A closing is a
dilation followed by an erosion. The result `dst` is an array similar to `A`.

See [`closing!`](@ref) for an in-place version of the method, [`opening`](@ref) for a
related filter, and [`erode`](@ref) or [`dilate`](@ref) for a description of these
operations and for the meaning of keywords.

""" closing

"""
    closing!(dst, wrk, A, B=3; order=FORWARD_FILTER, slow=false) -> dst

overwrites `dst` with the result of a closing of `A` by the structuring element defined by
`B` using `wrk` as a workspace array. The arguments `dst`, `wrk`, and `A` must be similar
arrays, `dst` and `A` may be identical, but `wrk` must not be the same array as `A` or
`dst`. The destination `dst` is returned.

See [`closing`](@ref) for a description of this kind of filter and for the meaning of the
arguments and keywords.

"""
function closing!(dst::AbstractArray{<:Any,N}, wrk::AbstractArray{<:Any,N},
                  A::AbstractArray{<:Any,N}, B::AbstractArray{<:Any,N}; kwds...) where {N}
    return erode!(dst, dilate!(wrk, A, B; kwds...), B; kwds...)
end


"""
    opening(A, B=3; order=FORWARD_FILTER, slow=false) -> dst

yields an opening of array `A` by the structuring element defined by `B`. An opening is an
erosion followed by a dilation. The result `dst` is an array similar to `A`.

See [`opening!`](@ref) for an in-place version of the method, [`closing`](@ref) for a
related filter, and [`erode`](@ref) or [`dilate`](@ref) for a description of these
operations and for the meaning of keywords.

""" opening

"""
    opening!(dst, wrk, A, B=3; order=FORWARD_FILTER, slow=false) -> dst

overwrites `dst` with the result of an opening of `A` by the structuring element defined
by `B` using `wrk` as a workspace array. The arguments `dst`, `wrk`, and `A` must be
similar arrays, `dst` and `A` may be identical, but `wrk` must not be the same array as
`A` or `dst`. The destination `dst` is returned.

See [`opening`](@ref) for a description of this kind of filter and for the meaning of the
arguments and keywords.

"""
function opening!(dst::AbstractArray{<:Any,N}, wrk::AbstractArray{<:Any,N},
                  A::AbstractArray{<:Any,N}, B::AbstractArray{<:Any,N}; kwds...) where {N}
    return dilate!(dst, erode!(wrk, A, B; kwds...), B; kwds...)
end

for f in (:closing, :opening)
    f! = Symbol(f,:!)
    @eval begin
        # Provide destination and workspace.
        function $f(A::AbstractArray{<:Any,N},
                    B::Kernel{N} = 3; kwds...) where {N}
            return $f!(similar(A), similar(A), A, B; kwds...)
        end

        # Provide default structuring element and build kernel.
        function $f!(dst::AbstractArray{<:Any,N}, wrk::AbstractArray{<:Any,N},
                     A::AbstractArray{<:Any,N}, B::Window{N} = 3; kwds...) where {N}
            return $f!(dst, wrk, A, kernel(Dims{N}, B); kwds...)
        end
    end
end

"""
    top_hat(A, B=3[, C]; order=FORWARD_FILTER, slow=false) -> dst

performs a *summit detection* by applying a top-hat filter to array `A` using the
structuring element defined by `B` for the feature detection. Top-hat filtering is
equivalent to:

    dst = A .- opening(A, B)

Optional argument `C` specifies the structuring element for smoothing `A` prior to top-hat
filtering. If `B` and `C` are specified as the radii of the structuring elements, then `C`
should be smaller than `B`. For instance:

    top_hat(bitmap, 3, 1)

may be used to detect text or lines in a bitmap image.

Keyword `order` specifies the filter(s) direction(s), `FORWARD_FILTER` by default. If `C`
is specified, `order` may be a 2-tuple to specify a first order for `B` and a second one
for `C`.

See [`bottom_hat`](@ref) for a related operation, [`LocalFilters.top_hat!`](@ref) for an
in-place version.

""" top_hat

"""
    LocalFilters.top_hat!(dst, wrk, A, B=3; order=FORWARD_FILTER, slow=false) -> dst

overwrites `dst` with the result of a top-hat filter applied to `A` with structuring
element `B`, and using `wrk` as a workspace whose contents is not preserved. The arguments
`A`, `dst`, and `wrk` must be similar but different arrays. The destination `dst` is
returned.

See also [`top_hat`](@ref) for more details.

"""
function top_hat!(dst::AbstractArray{<:Any,N},
                  wrk::AbstractArray{<:Any,N},
                  A::AbstractArray{<:Any,N},
                  B::Kernel{N}; kwds...) where {N}
    opening!(dst, wrk, A, B; kwds...)
    @inbounds for i in eachindex(dst, A)
        dst[i] = A[i] - dst[i]
    end
    return dst
end

"""
    bottom_hat(A, B=3[, C]; order=FORWARD_FILTER, slow=false) -> dst

performs a *valley detection* by applying a bottom-hat filter to array `A` using the
structuring element defined by `B` for the feature detection. Bottom-hat filtering is
equivalent to:

    dst = closing(A, B) .- A

Optional argument `C` specifies the structuring element for smoothing `A` prior to
bottom-hat filtering. If `B` and `C` are specified as the radii of the structuring
elements, then `C` should be smaller than `B`.

Keyword `order` specifies the filter(s) direction(s), `FORWARD_FILTER` by default. If `C`
is specified, `order` may be a 2-tuple to specify a first order for `B` and a second one
for `C`.

See [`top_hat`](@ref) for a related operation, [`LocalFilters.bottom_hat!`](@ref) for an
in-place version.

""" bottom_hat

"""
    LocalFilters.bottom_hat!(dst, wrk, A, B=3; order=FORWARD_FILTER, slow=false) -> dst

overwrites `dst` with the result of a bottom-hat filter applied to `A` with structuring
element `B` and optional smoothing element `C`. Argument `wrk` is a workspace array whose
contents is not preserved. The arguments `A`, `dst`, and `wrk` must be similar but
different arrays. The destination `dst` is returned.

See also [`bottom_hat`](@ref) for more details.

"""
function bottom_hat!(dst::AbstractArray{<:Any,N}, wrk::AbstractArray{<:Any,N},
                     A::AbstractArray{<:Any,N}, B::Kernel{N} = 3; kwds...) where {N}
    closing!(dst, wrk, A, B; kwds...)
    @inbounds for i in eachindex(dst, A)
        dst[i] -= A[i]
    end
    return dst
end

for (f, pf!) in ((:top_hat,    :(closing!)),
                 (:bottom_hat, :(opening!)))
    f! = Symbol(f,:!)
    @eval begin
        # Provide destination and workspace. Out-of-place top/bottom hat filters require 2
        # allocations without a pre-filtering, 3 allocations with a pre-filtering.
        $f(A::AbstractArray{<:Any,N}, B::Kernel{N} = 3; kwds...) where {N} =
            $f!(similar(A), similar(A), A, B; kwds...)
        function $f(A::AbstractArray{<:Any,N}, B::Kernel{N}, C::Kernel{N};
                    order::Union{FilterOrdering,NTuple{2,FilterOrdering}} = FORWARD_FILTER,
                    kwds...) where {N}
            B_order = order isa FilterOrdering ? order : order[1]
            C_order = order isa FilterOrdering ? order : order[2]
            wrk = similar(A)
            F = $pf!(similar(A), wrk, A, C; order=C_order, kwds...) # pre-filter
            return $f!(similar(A), wrk, F, B; order=B_order, kwds...)
        end
    end
end
