#
# LocalFilters.jl --
#
# Local filters for Julia arrays.
#
#------------------------------------------------------------------------------
#
# This file is part of the `LocalFilters.jl` package licensed under the MIT
# "Expat" License.
#
# Copyright (C) 2017, Éric Thiébaut.
#

module LocalFilters

import Base: CartesianRange, eltype, ndims, size, length, first, last, tail,
    getindex, setindex!, convert

export localfilter!,
    localmean, localmean!,
    convolve, convolve!,
    erode, erode!,
    dilate, dilate!,
    closing, closing!,
    opening, opening!,
    top_hat,
    bottom_hat,
    localextrema, localextrema!

"""
All neighborhoods are instances of a type derived from `Neighborhood`.
"""
abstract type Neighborhood{N}; end

# Default implementation of common methods.
ndims(::Neighborhood{N}) where N = N
length(B::Neighborhood) = prod(size(B))
size(B::Neighborhood{N}) where N = ntuple(i -> size(B, i), N)

"""

    anchor(B)    -> I::CartesianIndex{N}

yields the anchor of the structuring element `B` that is the Cartesian index of
the central position in the structuring element within its bounding-box.  `N`
is the number of dimensions.  Argument can also be `K` or `size(K)` to get the
default anchor for kernel `K` (an array).

"""
anchor(dims::NTuple{N,Integer}) where {N} =
    CartesianIndex(ntuple(d -> (Int(dims[d]) >> 1) + 1, N))
anchor(B::Neighborhood) = (I = first(B); one(I) - I)
anchor(A::AbstractArray) = anchor(size(A))

"""
The `limits` method yields the corners (as a tuple of 2 `CartesianIndex`)
of `B` (an array, a `CartesianRange` or a `Neighborhood`) and the
infium and supremum of a type `T`:

    limits(B) -> first(B), last(B)
    limits(T) -> typemin(T), typemax(T)

"""
limits(R::CartesianRange) = first(R), last(R)
limits(::Type{T}) where {T} = typemin(T), typemax(T)
limits(A::AbstractArray) = limits(CartesianRange(size(A)))
limits(B::Neighborhood) = first(B), last(B)

CartesianRange(B::Neighborhood{N}) where {N} =
    CartesianRange{CartesianIndex{N}}(first(B), last(B))


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

dilate(A::AbstractArray, args...) = dilate!(similar(A), A, args...)

dilate!(dst, src::AbstractArray{T,N}, B=3) where {T,N} =
    dilate!(dst, src, convert(Neighborhood{N}, B))

localextrema(A::AbstractArray, args...) =
    localextrema!(similar(A), similar(A), A, args...)

function localextrema!(Amin::AbstractArray{T,N},
                       Amax::AbstractArray{T,N},
                       A::AbstractArray{T,N}, B=3) where {T,N}
    localextrema!(Amin, Amax, A, convert(Neighborhood{N}, B))
end

@doc @doc(erode) erode!
@doc @doc(erode) dilate
@doc @doc(erode) dilate!
@doc @doc(erode) localextrema
@doc @doc(erode) localextrema!

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

"""
A local filtering operation can be performed by calling:

    localfilter!(dst, A, B, initial, update, store) -> dst

where `dst` is the destination, `A` is the source, `B` defines the
neighborhood, `initial`, `update` and `store` are three functions whose
purposes are explained by the following pseudo-code to implement the local
operation:

    for i ∈ Sup(A)
        v = initial()
        for j ∈ Sup(A) and i - j ∈ Sup(B)
            v = update(v, A[j], B[i-j])
        end
        store(dst, i, v)
    end

where `A` `Sup(A)` yields the support of `A` (that is the set of indices in
`A`) and likely `Sub(B)` for `B`.

For instance, to compute a local minimum (that is an erosion):

    localfilter!(dst, A, B,
                 ()      -> typemax(T),
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

# Include code for basic operations with specific structuring element
# types.
include("centeredboxes.jl")
include("cartesianboxes.jl")
include("kernels.jl")

#------------------------------------------------------------------------------

# To implement variants and out-of-place versions, we define conversion rules
# to convert various types of arguments into a neighborhood suitable with the
# source (e.g., of given rank `N`).

convert(::Type{Neighborhood{N}}, dim::Integer) where {N} =
    CenteredBox(ntuple(i->dim, N))

convert(::Type{Neighborhood{N}}, dims::Vector{T}) where {N,T<:Integer} =
    (@assert length(dims) == N; CenteredBox(dims...))

convert(::Type{Neighborhood{N}}, A::AbstractArray{T,N}) where {T,N} =
    Kernel(A)

function convert(::Type{Neighborhood{N}},
                 R::CartesianRange{CartesianIndex{N}}) where {N}
    CartesianBox(R)
end

function convert(::Type{Neighborhood{N}},
                 inds::NTuple{N,AbstractUnitRange{T}}) where {N,T<:Integer}
    CartesianBox(inds)
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

end
