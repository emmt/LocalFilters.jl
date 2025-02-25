#
# generic.jl --
#
# Generic methods for local filters.
#
#-----------------------------------------------------------------------------------------
#
# This file is part of the `LocalFilters.jl` package licensed under the MIT "Expat"
# License.
#
# Copyright (C) 2017-2025, Éric Thiébaut.
#

"""
    localfilter(A, args...; kwds...) -> dst

out of place version of [`localfilter!`](@ref) which is equivalent to:

    localfilter!(similar(A), A, args...; kwds...)

"""
localfilter(A::AbstractArray, args...; kwds...) =
    localfilter!(similar(A), A, args...; kwds...)

"""
    localfilter!(dst, A, [ord = FORWARD_FILTER,] B, initial,
                 update::Function, final::Function = identity) -> dst

overwrites the destination `dst` with the result of a local filter applied to the source
`A`, on a relative neighborhood defined by `B`, and implemented by `initial`, `update`,
and `final`. The `initial` argument may be a function or not. The purpose of these latter
arguments is explained by the following pseudo-codes implementing the local filtering. If
`ord = FORWARD_FILTER`:

    @inbounds for i ∈ indices(dst)
        v = initial isa Function ? initial(A[i]) : initial
        for j ∈ indices(A) ∩ (indices(B) + i)
            v = update(v, A[j], B[j-i])
        end
        dst[i] = final(v)
    end

else if `ord = REVERSE_FILTER`:

    @inbounds for i ∈ indices(dst)
        v = initial isa Function ? initial(A[i]) : initial
        for j ∈ indices(A) ∩ (i - indices(B))
            v = update(v, A[j], B[i-j])
        end
        dst[i] = final(v)
    end

where `indices(A)` denotes the range of indices of any array `A` while `indices(B) + i`
and `i - indices(B)` respectively denote the set of indices `j` such that `j - i ∈
indices(B)` and `i - j ∈ indices(B)`. In other words, `j ∈ indices(A) ∩ (i - indices(B))`
means all indices `j` such that `j ∈ indices(A)` and `i - j ∈ indices(B)` so that `A[j]`
and `B[i-j]` are in-bounds.

If `initial` is a function, the initial value of the state variable `v` in the above
pseudo-codes is given by `v = initial(A[i])` with `i` the current index in `dst`. Hence,
in that case, the destination array `dst` and the source array `src` must have the same
axes.

For example, implementing a local minimum filter (that is, an *erosion*), is as simple as:

    localfilter!(dst, A, ord, B, typemax(eltype(A)),
                 (v,a,b) -> ifelse(b, min(v,a), v))

As another example, implementing a convolution by `B` writes:

    T = promote_type(eltype(A), eltype(B))
    localfilter!(dst, A, ord, B, zero(T), (v,a,b) -> v + a*b)

""" localfilter!

# This version provides a default ordering.
function localfilter!(dst::AbstractArray{<:Any,N},
                      A::AbstractArray{<:Any,N},
                      B::Union{Window{N},AbstractArray{<:Any,N}},
                      initial,
                      update::Function,
                      final::Function = identity) where {N}
    return localfilter!(dst, A, FORWARD_FILTER, B, initial, update, final)
end

# This version builds a kernel.
function localfilter!(dst::AbstractArray{<:Any,N},
                      A::AbstractArray{<:Any,N},
                      ord::FilterOrdering,
                      B::Window{N},
                      initial,
                      update::Function,
                      final::Function = identity) where {N}
    return localfilter!(dst, A, ord, kernel(Dims{N}, B), initial, update, final)
end

@inline function localfilter!(dst::AbstractArray{<:Any,N},
                              A::AbstractArray{<:Any,N},
                              ord::FilterOrdering,
                              B::AbstractArray{<:Any,N},
                              initial,
                              update::Function,
                              final::Function = identity) where {N}
    !(initial isa Function) || axes(dst) == axes(A) || throw(DimensionMismatch(
        "destination and source arrays must have the same axes when `initial` is a function"))
    indices = Indices(dst, A, B)
    @inbounds for i in indices(dst)
        v = initial isa Function ? initial(A[i]) : initial
        @simd for j in localindices(indices(A), ord, indices(B), i)
            v = update(v, A[j], B[ord(i,j)])
        end
        dst[i] = final(v)
    end
    return dst
end

"""
    localfilter!(dst, A, [ord=FORWARD_FILTER,] B, filter!) -> dst

overwrites `dst` with the result of filtering the source `A` by the kernel specified by
`B`, with ordering `ord`, and the function `filter!` which is called as:

    filter!(dst, A, ord, ker, i, J)

for every index `i` of `dst` and with `J` the subset of indices in the local neighborhood
of `i` and:

    ker = kernel(Dims{ndims(A)}, B)

the array representing the filter kernel. The function `filter!` shall compute the result
of the local filtering operation and store it in the destination `dst` at position `i`.

- If `ord = FORWARD_FILTER`, then `J` is the subset of all indices `j` such that `A[j]`
  and `B[j-i]` are in-bounds. This is the natural ordering to implement discrete
  correlations.

- If `ord = REVERSE_FILTER`, then `J` is the subset of all indices `j` such that `A[j]`
  and `B[i-j]` are in-bounds. This is the natural ordering to implement discrete
  convolutions.

To be agnostic to the ordering, just use `B[ord(i,j)]` in the code of `filter!` to
automatically yield either `B[j-i]` or `B[i-j]` depending on whether `ord` is
`FORWARD_FILTER` or `REVERSE_FILTER`.

"""
function localfilter!(dst::AbstractArray{<:Any,N},
                      A::AbstractArray{<:Any,N},
                      B::Union{Window{N},AbstractArray{<:Any,N}},
                      filter!::Function) where {N}
    # Provide default ordering.
    return localfilter!(dst, A, FORWARD_FILTER, B, filter!)
end

function localfilter!(dst::AbstractArray{<:Any,N},
                      A::AbstractArray{<:Any,N},
                      ord::FilterOrdering,
                      B::Window{N},
                      filter!::Function) where {N}
    # Build kernel.
    return localfilter!(dst, A, ord, kernel(Dims{N}, B), filter!)
end

function localfilter!(dst::AbstractArray{<:Any,N},
                      A::AbstractArray{<:Any,N},
                      ord::FilterOrdering,
                      B::AbstractArray{<:Any,N},
                      filter!::Function) where {N}
    indices = Indices(dst, A, B)
    @inbounds for i in indices(dst)
        J = localindices(indices(A), ord, indices(B), i)
        filter!(dst, A, ord, B, i, J)
    end
    return dst
end
