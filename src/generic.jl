#
# generic.jl --
#
# Generic methods for local filters.
#
#------------------------------------------------------------------------------
#
# This file is part of the `LocalFilters.jl` package licensed under the MIT
# "Expat" License.
#
# Copyright (C) 2017-2022, Éric Thiébaut.
#

"""
    LocalFilters.store!(A, I, x)

stores value `x` in array `A` at index `I`, taking care of rounding `x` if it
is of floating-point type while the elements of `A` are integers.  This method
propagates the current in-bounds settings.

"""
@inline @propagate_inbounds function store!(A::AbstractArray{T}, I,
                                            x::AbstractFloat) where {T<:Integer}
    A[I] = round(T, x)
end

@inline @propagate_inbounds function store!(A::AbstractArray, I, x)
    A[I] = x
end

"""
    localfilter!(dst, A, [ord = ForwardFilter,] B, initial, update, store!) -> dst

overwrites the destination `dst` with the result of a local filter applied to
the source `A`, on a relative neighborhood defined by `B`, and implemented by
the functions `initial`, `update`, and `store!`.  The purpose of these functions
is explained by the following pseudo-codes implementing the local filtering.
If `ord = ForwardFilter`:

    @inbounds for i ∈ Sup(A)
        v = initial(dst, A, B)
        for j ∈ Sup(A) ∩ (Sup(B) + i)
            v = update(v, A[j], B[j-i])
        end
        store!(dst, i, v)
    end

else if `ord = ReverseFilter`:

    @inbounds for i ∈ Sup(A)
        v = initial(dst, A, B)
        for j ∈ Sup(A) ∩ (i - Sup(B))
            v = update(v, A[j], B[i-j])
        end
        store!(dst, i, v)
    end

where `Sup(A)` denotes the support of `A` (that is the set of indices in `A`)
while `Sup(B) + i` and `i - Sup(B)` respectively denote the set of indices `j`
such that `j - i ∈ Sup(B)` and `i - j ∈ Sup(B)` with `Sup(B)` the support of
`B`.  In other words, `j ∈ Sup(A) ∩ (i - Sup(B))` means all indices `j` such
that `j ∈ Sup(A)` and `i - j ∈ Sup(B)` so that `A[j]` and `B[i-j]` are
in-bounds.

!!! warning
    The loop(s) in `localfilter!` are performed without bounds checking of the
    destination and it is the caller's responsibility to insure that the
    destination have the correct size.  It is however always possible to write
    `store!` so that it performs bounds checking.

For example, implementing a local minimum filter (that is, an *erosion*), is as
simple as:

    localfilter!(dst, A, ord, B,
                 (d,a,b) -> typemax(eltype(a)),
                 (v,a,b) -> min(v,a),
                 (d,v,i) -> @inbounds(d[i] = v))

As another example, implementing a convolution by `B` writes:

    localfilter!(dst, A, ord, B,
                 (d,a,b) -> zero(promote_type(eltype(a), eltype(b))),
                 (v,a,b) -> v + a*b,
                 (d,v,i) -> @inbounds(d[i] = v))

If argument `initial` is not a function (that is, an instance of `Function`),
it is assumed to be the initial value of the state variable.  The disrete
correlation example can be rewritten as

    localfilter!(dst, A, ord, B,
                 zero(promote_type(eltype(A), eltype(B))),
                 (v,a,b) -> v + a*b,
                 (d,v,i) -> @inbounds(d[i] = v))

!!! note
    If the methods `init`, `update`, and/or `store!` are anonymous functions or
    closures, beware that they should not depend on local variables because
    this may have a strong impact on performances.

""" localfilter!

# This version provides a default ordering.
@inline @propagate_inbounds localfilter!(
    dst,
    A::AbstractArray{<:Any,N},
    B::Union{Window{N},AbstractArray{<:Any,N}},
    initial,
    update::Function,
    store!::Function) where {N} = localfilter!(
        dst, A, ForwardFilter, B, initial, update, store!)

# This version builds a kernel.
@inline @propagate_inbounds localfilter!(
    dst,
    A::AbstractArray{<:Any,N},
    ord::FilterOrdering,
    B::Window{N},
    initial,
    update::Function,
    store!::Function) where {N} = localfilter!(
        dst, A, ord, kernel(Dims{N}, B), initial, update, store!)

# This version converts the initial value into a constant producer.
@inline @propagate_inbounds localfilter!(
    dst,
    A::AbstractArray{<:Any,N},
    ord::FilterOrdering,
    B::AbstractArray{<:Any,N},
    initial,
    update::Function,
    store!::Function) where {N} = localfilter!(
        dst, A, ord, B, ConstantProducer(initial), update, store!)

@inline @propagate_inbounds function localfilter!(dst::AbstractArray{<:Any,N},
                                                  A::AbstractArray{<:Any,N},
                                                  ord::FilterOrdering,
                                                  B::AbstractArray{<:Any,N},
                                                  initial::Function,
                                                  update::Function,
                                                  store!::Function) where {N}
    indices = Indices(dst, A, B)
    for i in indices(dst)
        v = initial(A[i]) # FIXME: this is unsafe, only the bilateral filter needs it?
        @inbounds @simd for j in localindices(indices(A), ord, indices(B), i)
            v = update(v, A[j], B[ord(i,j)])
        end
        store!(dst, i, v)
    end
    return dst
end



# When destination is not an array with the same number of dimensions as the
# source, the outer loop is on the indices of the source.  This may be unsafe,
# so we propaget the "in-bounds" settings and leave to the caller the
# responsibility to store with/without bounds checking.

@inline @propagate_inbounds function localfilter!(dst,
                                                  A::AbstractArray{<:Any,N},
                                                  ord::FilterOrdering,
                                                  B::AbstractArray{<:Any,N},
                                                  initial::Function,
                                                  update::Function,
                                                  store!::Function) where {N}
    indices = Indices(A, B)
    for i in indices(A)
        @inbounds v = initial(A[i])
        @inbounds @simd for j in localindices(indices(A), ord, indices(B), i)
            v = update(v, A[j], B[ord(i,j)])
        end
        store!(dst, i, v)
    end
    return dst
end

"""
    localfilter!(dst, A, [ord=ForwardFilter,] ker, filter!) -> dst

overwrites `dst` with the result of filtering the source `A` by the kernel `ker`,
with ordering `ord`, and the function `filter!` which is called as:

    filter!(dst, A, ord, B, i, J)

for every index `i` of `dst` or of `A` depending whether `dst` has the same
number of dimensions as `A` or not and with `J` the subset of indices in the
local neighborhood of `i` and:

    B = kernel(Dims{N}, ker)

the array representing the filter kernel.  The function `filter!` shall compute
the result of the local filtering operation and store it in the destination
`dst` at position `i`.

- If `ord = ForwardFilter`, then `J` is the subset of all indices `j` such that
  `A[j]` and `B[j-i]` are in-bounds.  This is the natural ordering to implement
  discrete correlations.

- If `ord = ReverseFilter`, then `J` is the subset of all indices `j` such that
  `A[j]` and `B[i-j]` are in-bounds.  This is the natural ordering to implement
  discrete convolutions.

To be agnostic to the ordering, just use `B[ord(i,j)]` in the code of `filter!`
to automatically yield either `B[j-i]` or `B[i-j]` depending on whether `ord`
is `ForwardFilter` or `ReverseFilter`.

"""
@inline function localfilter!(dst,
                              A::AbstractArray{<:Any,N},
                              B::Union{Window{N},AbstractArray{<:Any,N}},
                              filter!::Function) where {N}
    # Provide default ordering.
    return localfilter!(dst, A, ForwardFilter, B, filter!)
end

@inline function localfilter!(dst,
                              A::AbstractArray{<:Any,N},
                              ord::FilterOrdering,
                              B::Window{N},
                              filter!::Function) where {N}
    # Build kernel.
    return localfilter!(dst, A, ForwardFilter, kernel(Dims{N}, B), filter!)
end

@inline function localfilter!(dst::AbstractArray{<:Any,N},
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

# When destination is not an array with the same number of dimensions as the
# source, the outer loop is on the indices of the source.  This may be unsafe.
@inline function localfilter!(dst,
                              A::AbstractArray{<:Any,N},
                              ord::FilterOrdering,
                              B::AbstractArray{<:Any,N},
                              filter!::Function) where {N}
    indices = Indices(A, B)
    @inbounds for i in indices(A)
        J = localindices(indices(A), ord, indices(B), i)
        filter!(dst, A, ord, B, i, J)
    end
    return dst
end
