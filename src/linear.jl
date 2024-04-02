#
# linear.jl --
#
# Implementation of linear local filters.
#
#------------------------------------------------------------------------------
#
# This file is part of the `LocalFilters.jl` package licensed under the MIT
# "Expat" License.
#
# Copyright (c) 2017-2024, Éric Thiébaut.
#

"""
    localmean(A, [ord=ForwardFilter,] B=3; null=zero(eltype(A)))

yields the local mean of `A` in a neighborhood defined by `B`. The result is an
array similar to `A`. If `B` is not specified, the neighborhood is a
hyper-rectangular sliding window of size 3 in every dimension. Otherwise, `B`
may be specified as a Cartesian box, or as an array of booleans of same number
of dimensions as `A`. If `B` is a single odd integer (as it is by default), the
neighborhood is assumed to be a hyper-rectangular sliding window of size `B` in
every dimension.

Keyword `null` may be used to specify the value of the result where the sum of
the weights in a local neighborhood is zero.

See also [`localmean!`](@ref) and [`localfilter!`](@ref).

"""
function localmean(A::AbstractArray{<:Any,N},
                   B::Union{Window{N},AbstractArray{<:Any,N}} = 3) where {N}
    # Provides default ordering.
    return localmean(A, ForwardFilter, B)
end

function localmean(A::AbstractArray{<:Any,N},
                   ord::FilterOrdering,
                   B::Window{N} = 3) where {N}
    # Make `B` into a kernel array.
    return localmean(A, ord, kernel(Dims{N}, B))
end

function localmean(A::AbstractArray{<:Any,N},
                   ord::FilterOrdering,
                   B::AbstractArray{<:Any,N}) where {N}
    # Provide the destination array.
    T = mean_type(eltype(A), eltype(B))
    return localmean!(similar(A, T), A, ord, B)
end

"""
    localmean!(dst, A, [ord=ForwardFilter,] B=3; null=zero(eltype(dst))) -> dst

overwrites `dst` with the local mean of `A` in a neighborhood defined by `B`
and returns `dst`.

Keyword `null` may be used to specify the value of the result where the sum of
the weights in the a neighborhood is zero.

See also [`localmean`](@ref) and [`localfilter!`](@ref).

"""
function localmean!(dst::AbstractArray{<:Any,N},
                    A::AbstractArray{<:Any,N},
                    B::Union{Window{N},AbstractArray{<:Any,N}} = 3) where {N}
    # Provide default ordering.
    return localmean!(dst, A, ForwardFilter, B)
end

function localmean!(dst::AbstractArray{<:Any,N},
                    A::AbstractArray{<:Any,N},
                    ord::FilterOrdering,
                    B::Window{N} = 3) where {N}
    # Make `B` into a kernel array.
    return localmean!(dst, A, ord, kernel(Dims{N}, B))
end

# Local mean inside a simple sliding window.
function localmean!(dst::AbstractArray{<:Any,N},
                    A::AbstractArray{<:Any,N},
                    ord::FilterOrdering,
                    B::Box{N};
                    null = zero(eltype(dst))) where {N}
    null = nearest(eltype(dst), null)
    indices = Indices(dst, A, B)
    T_num = sum_type(eltype(A))
    @inbounds for i in indices(dst)
        J = localindices(indices(A), ord, indices(B), i)
        den = length(J)
        if den > 0
            num = zero(T_num)
            @simd for j in J
                num += A[j]
            end
            dst[i] = nearest(eltype(dst), _div(num, den))
        else
            dst[i] = null
        end
    end
    return dst
end

# Local mean with a shaped neighborhood or weighted local mean.
function localmean!(dst::AbstractArray{<:Any,N},
                    A::AbstractArray{<:Any,N},
                    ord::FilterOrdering,
                    B::AbstractArray{<:Any,N};
                    null = zero(eltype(dst))) where {N}
    null = nearest(eltype(dst), null)
    indices = Indices(dst, A, B)
    T_num = sum_prod_type(eltype(A), eltype(B))
    T_den = sum_type(eltype(B))
    @inbounds for i in indices(dst)
        num = zero(T_num)
        den = zero(T_den)
        J = localindices(indices(A), ord, indices(B), i)
        @simd for j in J
            wgt = B[ord(i,j)]
            num += _mul(A[j], wgt)
            den += wgt
        end
        if !iszero(den)
            dst[i] = nearest(eltype(dst), _div(num, den))
        else
            dst[i] = null
        end
    end
    return dst
end

# Yield the type of the sum of terms of a given type.
function sum_type(::Type{A}) where {A}
    x = oneunit(A)
    return typeof(_add(x, x))
end

# Yield the type of the sum of the product of terms of given types.
function sum_prod_type(::Type{A}, ::Type{B}) where {A,B}
    x = _mul(oneunit(A), oneunit(B))
    return typeof(_add(x, x))
end

# Yield the type of a local, possibly weighted, mean.
function mean_type(::Type{A} #= data type =#,
                   ::Type{B} #= weight type =#) where {A,B}
    a = oneunit(A)
    b = oneunit(B)
    c = _mul(a, b)
    return typeof(_div(_add(c, c), _add(b, b)))
end

# Compared to the base implementation in `bool.jl`, the following definition of
# the multiplication by a boolean yields a significantly faster (~50%)
# `local_sum_prod!` for big neighborhoods because `copysign` is avoided.
_mul(a::Any,  b::Bool) = ifelse(b, a, zero(a))
_mul(a::Bool, b::Any ) = _mul(b, a)
_mul(a::Bool, b::Bool) = a&b
_mul(a::Any,  b::Any ) = a*b

# Addition of terms as assumed by linear filters.
_add(a::Any,  b::Any) = a+b

# division of terms as assumed by linear filters.
_div(a::Any,  b::Any) = a/b

"""
    correlate(A, B) -> dst

yields the discrete correlation of the array `A` by the kernel defined by `B`.
The result `dst` is an array similar to `A`.

Using `Sup(A)` to denote the set of valid indices for array `A` and assuming
`B` is an array of numerical values, the discrete convolution of `A` by `B`
writes:

    T = let x = oneunit(eltype(A))*oneunit(eltype(B)); typeof(x + x); end
    dst = similar(A, T)
    for i ∈ Sup(A)
        v = zero(T)
        @inbounds for j ∈ Sup(A) ∩ (Sup(B) - i)
            v += A[j]*B[j-i]
        end
        dst[i] = v
    end

with `T` the type of the product of elements of `A` and `B`, and where `Sup(A)
∩ (i - Sup(A))` denotes the subset of indices `k` such that `k ∈ Sup(B)` and
`i - k ∈ Sup(A)` and thus for which `B[k]` and `A[i-k]` are valid.

See also [`correlate!`](@ref) and [`convolve`](@ref).

""" correlate

"""
    correlate!(dst, A, B) -> dst

overwrites `dst` with the discrete convolution of `A` by the kernel `B` and
returns `dst`.

See also [`correlate`](@ref) and [`localfilter!`](@ref).

""" correlate!

"""
    convolve(A, B)

yields the discrete convolution of array `A` by the kernel defined by `B`. The
result `dst` is an array similar to `A`.

Using `Sup(A)` to denote the set of valid indices for array `A` and assuming
`B` is an array of values, the discrete convolution of `A` by `B` writes:

    T = let x = oneunit(eltype(A))*oneunit(eltype(B)); typeof(x + x); end
    for i ∈ Sup(A)
        v = zero(T)
        @inbounds for j ∈ Sup(B) ∩ (i - Sup(A))
            v += A[i-j]*B[j]
        end
        dst[i] = v
    end

with `T` the type of the product of elements of `A` and `B`, and where `Sup(B)
∩ (i - Sup(A))` denotes the subset of indices `k` such that `k ∈ Sup(B)` and
`i - k ∈ Sup(A)` and thus for which `B[k]` and `A[i-k]` are valid.

See also [`convolve!`](@ref) and [`localfilter!`](@ref).

""" convolve

"""
    convolve!(dst, A, B) -> dst

overwrites `dst` with the discrete convolution of `A` by the kernel `B` and
returns `dst`.

See also [`convolve`](@ref) and [`localfilter!`](@ref).

""" convolve!

# Provide destination.
function local_sum_prod(A::AbstractArray{<:Any,N},
                        ord::FilterOrdering,
                        B::AbstractArray{<:Any,N}) where {N}
    T = sum_prod_type(eltype(A), eltype(B))
    return local_sum_prod!(similar(A, T), A, ord, B)
end

# Local sum inside a simple sliding window.
function local_sum_prod!(dst::AbstractArray{<:Any,N},
                         A::AbstractArray{<:Any,N},
                         ord::FilterOrdering,
                         B::Box{N}) where {N}
    indices = Indices(dst, A, B)
    T = sum_type(eltype(A))
    @inbounds for i in indices(dst)
        J = localindices(indices(A), ord, indices(B), i)
        v = zero(T)
        @simd for j in J
            v += A[j]
        end
        dst[i] = nearest(eltype(dst), v)
    end
    return dst
end

# Correlation/convolution or local sum with a shaped neighborhood.
function local_sum_prod!(dst::AbstractArray{<:Any,N},
                         A::AbstractArray{<:Any,N},
                         ord::FilterOrdering,
                         B::AbstractArray{<:Any,N}) where {N}
    indices = Indices(dst, A, B)
    T = sum_prod_type(eltype(A), eltype(B))
    @inbounds for i in indices(dst)
        J = localindices(indices(A), ord, indices(B), i)
        v = zero(T)
        @simd for j in J
            v += _mul(A[j], B[ord(i,j)])
        end
        dst[i] = nearest(eltype(dst), v)
    end
    return dst
end

for (f, ord) in ((:correlate, :ForwardFilter),
                 (:convolve,  :ReverseFilter))
    f! = Symbol(f,"!")
    @eval begin
        function $f(A::AbstractArray{<:Any,N},
                    B::Union{Window{N},AbstractArray{<:Any,N}}) where {N}
            return local_sum_prod(A, $ord, kernel(Dims{N}, B))
        end
        function $f!(dst::AbstractArray{<:Any,N},
                     A::AbstractArray{<:Any,N},
                     B::Union{Window{N},AbstractArray{<:Any,N}}) where {N}
            return local_sum_prod!(dst, A, $ord, kernel(Dims{N}, B))
        end
    end
end
