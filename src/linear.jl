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
# Copyright (C) 2017-2022, Éric Thiébaut.
#

"""
    localmean(A, [ord=ForwardFilter,] B=3)

yields the local mean of `A` in a neighborhood defined by `B`. The result is an
array similar to `A`. If `B` is not specified, the neighborhood is a
hyperrectangular sliding window of size 3 in every dimension. Otherwise, `B`
may be specified as a Cartesian box, or as an array of booleans of same number
of dimensions as `A`. If `B` is a single odd integer (as it is by default), the
neighborhood is assumed to be a hyperrectangular sliding window of size `B` in
every dimension.

See also [`localmean!`](@ref) and [`localfilter!`](@ref).

""" localmean

"""
    localmean!(dst, A, [ord=ForwardFilter,] B=3) -> dst

overwrites `dst` with the local mean of `A` in a neighborhood defined by `B`
and returns `dst`.

See also [`localmean`](@ref) and [`localfilter!`](@ref).

""" localmean!

# Local mean inside a simple sliding window.
function localmean!(dst::AbstractArray{<:Any,N},
                    A::AbstractArray{<:Any,N},
                    ord::FilterOrdering,
                    B::Box{N}) where {N}
    indices = Indices(dst, A, B)
    @inbounds for i in indices(dst)
        J = localindices(indices(A), ord, indices(B), i)
        den = length(J)
        if den > 0
            num = zero(result_eltype(+, A))
            @simd for j in J
                num += A[j]
            end
            store!(dst, i, num/den)
        else
            store!(dst, i, zero(eltype(dst)))
        end
    end
    return dst
end

# Local mean with a shaped neighborhood.
function localmean!(dst::AbstractArray{<:Any,N},
                    A::AbstractArray{<:Any,N},
                    ord::FilterOrdering,
                    B::AbstractArray{Bool,N}) where {N}
    indices = Indices(dst, A, B)
    @inbounds for i in indices(dst)
        num = zero(result_eltype(+, A))
        den = 0
        J = localindices(indices(A), ord, indices(B), i)
        @simd for j in J
            b = B[ord(i,j)]
            num += ifelse(b, A[j], zero(eltype(A)))
            den += b
        end
        if den != zero(den)
            store!(dst, i, num/den)
        else
            store!(dst, i, zero(eltype(dst)))
        end
    end
    return dst
end

# Weighted local mean.
function localmean!(dst::AbstractArray{<:Any,N},
                    A::AbstractArray{<:Any,N},
                    ord::FilterOrdering,
                    B::AbstractArray{<:Any,N}) where {N}
    indices = Indices(dst, A, B)
    @inbounds for i in indices(dst)
        num = zero(result_eltype(+, A))
        den = zero(result_eltype(+, B))
        J = localindices(indices(A), ord, indices(B), i)
        @simd for j in J
            num += A[j]
            den += B[ord(i,j)]
        end
        if den != zero(den)
            store!(dst, i, num/den)
        else
            store!(dst, i, zero(eltype(dst)))
        end
    end
    return dst
end

"""
    correlate(A, B=3) -> dst

yields the discrete correlation of the array `A` by the kernel defined by `B`.
The result `dst` is an array similar to `A`.

Using `Sup(A)` to denote the set of valid indices for array `A` and assuming
`B` is an array of numerical values, the discrete convolution of `A` by `B`
writes:

    T = promote_type(eltype(A), eltype(B))
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

See also [`correlate!`](@ref), [`convolve`](@ref), and
[`LocalFilters.multiply_add`](@ref).

""" correlate

"""
    correlate!(dst, A, B) -> dst

overwrites `dst` with the discrete convolution of `A` by the kernel `B` and
returns `dst`.

See also [`correlate`](@ref) and [`localfilter!`](@ref).

""" correlate!

"""
    convolve(A, B=3)

yields the discrete convolution of array `A` by the kernel defined by `B`. The
result `dst` is an array similar to `A`.

Using `Sup(A)` to denote the set of valid indices for array `A` and assuming
`B` is an array of values, the discrete convolution of `A` by `B` writes:

    T = promote_type(eltype(A), eltype(B))
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

"""
    multiply_add(A, [ord=ForwardFilter,] B=3) -> dst

yields the discrete correlation (if `ord=ForwardFilter`) or the discrete
convolution (if `ord=ReverseFilter`) of `A` by `B`. The result is an array
similar to `A`.

See also [`multiply_add!`](@ref), [`correlate`](@ref), [`convolve`](@ref), and
[`localfilter!`](@ref).

""" multiply_add

"""
    multiply_add!(dst, A, [ord=ForwardFilter,] B=3) -> dst

overwrites `dst` with the discrete correlation (if `ord=ForwardFilter`) or the
discrete convolution (if `ord=ReverseFilter`) of `A` by `B`.

See also [`multiply_add`](@ref) and [`localfilter!`](@ref).

""" multiply_add!

# Local sum inside a simple sliding window.
function multiply_add!(dst::AbstractArray{<:Any,N},
                       A::AbstractArray{<:Any,N},
                       ord::FilterOrdering,
                       B::Box{N}) where {N}
    indices = Indices(dst, A, B)
    @inbounds for i in indices(dst)
        J = localindices(indices(A), ord, indices(B), i)
        v = zero(result_eltype(+, A))
        @simd for j in J
            v += A[j]
        end
        store!(dst, i, v)
    end
    return dst
end

# Local sum with a shaped neighborhood.
function multiply_add!(dst::AbstractArray{<:Any,N},
                       A::AbstractArray{<:Any,N},
                       ord::FilterOrdering,
                       B::AbstractArray{Bool,N}) where {N}
    indices = Indices(dst, A, B)
    @inbounds for i in indices(dst)
        J = localindices(indices(A), ord, indices(B), i)
        v = zero(result_eltype(+, A))
        @simd for j in J
            v += mult(A[j], B[ord(i,j)])
        end
        store!(dst, i, v)
    end
    return dst
end

# Compared to the base implementation in `bool.jl`, the following definition of
# the multiplication by a boolean yields a significantly faster (~50%)
# `multiply_add!` for big neighborhoods because `copysign` is avoided.
mult(a::Number, b::Bool) = ifelse(b, a, zero(a))
mult(a::Bool, b::Number) = mult(b, a)
mult(a::Bool, b::Bool) = a*b
mult(a, b) = a*b

# Correlation/convolution.
function multiply_add!(dst::AbstractArray{<:Any,N},
                       A::AbstractArray{<:Any,N},
                       ord::FilterOrdering,
                       B::AbstractArray{<:Any,N}) where {N}
    indices = Indices(dst, A, B)
    @inbounds for i in indices(dst)
        J = localindices(indices(A), ord, indices(B), i)
        v = zero(promote_type(eltype(A), eltype(B)))
        @simd for j in J
            v += A[j]*B[ord(i,j)]
        end
        store!(dst, i, v)
    end
    return dst
end

for f in (:localmean, :multiply_add)
    f! = Symbol(f,:(!))
    @eval begin
        # These versions provides a default ordering.
        function $f(A::AbstractArray{<:Any,N},
                    B::Union{Window{N},AbstractArray{<:Any,N}} = 3) where {N}
            return $f(A, ForwardFilter, B)
        end
        function $f!(dst::AbstractArray{<:Any,N},
                     A::AbstractArray{<:Any,N},
                     B::Union{Window{N},AbstractArray{<:Any,N}} = 3) where {N}
            return $f!(dst, A, ForwardFilter, B)
        end

        # These versions builds a kernel.
        function $f(A::AbstractArray{<:Any,N},
                    ord::FilterOrdering,
                    B::Window{N} = 3) where {N}
            return $f(A, ord, kernel(Dims{N}, B))
        end
        function $f!(dst::AbstractArray{<:Any,N},
                     A::AbstractArray{<:Any,N},
                     ord::FilterOrdering,
                     B::Window{N} = 3) where {N}
            return $f!(dst, A, ord, kernel(Dims{N}, B))
        end

        # This version provides the destination array.
        function $f(A::AbstractArray{<:Any,N},
                    ord::FilterOrdering,
                    B::AbstractArray{<:Any,N}) where {N}
            return $f!(similar(A, result_eltype($f, A, B)), A, ord, B)
        end
    end
end

for (f, ord) in ((:correlate, :ForwardFilter),
                 (:convolve,  :ReverseFilter))
    f! = Symbol(f,:(!))
    @eval begin
        $f(A::AbstractArray) = multiply_add(A, $ord)
        function $f(A::AbstractArray{<:Any,N},
                    B::Union{Window{N},AbstractArray{<:Any,N}}) where {N}
            return multiply_add(A, $ord, B)
        end
        function $f!(dst::AbstractArray{<:Any,N},
                     A::AbstractArray{<:Any,N}) where {N}
            return multiply_add!(dst, A, $ord)
        end

        function $f!(dst::AbstractArray{<:Any,N},
                     A::AbstractArray{<:Any,N},
                     B::Union{Window{N},AbstractArray{<:Any,N}}) where {N}
            return multiply_add!(dst, A, $ord, B)
        end
    end
end

# Extend `result_eltype` for `localmean`.
function result_eltype(::typeof(localmean),
                       A::AbstractArray{<:Any,N},
                       B::Union{CartesianIndices{N},
                                AbstractArray{Bool,N}}) where {N}
    return float(eltype(A))
end

function result_eltype(::typeof(localmean),
                       A::AbstractArray{<:Any,N},
                       B::AbstractArray{<:Any,N}) where {N}
    return float(promote_type(eltype(A), eltype(B)))
end

# Extend `result_eltype` for `multiply_add`.
function result_eltype(::typeof(multiply_add),
                       A::AbstractArray{<:Any,N},
                       B::Union{CartesianIndices{N},
                                AbstractArray{Bool,N}}) where {N}
    return result_eltype(+, A)
end

function result_eltype(::typeof(multiply_add),
                       A::AbstractArray{<:Any,N},
                       B::AbstractArray{<:Any,N}) where {N}
    return promote_type(eltype(A), eltype(B))
end
