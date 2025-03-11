"""
    localmean(A, B=3; null=zero(T), order=FORWARD_FILTER)

yields the local mean of `A` in a neighborhood defined by `B`. The result is an array
similar to `A`. If `B` is not specified, the neighborhood is a hyper-rectangular sliding
window of size 3 in every dimension. Otherwise, `B` may be specified as a Cartesian box,
or as an array of booleans of same number of dimensions as `A`. If `B` is a single odd
integer (as it is by default), the neighborhood is assumed to be a hyper-rectangular
sliding window of size `B` in every dimension.

Keyword `null` may be used to specify the value of the result where the sum of the weights
in a local neighborhood is zero. By default, `null = zero(T)` with `T` the element type of
the result which may be specified with keyword `eltype`.

Keyword `order` specifies the filter direction, `FORWARD_FILTER` by default.

See also [`localmean!`](@ref) and [`localfilter!`](@ref).

"""
localmean(A::AbstractArray{<:Any,N}, B::Kernel{N} = 3; kwds...) where {N} =
    localmean(A, kernel(Dims{N}, B); kwds...)

function localmean(A::AbstractArray{<:Any,N}, B::AbstractArray{<:Any,N};
                   eltype::Type = typeof_mean(eltype(A), eltype(B)), kwds...) where {N}
    return localmean!(similar(A, eltype), A, B; kwds...)
end

"""
    localmean!(dst, A, B=3; null=zero(eltype(dst)), order=FORWARD_FILTER) -> dst

overwrites `dst` with the local mean of `A` in a neighborhood defined by `B` and returns
`dst`.

Keyword `null` may be used to specify the value of the result where the sum of the weights
in the a neighborhood is zero.

Keyword `order` specifies the filter direction, `FORWARD_FILTER` by default.

See also [`localmean`](@ref) and [`localfilter!`](@ref).

"""
function localmean!(dst::AbstractArray{<:Any,N}, A::AbstractArray{<:Any,N},
                    B::Kernel{N} = 3; kwds...) where {N}
    return localmean!(dst, A, kernel(Dims{N}, B); kwds...)
end

# Local mean inside a simple sliding window.
function localmean!(dst::AbstractArray{<:Any,N}, A::AbstractArray{<:Any,N}, B::Box{N};
                    null = zero(eltype(dst)),
                    order::FilterOrdering = FORWARD_FILTER) where {N}
    null = as(eltype(dst), null)
    indices = Indices(dst, A, B)
    T_num = typeof_sum(eltype(A))
    @inbounds for i in indices(dst)
        J = localindices(indices(A), order, indices(B), i)
        den = length(J)
        if den > 0
            num = zero(T_num)
            @simd for j in J
                num += A[j]
            end
            store!(dst, i, _div(num, den))
        else
            dst[i] = null
        end
    end
    return dst
end

# Local mean with a shaped neighborhood or weighted local mean.
function localmean!(dst::AbstractArray{<:Any,N},
                    A::AbstractArray{<:Any,N},
                    B::AbstractArray{<:Any,N};
                    null = zero(eltype(dst)),
                    order::FilterOrdering = FORWARD_FILTER) where {N}
    null = as(eltype(dst), null)
    indices = Indices(dst, A, B)
    T_num = typeof_sumprod(eltype(A), eltype(B))
    T_den = typeof_sum(eltype(B))
    @inbounds for i in indices(dst)
        num = zero(T_num)
        den = zero(T_den)
        J = localindices(indices(A), order, indices(B), i)
        @simd for j in J
            wgt = B[order(i,j)]
            num += _mul(A[j], wgt)
            den += wgt
        end
        if !iszero(den)
            store!(dst, i, _div(num, den))
        else
            dst[i] = null
        end
    end
    return dst
end

"""
    correlate(A, B=3) -> dst

yields the discrete correlation of the array `A` by the kernel defined by `B`. The result
`dst` is an array similar to `A`.

Using `Sup(A)` to denote the set of valid indices for array `A` and assuming `B` is an
array of numerical values, the discrete convolution of `A` by `B` writes:

    dst = similar(A, T)
    for i ∈ Sup(A)
        v = zero(T)
        @inbounds for j ∈ Sup(A) ∩ (Sup(B) - i)
            v += A[j]*B[j-i]
        end
        dst[i] = v
    end

with `T` the type of the sum of the products of the elements of `A` and `B`, and where
`Sup(A) ∩ (i - Sup(A))` denotes the subset of indices `k` such that `k ∈ Sup(B)` and `i -
k ∈ Sup(A)` and thus for which `B[k]` and `A[i-k]` are valid.

See also [`correlate!`](@ref) and [`convolve`](@ref).

""" correlate

"""
    correlate!(dst, A, B) -> dst

overwrites `dst` with the discrete convolution of `A` by the kernel `B` and returns `dst`.

See also [`correlate`](@ref) and [`localfilter!`](@ref).

""" correlate!

"""
    convolve(A, B=3)

yields the discrete convolution of array `A` by the kernel defined by `B`. The result
`dst` is an array similar to `A`.

Using `Sup(A)` to denote the set of valid indices for array `A` and assuming `B` is an
array of values, the discrete convolution of `A` by `B` writes:

    for i ∈ Sup(A)
        v = zero(T)
        @inbounds for j ∈ Sup(B) ∩ (i - Sup(A))
            v += A[i-j]*B[j]
        end
        dst[i] = v
    end

with `T` the type of the sum of the products of the elements of `A` and `B`, and where
`Sup(B) ∩ (i - Sup(A))` denotes the subset of indices `k` such that `k ∈ Sup(B)` and `i -
k ∈ Sup(A)` and thus for which `B[k]` and `A[i-k]` are valid.

See also [`convolve!`](@ref) and [`localfilter!`](@ref).

""" convolve

"""
    convolve!(dst, A, B) -> dst

overwrites `dst` with the discrete convolution of `A` by the kernel `B` and returns `dst`.

See also [`convolve`](@ref) and [`localfilter!`](@ref).

""" convolve!

# Provide destination.
function sumprod(A::AbstractArray{<:Any,N}, B::AbstractArray{<:Any,N};
                 eltype::Type = typeof_sumprod(eltype(A), eltype(B)), kwds...) where {N}
    return sumprod!(similar(A, eltype), A, B; kwds...)
end

# Local sum inside a simple sliding window.
function sumprod!(dst::AbstractArray{<:Any,N}, A::AbstractArray{<:Any,N}, B::Box{N};
                  order::FilterOrdering = FORWARD_FILTER) where {N}
    indices = Indices(dst, A, B)
    T = typeof_sum(eltype(A))
    @inbounds for i in indices(dst)
        v = zero(T)
        @simd for j in localindices(indices(A), order, indices(B), i)
            v += A[j]
        end
        store!(dst, i, v)
    end
    return dst
end

# Correlation/convolution or local sum with a shaped neighborhood.
function sumprod!(dst::AbstractArray{<:Any,N}, A::AbstractArray{<:Any,N},
                  B::AbstractArray{<:Any,N};
                  order::FilterOrdering = FORWARD_FILTER) where {N}
    indices = Indices(dst, A, B)
    T = typeof_sumprod(eltype(A), eltype(B))
    @inbounds for i in indices(dst)
        v = zero(T)
        @simd for j in localindices(indices(A), order, indices(B), i)
            v += _mul(A[j], B[order(i,j)])
        end
        store!(dst, i, v)
    end
    return dst
end

for (f, order) in ((:correlate, :FORWARD_FILTER),
                   (:convolve,  :REVERSE_FILTER))
    f! = Symbol(f,"!")
    @eval begin
        $f(A::AbstractArray{<:Any,N}, B::Kernel{N}; kwds...) where {N} =
            sumprod(A, kernel(Dims{N}, B); order = $order, kwds...)
        $f!(dst::AbstractArray{<:Any,N}, A::AbstractArray{<:Any,N}, B::Kernel{N}; kwds...) where {N} =
            sumprod!(dst, A, kernel(Dims{N}, B); order = $order, kwds...)
    end
end

# `sum_init(T)` yields the initial value of the sum of elements of type `T`.
# The reasoning is that the sum of `n` identical values `x` is given by `n*x` (with
# `n` an `Int`).
sum_init(::Type{T}) where {T} = zero(T)*one(Int)

# `sumprod_init(A, B)` yields the initial value of the sum of products of elements of type
# `A` and `B`.
sumprod_init(::Type{A}, ::Type{B}) where {A,B} = _mul(zero(A), zero(B))*one(Int)

# Yield the type of the sum of terms of a given type.
@inline typeof_sum(::Type{T}) where {T} = typeof(sum_init(T))

# Yield the type of the sum of the product of terms of given types.
@inline typeof_sumprod(::Type{A}, ::Type{B}) where {A,B} = typeof(sumprod_init(A, B))

# Yield the type of a local, possibly weighted, mean. `A` is the type of the data, `B` is
# the type of the weights.
@inline typeof_mean(::Type{A}, ::Type{B}) where {A,B} =
    typeof(_div(oneunit(typeof_sumprod(A, B)), oneunit(typeof_sum(B))))

# See `base/reduce.jl`.
const SmallSigned = Union{filter(T -> sizeof(T) < sizeof(Int),
                                 [Int8, Int16, Int32, Int64, Int128])...,}
const SmallUnsigned = Union{filter(T -> sizeof(T) < sizeof(UInt),
                                   [UInt8, UInt16, UInt32, UInt64, UInt128])...,}
const SmallInteger = Union{SmallSigned, SmallUnsigned}

# Addition and multiplication of terms as assumed by linear filters. To avoid overflows,
# small integers are promoted to a wider type following the same rules as in `sum` or
# `prod` except that we want to preserve the signedness.
for (f, op) in ((:_add, :(+)), (:_mul, :(*)))
    @eval begin
        $f(x::SmallUnsigned, y::SmallUnsigned) = $op(UInt(x), UInt(y))
        $f(x::SmallInteger, y::SmallInteger) = $op(Int(x), Int(y))
        $f(x::Real, y::Real)::Real = $op(x, y)
        $f(x::Any, y::Any) = $op(x, y)
    end
end

# Compared to the base implementation in `bool.jl`, the following definition of the
# multiplication by a Boolean yields a significantly faster (~50%) `sumprod!` for big
# neighborhoods because `copysign` is avoided.
_mul(x::Real, y::Bool) = _mul(y, x)
_mul(x::Any,  y::Bool) = _mul(y, x)
_mul(x::Bool, y::Bool) = Int(x & y)
_mul(x::Bool, y::Real) = ifelse(x, y, zero(y))
_mul(x::Bool, y::Any ) = ifelse(x, y, zero(y))
_mul(x::Bool, y::SmallSigned  ) = _mul(x,  Int(y))
_mul(x::Bool, y::SmallUnsigned) = _mul(x, UInt(y))

# Division of terms as assumed by linear filters.
_div(x::Any, y::Any) = x/y
