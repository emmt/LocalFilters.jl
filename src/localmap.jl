"""
    localmap(f, A, B=3; eltype=eltype(A), null=zero(eltype), order=FORWARD_FILTER)

for each position in `A`, applies the function `f` to the values of `A` extracted from the
neighborhood defined by `B`.

Keyword `eltype` may be used to specify the the element type `T` of the result.
By default, it is the same as that of `A`.

The function `f` is never called with an empty vector of values. Keyword `null` may be
used to specify the value of the result where the neighborhood is empty. By default, `null
= zero(T)` with `T` the element type of the result.

Keyword `order` specifies the filter direction, `FORWARD_FILTER` by default.

For example, applying a median filter of the 2-dimensional image `img` in a sliding `5×5`
window can be done by:

``` julia
using Statistics
med = localmap(median!, img, 5; eltype=float(Base.eltype(A)), null=NaN)
```

As another example, with argument `f` set to `minimum` or `maximum` respectively yield the
erosion and the dilation of the input array. However [`erode`](@ref) and [`dilate`](@) are
much faster.

"""
function localmap(f::Function, A::AbstractArray{T,N}, B::Window{N};
                  eltype::Type=T, kwds...) where {T,N}
    return localmap!(f, similar(A, eltype), A, B; kwds...)
end

"""
    localmap!(f, dst, A, B=3; null=zero(eltype(dst)), order=FORWARD_FILTER)

set each entry of `dst`, to the result of applying the function `f` to the values of `A`
extracted from the neighborhood defined by `B`.

The function `f` is never called with an empty vector of values. Keyword `null` may be
used to specify the value of the result where the neighborhood is empty. By default, `null
= zero(T)` with `T` the element type of the result.

Keyword `order` specifies the filter direction, `FORWARD_FILTER` by default.

"""
function localmap!(f::Function, dst::AbstractArray{<:Any,N},
                   A::AbstractArray{<:Any,N}, B::Window{N}; kwds...) where {N}
    return localmap!(f, dst, A, kernel(Dims{N}, B); kwds...)
end

function localmap!(f::Function, dst::AbstractArray{<:Any,N},
                   A::AbstractArray{<:Any,N}, B::AbstractArray{Bool,N};
                   work::AbstractVector = Vector{eltype(A)}(undef, count(B)),
                   null = zero(eltype(dst)),
                   order::FilterOrdering = FORWARD_FILTER) where {N}
    maxlen = count(B)
    if maxlen == 0
        fill!(dst, null)
    else
        sizehint!(work, maxlen; shrink=false)
        indices = Indices(dst, A, B)
        if maxlen == length(B)
            # `B` is a simple box.
            indices = Indices(dst, A, B)
            @inbounds for i in indices(dst)
                J = localindices(indices(A), order, indices(B), i)
                len = length(J)
                if len > 0
                    length(work) == len || resize!(work, len)
                    k = 0
                    for j in J
                        work[k += 1] = A[j]
                    end
                    k == len || throw(AssertionError("unexpected count"))
                    dst[i] = f(work)
                else
                    dst[i] = null
                end
            end
        else
            # `B` has "holes".
            indices = Indices(dst, A, B)
            @inbounds for i in indices(dst)
                J = localindices(indices(A), order, indices(B), i)
                len = min(length(J), maxlen)
                if len > 0
                    length(work) ≥ len || resize!(work, len)
                    k = 0
                    for j in J
                        if B[order(i,j)]
                            work[k += 1] = A[j]
                        end
                    end
                    k ≤ len || throw(AssertionError("unexpected count"))
                    len = k
                    if len > 0
                        length(work) == len || resize!(work, len)
                        dst[i] = f(work)
                    end
                end
                if len == 0
                    dst[i] = null
                end
            end
        end
    end
    return dst
end
