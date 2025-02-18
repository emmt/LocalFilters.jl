# Linear filters

`LocalFilters` provides a few linear filters: [`localmean`](@ref) or [`localmean!`](@ref)
to compute the mean of values in a neighborhood, and [`convolve`](@ref) or
[`convolve!`](@ref) to compute the discrete convolution of an array by a kernel.


## Local mean

The [`localmean`](@ref) method yields the local mean of an array `A` in a neighborhood
`B`:

```julia
localmean(A, B=3) -> dst
```

The result `dst` is an array similar to `A`. If `B` is not specified, the neighborhood is
a hyper-rectangular moving window of size 3 in every dimension. Otherwise, `B` may be
specified as a Cartesian box, or as an array of booleans of same number of dimensions as
`A`. If `B` is a single odd integer (as it is by default), the neighborhood is assumed to
be a hyper-rectangular moving window of size `B` in every dimension.

To avoid allocations, use the in-place version [`localmean!`](@ref) and call:

```julia
localmean!(dst, A, B=3) -> dst
```

to overwrite `dst` with the local mean of `A` in the neighborhood defined by `B`.


## Convolution

The [`convolve`](@ref) method yields the discrete convolution of an array by a kernel. Its
syntax is:

```julia
convolve(A, B) -> dst
```

to yield the discrete convolution of array `A` by the kernel defined by `B`. The result
`dst` is an array similar to `A`.

Using `Sup(A)` to denote the set of valid indices for array `A` and assuming `B` is an
array of values, the discrete convolution of `A` by `B` writes:

```julia
T = promote_type(eltype(A), eltype(B))
for i ∈ Sup(A)
    v = zero(T)
    @inbounds for k ∈ Sup(B) ∩ (i - Sup(A))
        v += A[i-k]*B[k]
    end
    dst[i] = v
end
```

with `T` the type of the product of elements of `A` and `B`, and where `Sup(B) ∩ (i -
Sup(A))` denotes the subset of indices `k` such that `k ∈ Sup(B)` and `i - k ∈ Sup(A)` and
thus for which `B[k]` and `A[i-k]` are valid.

Following the conventions in [`localfilter!`](@ref), the discrete convolution can also be
expressed as:

```julia
T = promote_type(eltype(A), eltype(B))
for i ∈ Sup(A)
    v = zero(T)
    @inbounds for j ∈ Sup(A) ∩ (i - Sup(B))
        v += A[j]*B[i-j]
    end
    dst[i] = v
end
```

If the kernel `B` is an array of booleans, the discrete convolution is computed as:

```julia
T = eltype(A)
for i ∈ Sup(A)
    v = zero(T)
    for j ∈ Sup(A) ∩ (i - Sup(B))
        if B[i-j]
            v += A[j]
        end
    end
    dst[i] = v
end
```

which amounts to computing the local sum of the values of `A` in the neighborhood defined
by the true entries of `B`.

To avoid allocations, use the in-place version [`convolve!`](@ref) and call:

```julia
convolve!(dst, A, B) -> dst
```

to overwrite `dst` with the discrete convolution of `A` by the kernel `B`.
