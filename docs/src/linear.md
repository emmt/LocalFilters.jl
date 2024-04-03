# Linear filters

`LocalFilters` provides a few linear filters: [`localmean`](@ref) or
[`localmean!`](@ref) to compute the mean of values in a neighborhood,
[`convolve`](@ref) or [`convolve!`](@ref) to compute the discrete convolution
of an array by a kernel, and [`correlate`](@ref) or [`correlate!`](@ref) to
compute the discrete correlation of an array by a kernel.


## Local mean

The [`localmean`](@ref) method yields the local mean of an array `A` in a
neighborhood `B`:

```julia
dst = localmean(A, B=3)
```

The result `dst` is an array similar to `A`. See Section *[Simple rules for
specifying neighborhoods and kernels](@ref)* for the interpretation of `B`.

The in-place version [`localmean!`](@ref) may be used to avoid allocations:

```julia
localmean!(dst, A, B=3)
```

which overwrites `dst` with the local mean of `A` in the neighborhood defined
by `B` and returns `dst`.


## Discrete convolution

Call the [`convolve`](@ref) method as:

```julia
dst = convolve(A, B)
```

to compute the discrete convolution of array `A` by the kernel defined by `B`.
The result `dst` is an array similar to `A`.

Using `indices(A)` to denote the set of valid indices for array `A` and
assuming `B` is an array of values, the discrete convolution of `A` by `B`
writes (see Section *[Discrete convolution and correlation](@ref)*):

```julia
for i ∈ indices(A)
    v = zero(T)
    @inbounds for k ∈ indices(B) ∩ (i - indices(A))
        v += A[i-k]*B[k]
    end
    dst[i] = v
end
```

with `T` a suitable element type for the result (see Section *[Element type of
the result](@ref)* below) and where `indices(B) ∩ (i - indices(A))` denotes the
subset of indices `k` such that `k ∈ indices(B)` and `i - k ∈ indices(A)` and
thus for which `B[k]` and `A[i-k]` are valid.

Following the conventions in [`localfilter!`](@ref), the discrete convolution
can also be expressed as:

```julia
for i ∈ indices(A)
    v = zero(T)
    @inbounds for j ∈ indices(A) ∩ (i - indices(B))
        v += A[j]*B[i-j]
    end
    dst[i] = v
end
```

If the kernel `B` is an array of Booleans, the discrete convolution is computed
as:

```julia
for i ∈ indices(A)
    v = zero(T)
    @inbounds for j ∈ indices(A) ∩ (i - indices(B))
        if B[i-j]
            v += A[j]
        end
    end
    dst[i] = v
end
```

which amounts to computing the local sum of the values of `A` in the
neighborhood defined by the true entries of `B`.

The in-place version [`convolve!`](@ref) may be used to avoid allocations:

```julia
convolve!(dst, A, B)
```

which overwrites `dst` with the discrete convolution of `A` by the kernel `B`
and returns `dst`.


## Discrete correlation

Call the [`correlate`](@ref) method as:

```julia
dst = correlate(A, B)
```

to compute the discrete correlation of array `A` by the kernel defined by `B`.
The result `dst` is an array similar to `A`.

Using `indices(A)` to denote the set of valid indices for array `A` and
assuming `B` is an array of values, the discrete correlation of `A` by `B`
writes (see Section *[Discrete convolution and correlation](@ref)*):

```julia
for i ∈ indices(A)
    v = zero(T)
    @inbounds for k ∈ indices(B) ∩ (indices(A) - i)
        v += A[i+k]*conj(B[k])
    end
    dst[i] = v
end
```

with `T` a suitable element type for the result (see Section *[Element type of
the result](@ref)* below) and where `indices(B) ∩ (indices(A) - i)` denotes the
subset of indices `k` such that `k ∈ indices(B)` and `i + k ∈ indices(A)` and
thus for which `B[k]` and `A[i+k]` are valid.

Following the conventions in [`localfilter!`](@ref), the discrete correlation
can also be expressed as:

```julia
for i ∈ indices(A)
    v = zero(T)
    @inbounds for j ∈ indices(A) ∩ (indices(B) + i)
        v += A[j]*conj(B[j-i])
    end
    dst[i] = v
end
```

If the kernel `B` is an array of Booleans, the discrete correlation is computed
as:

```julia
for i ∈ indices(A)
    v = zero(T)
    @inbounds for j ∈ indices(A) ∩ (indices(B) + i)
        v += A[j]
    end
    dst[i] = v
end
```

which amounts to computing the local sum of the values of `A` in the
neighborhood defined by the true entries of `B`.

The in-place version [`correlate!`](@ref) may be used to avoid allocations:

```julia
correlate!(dst, A, B)
```

which overwrites `dst` with the discrete correlation of `A` by the kernel `B`
and returns `dst`.

SInce accessing the indices of `A` and `B` in the same order is generally
faster (e.g. it is easier to optimize via loop vectorization), the discrete
convolution `convolve(A,B)` of `A` by `B` may be computed by:

```julia
correlate(A, reverse_kernel(B))
```

provided the entries of `B` are reals, not complexes.


## Element type of the result

Choosing a suitable element type for the result may be tricky if the entries of
the source array `A` and of the kernel `B` have different types or have units.

For example, a suitable element type `T` for the result of the convolution or
correlation of `A` by `B` is given by:

```julia
T = let a = oneunit(eltype(A)), b = oneunit(eltype(B)), c = a*b
    typeof(c + c)
end
```

which is the type of the sum of the element-wise product of the entries of `A`
and `B`.

For the local mean, a similar reasoning yields:

```julia
T = let a = oneunit(eltype(A)), b = oneunit(eltype(B)), c = a*b
    typeof((c + c)/(b + b))
end
```

which is the type of the sum of the element-wise product of the entries of `A`
and `B` divided by the sum of the entries in `B` (the so-called weights).

These rules are the ones used by the out-of-place versions of the linear
filters of `LocalFilter` when the destination is not provided.
