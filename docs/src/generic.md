# Generic local filters

Most filters provided by the `LocalFilters` package are implemented by the generic
in-place method [`localfilter!`](@ref) or its out-of-place version [`localfilter](@ref).


## The `localfilter!` method

A local filtering operation can be performed by calling the [`localfilter!`](@ref) method:

```julia
localfilter!(dst, A, B, initial, update, final)
```

where `dst` is the destination, `A` is the source, `B` defines the neighborhood, `initial`
is a function or the initial value of the state variable, `update` is a function to update
the state variable for each entry of the neighborhood, and `final` is a function to yield
the local result of the filter given the final value of the state variable. The
[`localfilter!`](@ref) method returns the destination `dst`.

The purpose of arguments `initial`, `update`, and `final` is explained by the following
pseudo-code implementing the local filtering:

```julia
@inbounds for i ∈ indices(dst)
    v = initial isa Function ? initial(A[i]) : initial
    for j ∈ indices(A) ∩ (indices(B) + i)
        v = update(v, A[j], B[j-i])
    end
    dst[i] = final(v)
end
```

where `indices(A)` denotes the set of indices of `A` while `indices(B) + i` denotes the
set of indices `j` such that `j - i ∈ indices(B)`. In other words, `j ∈ indices(A) ∩
(indices(B) + i)` means all indices `j` such that `j ∈ indices(A)` and `j - i ∈
indices(B)`, hence `A[j]` and `B[j-i]` are both in-bounds. In `localfilter!`, indices `i`
and `j` are Cartesian indices for multi-dimensional arrays, thus `indices(A)` is the
analogous of `CartesianIndices(A)` in Julia in that case. For vectors, indices `i` and `j`
are linear indices.

The behavior of the filter is fully determined by the *neighborhood* `B` (see Section
*[Neighborhoods, structuring elements, and kernels](@ref)*) and by the other arguments to
deal with the state variable `v`:

- `initial` may be a function, in which case the state variable is initially given by `v =
  initial(A[i])`; otherwise, `initial` is assumed to be the initial value of the state
  variable. As a consequence, if `initial` is a function, `dst` and `A` must have the same
  axes.

- `update(v, a, b)` yields the updated state variable `v` with `a = A[j]` and `b =
  B[j-i]`.

- `final(v)` yields the result of the filter given the state variable `v` at the end of
  the loop on the neighborhood. If not specified, `final = identity` is assumed.

The `localfilter!` method takes a keyword `order` to specify the *ordering* of the filter.
By default, `order = FORWARD_FILTER` which is implemented by the above pseudo-code and
which corresponds to a **correlation** for a shift-invariant linear filter. The other
possibility is `order = REVERSE_FILTER` which corresponds to a **convolution** for a
shift-invariant linear filter and which amounts to:

```julia
@inbounds for i ∈ indices(dst)
    v = initial isa Function ? initial(A[i]) : initial
    for j ∈ indices(A) ∩ (i - indices(B))
        v = update(v, A[j], B[i-j])
    end
    dst[i] = final(v)
end
```

If `B` is symmetric, in the sense that `B[-j] = B[j]` for any in-bounds `j`, both
orderings yield the same result but `FORWARD_FILTER` is generally faster which is why it
is used by default.


## The `localfilter` method

The [`localfilter`](@ref) method is similar to [`localfilter!`](@ref) except that it allocates
the destination:

```julia
localfilter(A, B, initial, update, final)
```

is the same as:

```julia
localfilter!(similar(A), A, B, initial, update, final)
```

The element type `T` of the destination can however be specified as the first argument:

```julia
localfilter(T, A, B, initial, update, final)
```

which is the same as:

```julia
localfilter!(similar(A, T), A, B, initial, update, final)
```


## Examples

Implementing a local minimum filter (that is, an *erosion*) with [`localfilter`](@ref) is
as simple as:

```julia
res = localfilter(A, B,
                  #= initial =# typemax(eltype(A)),
                  #= update  =# (v,a,b) -> min(v,a))
```

This is typically how *[Basic morphological operations](@ref)* are implemented in
`LocalFilters`. Here `B` is only used to define the neighborhood, it is usually called a
*structuring element* in this context.

As another example, implementing a convolution of `A` by `B` writes:

```julia
res = localfilter(similar(A), A, B,
                   #= initial =# zero(eltype(A)),
                   #= update  =# (v,a,b) -> v + a*b; order = REVERSE_FILTER)
```

while:

```julia
dst = localfilter!(similar(A), A, B,
                   #= initial =# zero(eltype(A)),
                   #= update  =# (v,a,b) -> v + a*b; order = FORWARD_FILTER)
```

computes a correlation of `A` by `B`. The only difference is the `order` keyword which may
be omitted in the latter case as `FORWARD_FILTER` is the default. In the case of
convolutions and correlations, `B` defines the neighborhood but also the weights, it is
usually called a *kernel* in this context.

Apart from specializations to account for the type of neighborhood defined by `B`, it is
essentially the way the [`correlate`](@ref) and [`convolve!`](@ref) methods (described in
the Section *[Linear filters](@ref)*) are implemented in `LocalFilters`.

In the above examples, the `initial` value of the state variable is always the same and
directly provided while the default `final = identity` is assumed. Below is a more
involved example to compute the *roughness* defined by Wilson et al. (in [*Marine Geodesy*
30:3-35, 2007](https://www.tandfonline.com/doi/abs/10.1080/01490410701295962)) as the
maximum absolute difference between a central cell and surrounding cells:

```julia
function roughness(A::AbstractArray{<:Real}, B=3)
    initial(a) = (; result = zero(a), center = a)
    update(v, a, _) = (; result = max(v.result, abs(a - v.center)), center=v.center)
    final(v) = v.result
    return localfilter!(similar(A), A, B, initial, update, final)
end
```

This example has been borrowed from the
[`Geomorphometry`](https://github.com/Deltares/Geomorphometry.jl) package for the analysis
of *Digital Elevation Models* (DEM).
