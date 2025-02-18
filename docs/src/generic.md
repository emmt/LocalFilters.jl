# Generic local filters

Most filters provided by the `LocalFilters` package are implemented by the generic
[`localfilter!`](@ref) method.


## The `localfilter!` method

A local filtering operation can be performed by calling the [`localfilter!`](@ref) method
as follows:

```julia
localfilter!(dst, A, B, initial, update, final = identity) -> dst
```

where `dst` is the destination, `A` is the source, `B` defines the neighborhood, `initial`
gives the initial value of the state variable, `update` is a function to update the state
variable for each entry of the neighborhood, and `final` is a function to yield the local
result of the filter given the final value of the state variable. The purposes of these
parameters are explained by the following pseudo-code implementing the local filtering:

```julia
@inbounds for i ∈ indices(dst)
    v = initial
    for j ∈ indices(A) ∩ (indices(B) + i)
        v = update(v, A[j], B[j-i])
    end
    dst[i] = final(v)
end
```

where `indices(A)` denotes the set of indices of `A` while `indices(B) + i` denotes the
set of indices `j` such that `j - i ∈ indices(B)` with `indices(B)` the set of indices of
`B`. In other words, `j ∈ indices(A) ∩ (indices(B) + i)` means all indices `j` such that
`j ∈ indices(A)` and `j - i ∈ indices(B)`, hence `A[j]` and `B[j-i]` are in-bounds. In
`LocalFilters`, indices `i` and `j` are multi-dimensional Cartesian indices, thus
`indices(A)` is the analogous of `CartesianIndices(A)` in Julia.

The behavior of the filter is fully determined by the *neighborhood* `B` (see Section
[*Neighborhoods, structuring elements, and kernels*](neighborhoods.html)), by the type and
initial value of the state variable `v`, and by the methods:

- `update(v, a, b)` which yields the updated state variable `v` given the state variable
  `v`, `a = A[j]`, and `b = B[j-i]`;

- `final(v)` which yields the result of the filter given the state variable `v`.


## Examples

For example, implementing a local minimum filter (that is, an *erosion*), is as simple as:

```julia
localfilter!(dst, A, B, typemax(a),
             (v,a,b) -> min(v,a),
             (d,i,v) -> @inbounds(d[i] = v))
```

This is typically how [mathematical morphology](#morphology) methods are implemented in
`LocalFilters`.

As another example, implementing a convolution by `B` writes:

```julia
localfilter!(dst, A, B,
             (a)     -> zero(a),
             (v,a,b) -> v + a*b)
```

Apart from specializations to account for the type of neighborhood defined by `B`, it is
essentially the way the [`convolve`](@ref) and [`convolve!`](@ref) methods (described in
the Section [*Linear filters*](linear.html)) are implemented in `LocalFilters`.
