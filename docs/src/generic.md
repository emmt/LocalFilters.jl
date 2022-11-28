# Generic local filters

Most filters provided by the `LocalFilters` package are implemented by the
generic [`localfilter!`](@ref) method.


## The `localfilter!` method

A local filtering operation can be performed by calling the
[`localfilter!`](@ref) method as follows:

```julia
localfilter!(dst, A, B, initial, update, store!) -> dst
```

where `dst` is the destination, `A` is the source, `B` defines the
neighborhood, `initial`, `update`, and `store!` are three functions whose
purpose is explained by the following pseudo-code implementing the local
filtering:

```julia
@inbounds for i ∈ Sup(A)
    v = initial(A[i])
    for j ∈ Sup(A) ∩ (i - Sup(B))
        v = update(v, A[j], B[i-j])
    end
    store!(dst, i, v)
end
```

where `Sup(A)` denotes the support of `A` (that is the set of indices of `A`)
and `i - Sup(B)` denotes the set of indices `j` such that `i - j ∈ Sup(B)` with
`Sup(B)` the support of `B`. In other words, `j ∈ Sup(A) ∩ (i - Sup(B))` means
all indices `j` such that `j ∈ Sup(A)` and `i - j ∈ Sup(B)`, hence `A[j]` and
`B[i-j]` are in-bounds. Here, indices `i` and `j` are multi-dimensional
Cartesian indices thus `Sup(A)` is the analogous of `CartesianIndices(A)` in
Julia.

The behavior of the filter is fully determined by the *neighborhood* `B` (see
Section [*Neighborhoods, structuring elements, and
kernels*](neighborhoods.html)), by the type of the state variable `v`, and by
the methods:

- `initial(a)` which yields the initial value of the state variable `v` given
  `a = A[i]`;

- `update(v, a, b)` which yields the updated state variable `v` given the state
  variable `v`, `a = A[j]`, and `b = B[i-j]`;

- `store!(dst, i, v)` which extracts the result of the filter from the state
  variable `v` and stores it at index `i` in the destination `dst`.

!!! warning
    The loop(s) in `localfilter!` are performed without bounds checking of the
    destination and it is the caller's responsibility to insure that the
    destination have the correct size. It is however always possible to write
    `store!` so that it performs bounds checking.


## Examples

For example, implementing a local minimum filter (that is, an *erosion*), is as
simple as:

```julia
localfilter!(dst, A, B,
             (a)     -> typemax(a),
             (v,a,b) -> min(v,a),
             (d,i,v) -> @inbounds(d[i] = v))
```

This is typically how [mathematical morphology](#morphology) methods are
implemented in `LocalFilters`.

As another example, implementing a convolution by `B` writes:

```julia
localfilter!(dst, A, B,
             (a)     -> zero(a),
             (v,a,b) -> v + a*b,
             (d,i,v) -> @inbounds(d[i] = v))
```

Apart from specializations to account for the type of neighborhood defined by
`B`, it is essentially the way the [`convolve`](@ref) and [`convolve!`](@ref)
methods (described in the Section [*Linear filters*](linear.html)) are
implemented in `LocalFilters`.
