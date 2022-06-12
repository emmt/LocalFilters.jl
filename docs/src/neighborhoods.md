# Neighborhoods, structuring elements, and kernels

In `LocalFilters`, at each index `i` of a source array `A`, a local filter
involves the values `A[i-k]` of `A` for all indices `k` in the *neighborhood*
`B` which is part of the filter.  For some neighborhoods, only the support of
`B` (that is the range of valid indices `k`) is relevant, for others the values
`B[k]` may also be considered.

Support-only neighborhoods can be hyperrectangular Cartesian regions
represented by a [`RectangularBox`](@ref) instance or regions with more complex
shapes which are represented by (usually) small arrays with offset axes and
boolean entries (`true` where entries are part of the neighborhood).  Such
neighborhoods are used to define the so-called *structuring element* of
mathematical morphology non-linear filters (see Section [*Non-linear
morphological filters*](morphology.html)).


## Neighborhoods and structuring elements

[`Neighborhood`](@ref) is the abstract type of which all neighborhood types
inherit.  The most simple neighborhood is a hyperrectangular Cartesian region
constructed by:

```julia
RectangularBox(start, stop)
```

where `start` and `stop` are Cartesian indices (instances of
`CartesianIndex{N}`) respectively specifying the first and last indices of the
region.

Another possibility is to specify the dimensions of the box and the offsets of
its central element:

```julia
RectangularBox(dims, offs)
```

with `dims` a `N`-tuple of dimensions and `offs` either a `N`-tuple of indices
of an instance of `CartesianIndex{N}`.

A `RectangularBox` can also be defined by the index ranges along all the
dimensions.  For example:

```julia
RectangularBox(-3:3, 0, -2:1)
RectangularBox((-3:3, 0, -2:1))
```

both yield a 3-dimensional `RectangularBox` of size `7×1×4` and whose first
index varies on `-3:3`, its second index is `0` while its third index varies on
`-2:1`.

Finally, a `RectangularBox` can be defined as:

 ```julia
RectangularBox(R)
```

where `R` is an instance of `CartesianIndices`.


## Kernels

A [`LocalFilters.Kernel`](@ref) can be used to define a weighted neighborhood
(for weighted local average or for convolution) or a structuring element (for
mathematical morphology).  It is a rectangular array of coefficients over a,
possibly off-centered, rectangular neighborhood.  In general, it is sufficient
to specify `::LocalFilters.Kernel{T,N}` in the signature of methods, with `T`
the type of the coefficients and `N` the number of dimensions (the third
parameter `A` of the type is to fully qualify the type of the array of
coefficients).

A kernel is built as:

```julia
B = LocalFilters.Kernel{T}(C, start=default_start(C))
```

where `C` is the array of coefficients (which can be retrieved by `coefs(B)`)
and `start` the initial `CartesianIndex` for indexing the kernel (which can be
retrieved by `first_cartesian_index(B)`).  The `start` parameter let the caller
choose an arbitrary origin for the kernel coefficients; when a filter is
applied, the following mapping is assumed:

```julia
B[k] ≡ C[k + off]
```

where `off = first_cartesian_index(C) - first_cartesian_index(B)`.

If `start` is omitted, its value is set so that the *origin* (whose index is
`zero(CartesianIndex{N})` with `N` the number of dimensions) of the kernel
indices is at the geometric center of the array of coefficients (see
[`LocalFilters.default_start`](@ref)).  Optional type parameter `T` is to
impose the type of the coefficients.

To convert the element type of the coefficients of an existing kernel, do:

```julia
LocalFilters.Kernel{T}(K)
```

which yields a kernel whose coefficients are those of the kernel `K`
converted to type `T`.

It is also possible to convert instances of [`RectangularBox`](@ref) into a
kernel with boolean coefficients by calling:

```julia
LocalFilters.Kernel(B)
```

where `B` is the neighborhood to convert into an instance of `Kernel`.
