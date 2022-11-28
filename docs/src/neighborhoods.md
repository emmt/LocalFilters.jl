# Neighborhoods, structuring elements, and kernels

In `LocalFilters`, a filtering operation, say

```julia
dst = filter(A, B)
```

involves, at each index `i` of a source array `A`, the values `A[i-k]` of `A`
for all indices `k` of `B`. In `LocalFilters`, the following terminology is
used for `B`:

* `B` is called a *neighborhood* or a *structuring element* for mathematical
  morphology operations (see Section [*Non-linear morphological
  filters*](morphology.html)) if its purpose is to define the indices in the
  source relatively to a given index in the destination. Such neighborhoods can
  be hyper-rectangular Cartesian sliding windows represented by a
  [`RectangularBox`](@ref) instance or regions with more complex shapes which
  are represented by arrays with offset axes and boolean entries (`true` where
  entries are part of the neighborhood).

* `B` is called a *kernel* when its values are combined by the filter with
  those of the source. This is typically the case of discrete convolutions and
  correlations.


## Neighborhoods and structuring elements

A neighborhood (a.k.a. *structuring element* in mathematical morphology) is a
central concept in `LocalFilters`. The neighborhood defines which values are
involved in a local operation for each output value of the filter.
Neighborhoods are assumed to be shift invariant but may have any support shape
and may have embedded weights (*e.g.*, to implement *local convolution*).


### Types of neighborhoods

From the user point of view, there are three kinds of neighborhoods:

* **Rectangular boxes** are rectangular neighborhoods whose edges are aligned
  with the axes of array indices and which may be centered or have arbitrary
  offsets along the dimensions. These neighborhoods are represented by
  instances of `LocalFilters.RectangularBox`.

* **Arbitrarily shaped neighborhoods** are neighborhoods with arbitrary shape
  and offset. These neighborhoods are represented by instances of
  `LocalFilters.Kernel` with boolean element type. These neighborhoods are
  constructed from an array of booleans and an optional starting index.

* **Kernels** are neighborhoods whose elements are weights and which may have
  arbitrary offset. These neighborhoods are represented by instances of
  `LocalFilters.Kernel` with numerical element type. These neighborhoods are
  constructed from an array of weights and an optional starting index.


### Syntaxes for neighborhoods

* The *default neighborhood* is a centered hyper-rectangular Cartesian sliding
  window of width 3 in each of its dimensions. He *Cartesian* means that the
  edges of the neighborhood are algned with the array axes.

* A *scalar* integer `w` yields a centered rectangular box of size `w` along
  all dimensions. `w` must be at least equal to 1 and the geometrical center of
  the box is defined according to the conventions in `fftshift`.

* A *tuple* `t` of integers yields a centered rectangular box whose size is
  `t[i]` along the `i`-th dimension. All values of `t` must be larger or equal
  to 1. Tip: Remember that you can use `v...` to convert a *vector* `v` into a
  tuple.

* An *array* `A` yields a `LocalFilters.Kernel` whose coefficients are the
  values of `A` and whose neighborhood is the centered bounding-box of `A`.

* A *Cartesian region* `R` (an instance of `CartesianIndices`) yields a
  `LocalFilters.RectangularBox` which is a rectangular neighborhood whose
  support contains all relative positions within `first(R)` and `last(R)`.

* A rectangular box neighborhood created by calling
  `LocalFilters.RectangularBox` as:

  ```julia
  LocalFilters.RectangularBox(R)
  LocalFilters.RectangularBox(I1, I2)
  LocalFilters.RectangularBox(dims, offs)
  LocalFilters.RectangularBox(inds)
  ```

  where `R` is an instance of`CartesianIndices`, `I1` and `I2` are two
  `CartesianIndex` specifying the first and last relative position within the
  neighborhood, `dims` and `offs` are tuples of integers specifying the
  dimensions of the neighborhood and its offsets, `inds` are unit ranges.

  Assuming `dim` is an integer, then:

  ```julia
  LocalFilters.RectangularBox{N}(dim)
  ```

  yields an `N`-dimensional rectangular box of size `dim` along all dimensions
  and centered at the geometrical center of the box (with the same conventions
  as `fftshift`).

  Similarly, assuming `i1` and `i2` are integers, then:

  ```julia
  LocalFilters.RectangularBox{N}(i1:i2)
  ```

  yields an `N`-dimensional rectangular box with index range `i1:i2` along all
  dimensions.


### Methods on a neighborhood

The following statements make sense on a neighborhood `B`:

```julia
eltype(B) -> element type of B
ndims(B)  -> number of dimensions of B
length(B) -> number of elements in the bounding-box of B
size(B)   -> size of the bounding-box of B along all dimensions
size(B,d) -> size of the bounding-box of B along d-th dimension
first(B)  -> CartesianIndex of first position in the bounding-box
             of B relative to its anchor
last(B)   -> CartesianIndex of last position in the bounding-box
             of B relative to its anchor
B[i]      -> yields the kernel value of B at index i
```

Note that the index `i` in `B[i]` is assumed to be between `first(B)` and
`last(B)`, for efficiency reasons this is not checked. The type returned by
`eltype(B)` is `Bool` for a neighborhood which is just defined by its support
(*e.g.* a `LocalFilters.CenteredBox` or a `LocalFilters.RectangularBox`), the
element type of its kernel otherwise.

```julia
CartesianIndices(B)
```

yields the Cartesian indices of relative positions of the bounding-box of
neighborhood `B`.

If the argument `B` which defines a neighborhood (see previous section) is not
an instance of a type derived from `LocalFilters.Neighborhood`, it may be
explicitly converted by:

```julia
convert(LocalFilters.Neighborhood{N}, B)
```

with `N` the number of dimensions of the target array.


## Kernels

A [`LocalFilters.Kernel`](@ref) can be used to define a weighted neighborhood
(for weighted local average or for convolution) or a structuring element (for
mathematical morphology). It is a rectangular array of coefficients over a,
possibly off-centered, rectangular neighborhood. In general, it is sufficient
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
retrieved by `first_cartesian_index(B)`). The `start` parameter let the caller
choose an arbitrary origin for the kernel coefficients; when a filter is
applied, the following mapping is assumed:

```julia
B[k] ≡ C[k + off]
```

where `off = first_cartesian_index(C) - first_cartesian_index(B)`.

If `start` is omitted, its value is set so that the *origin* (whose index is
`zero(CartesianIndex{N})` with `N` the number of dimensions) of the kernel
indices is at the geometric center of the array of coefficients (see
[`LocalFilters.default_start`](@ref)). Optional type parameter `T` is to impose
the type of the coefficients.

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
