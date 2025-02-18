# Efficient separable filters for associative operations

## Out-of-place version

The [`localfilter`](@ref) method may be called as:

```julia
dst = localfilter([T=eltype(A),] A, dims, op, rngs [, w])
```

to apply the van Herk-Gil-Werman algorithm to filter array `A` along dimension(s) `dims`
with (associative) binary operation `op` and contiguous structuring element(s) defined by
the interval(s) `rngs`. Optional argument `T` is the element type of the result `dst` (by
default `T = eltype(A)`). Optional argument `w` is a workspace array which is
automatically allocated if not provided; otherwise, it must be a vector with the same
element type as `A` which is resized as needed (by calling the `resize!` method).

Argument `dims` specifies along which dimension(s) of `A` the filter is to be applied, it
can be a single integer, several integers or a colon `:` to specify all dimensions.
Dimensions are processed in the order given by `dims` (the same dimension may appear
several times) and there must be a matching interval in `rngs` to specify the structuring
element (except that if `rngs` is a single interval, it is used for every dimension in
`dims`). An interval is either an integer or an integer valued unit range in the form
`kmin:kmax` (an interval specified as a single integer, say `k`, is the same as specifying
`k:k`).

Assuming a mono-dimensional array `A`, the single filtering pass:

```julia
localfilter!(dst, A, :, op, rng)
```

amounts to computing:

```julia
dst[j] = A[j-kmax] ⋄ A[j-kmax+1] ⋄ A[j-kmax+2] ⋄ ... ⋄ A[j-kmin]
```

for all `j ∈ [first(axes(A,1)):last(axes(A,1))]`, with `x ⋄ y = op(x, y)`, `kmin =
first(rng)` and `kmax = last(rng)`. Note that if `kmin = kmax = k` (which occurs if `rng`
is a simple integer), the result of the filter is to operate a simple shift by `k` along
the corresponding dimension and has no effects if `k = 0`. This can be exploited to not
filter some dimension(s).


## In-place version

The [`localfilter!`](@ref) method implement the *in-place* version of the van
Herk-Gil-Werman algorithm:

```julia
localfilter!([dst = A,] A, dims, op, rngs [, w]) -> dst
```

overwrites the contents of `dst` with the result of the filter and returns `dst`. The
destination array `dst` must have the same indices as the source `A` (that is, `axes(dst)
== axes(A)`). If `dst` is not specified or if `dst` is `A`, the operation is performed
in-place.


## Examples

The in-place *morphological erosion* (local minimum) of the array `A` on a centered
structuring element of width 7 in every dimension can be applied by:

```julia
localfilter!(A, :, min, -3:3)
```

Index interval `0` may be specified to do nothing along the corresponding dimension. For
instance, assuming `A` is a three-dimensional array:

```julia
localfilter!(A, :, max, (-3:3, 0, -4:4))
```

overwrites `A` by its *morphological dilation* (*i.e.* local maximum) in a centered local
neighborhood of size `7×1×9` (nothing is done along the second dimension). The same result
may be obtained with:

```julia
localfilter!(A, (1,3), max, (-3:3, -4:4))
```

where the second dimension is omitted from the list of dimensions.

The *local average* of the two-dimensional array `A` on a centered moving window of size
11×11 can be computed as:

```julia
localfilter(A, :, +, (-5:5, -5:5))*(1/11)
```

## Efficiency and restrictions

The van Herk-Gil-Werman algorithm is very fast for rectangular structuring elements. It
takes at most 3 operations to filter an element along a given dimension whatever the width
`p` of the considered local neighborhood. For `N`-dimensional arrays, the algorithm
requires only `3N` operations per element instead of `p^N - 1` operations for a naive
implementation. This however requires to make a pass along each dimension so [memory page
faults](https://en.wikipedia.org/wiki/Page_fault) may reduce the performances. This is
somewhat attenuated by the fact that the algorithm can be applied in-place. For efficient
multi-dimensional out-of-place filtering, it is recommended to make the first pass with a
fresh destination array and then all other passes in-place on the destination array.

To apply the van Herk-Gil-Werman algorithm, the structuring element must be separable
along the dimensions and its components must be contiguous. In other words, the algorithm
is only applicable for `N`-dimensional rectangular neighborhoods, so-called
*hyperrectangles*. The structuring element may however be off-centered by arbitrary
offsets along each dimension.

To take into account boundary conditions (for now, only nearest neighbor is implemented)
and allow for in-place operation, the algorithm allocates a workspace array.


## References

* Marcel van Herk, "*A fast algorithm for local minimum and maximum filters on rectangular
  and octagonal kernels*" in Pattern Recognition Letters **13**, 517-521 (1992).

* Joseph Gil and Michael Werman, "*Computing 2-D Min, Median, and Max Filters*" in IEEE
  Transactions on Pattern Analysis and Machine Intelligence **15**, 504-507 (1993).
