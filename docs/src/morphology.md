# Non-linear morphological filters

## Basic morphological operations

Erosion and dilation are the basic operations of mathematical morphology, they are
implemented by methods [`erode`](@ref) and [`dilate`](@ref):

```julia
erode(A, R=3) -> Amin
dilate(A, R=3) -> Amax
```

which respectively return the local minima `Amin` and the local maxima `Amax` of argument
`A` in a *structuring element* defined by `R`. The notion of *structuring element* in
mathematical morphology is equivalent to that of *neighborhood* in `LocalFilters`. The
returned result is similar to `A` (same size and type). If `R` is not specified, a default
hyper-rectangular moving window 3 samples wide in every dimension of `A` is used. If the
structuring element `R` is a simple hyper-rectangular moving window, the much faster van
Herk-Gil-Werman algorithm is used

The [`localextrema`](@ref) method combines these two operations in one call:

```julia
localextrema(A, R=3) -> Amin, Amax
```

Calling [`localextrema`](@ref) is usually almost twice as fast as calling [`erode`](@ref)
and [`dilate`](@ref).

To avoid allocating new arrays, the methods [`erode!`](@ref), [`dilate!`](@ref), and
[`localextrema!`](@ref) provide in-place versions which apply the operation to `A` with
structuring element `R` and store the result in the provided arrays `Amin` and/or `Amax`:

```julia
erode!(Amin, A, R=3) -> Amin
dilate!(Amax, A, R=3) -> Amax
localextrema!(Amin, Amax, A, R=3) -> Amin, Amax
```

If the structuring element `R` is a simple hyper-rectangular moving window, the much
faster van Herk-Gil-Werman algorithm is used and the operation can be done in-place. That
is, `A` and `Amin` can be the same arrays. In that case, the following syntax is allowed:

```julia
erode!(A, R=3) -> A
dilate!(A, R=3) -> A
```

## Opening and closing filters

```julia
closing(A, R=3)
opening(A, R=3)
```

respectively perform a closing or an opening of array `A` by the structuring element `R`.
If `R` is not specified, a default hyper-rectangular moving window of size 3 in every
dimension of `A` is used. A closing is a dilation followed by an erosion, whereas an
opening is an erosion followed by a dilation.

The in-place versions are:

```julia
closing!(dst, wrk, A, R=3) -> dst
opening!(dst, wrk, A, R=3) -> dst
```

which perform the operation on the source `A` and store the result in destination `dst`
using `wrk` as a workspace array. The 3 arguments `dst`, `wrk`, and `A` must be similar
arrays; `dst` and `A` may be identical, but `wrk` must not be the same array as `A` or
`dst`. The destination `dst` is returned.


## Top-hat and bottom-hat filters

Methods [`top_hat`](@ref) and [`bottom_hat`](@ref) perform a summit/valley detection by
applying a top-hat filter to an array. They are called as:

```julia
top_hat(A, R[, S]) -> dst
bottom_hat(A, R[, S]) -> dst
```

to yield the result of the filter applied to array `A`. Argument `R` defines the
structuring element for the feature detection. Optional argument `S` specifies the
structuring element for smoothing `A` prior to the top-/bottom-hat filter. If `R` and `S`
are specified as the radii of the structuring elements, then `S` should be smaller than
`R`. For instance:

```julia
top_hat(bitmap, 3, 1)
```

may be used to detect text or lines in a bitmap image.

Methods [`LocalFilters.top_hat!`](@ref) and [`LocalFilters.bottom_hat!`](@ref) implement
the in-place versions of these filters:

```julia
top_hat!(dst, wrk, A, R[, S]) -> dst
bottom_hat!(dst, wrk, A, R[, S]) -> dst
```

apply the top-/bottom-hat filter on the source `A` and store the result in the destination
`dst` using `wrk` as a workspace array. The 3 arguments `dst`, `wrk`, and `A` must be
similar but different arrays. The destination `dst` is returned.
