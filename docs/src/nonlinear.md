# Non-linear filters

`LocalFilters` provides a number of non-linear filters such as the [bilateral
filter](https://en.wikipedia.org/wiki/Bilateral_filter) and mathematical morphology
filters. The latter are described in the [Mathematical morphology section](#morphology).

## The bilateral filter

    bilateralfilter([T=float(eltype(A)),] A, F, G, ...)

yields the result of applying the bilateral filter on array `A`.

Argument `F` specifies how to smooth the differences in values. It may be function which
takes two values from `A` as arguments and returns a nonnegative weight. It may be a real
which is assumed to be the standard deviation of a Gaussian.

Arguments `G, ...` specify the settings of the distance filter for smoothing differences
in coordinates. There are several possibilities:

- `G, ...` can be a [`LocalFilters.Kernel`](@ref) instance (specified as a single
  argument).

- Argument `G` may be a function taking as argument the Cartesian index of the coordinate
  differences and returning a nonnegative weight. Argument `G` may also be a real
  specifying the standard deviation of the Gaussian used to compute weights. Subsequent
  arguments `...` are to specify the neighborhood where to apply the distance filter
  function, they can be a [`Neighborhood`](@ref) object such as a [`RectangularBox`](@ref)
  or anything that may defined a neighborhood such as an odd integer assumed to be the
  width of the neighborhood along every dimensions of `A`. If a standard deviation `σ` is
  specified for `G` with no subsequent arguments, a default window of radius `3σ` is
  assumed.

Optional argument `T` can be used to force the element type used for (most) computations.
This argument is needed if the element type of `A` is not a real.

See [`bilateralfilter!`](@ref) for an in-place version of this function.

    bilateralfilter!([T=float(eltype(A)),] dst, A, F, G, ...) -> dst

overwrites `dst` with the result of applying the bilateral filter on array `A` and returns
`dst`.

See [`bilateralfilter`](@ref) for a description of the other arguments than `dst`.

See [wikipedia](https://en.wikipedia.org/wiki/Bilateral_filter) for a description of the
bilateral filter.
