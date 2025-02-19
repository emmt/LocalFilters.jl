# New features and user visible changes in branch 2.0

* `LocalFilters.ball(DimS{N}, r)` now yields a *centered* `N`-dimensional ball where
  values are set according to whether the distance to the center is `≤ r`. Compared to the
  previous versions, add `1//2` to `r` to get a similar shape. The algorithm is faster and
  has been fixed for `N > 2`. The result is identical whether `r` is integer or
  floating-point.

* New exported function `reverse_kernel(A)` to yield a kernel `B` such that a correlation
  by `B` yields the same result as a convolution by `A` and conversely.

* Non-exported `Box{N}` is now an alias to `FastUniformArray{Bool,N,true}` from the
  `StructuredArrays` package.

* Morphological methods have a `slow` keyword (false by default) to force not using the
  the van Herk-Gil-Werman algorithm.

* Macro `@public` to declare *public* methods even though they are not exported. This
  concept was introduced in Julia 1.11, for older Julia versions, nothing change.

## Version 2.0.1

* Fix import of unused function that has been removed (see PR #9).

## Version 2.0.0

Version 2 of `LocalFilters` better integrates in the Julia ecosystem as fewer custom types
are introduced:

* To represent hyper-rectangular Cartesian sliding windows, former type `RectangularBox`
  has been replaced by `CartesianIndices`.

* Kernels with values or neighborhoods with more complex shape than hyper-rectangular
  Cartesian sliding windows are all represented by abstract arrays (with boolean entries
  to define a neighborhood). The former type `LocalFilters.Kernel` is no longer provided.
  To account for offsets in the indexing, it is recommended to use the
  [`OffsetArrays`](https://github.com/JuliaArrays/OffsetArrays.jl) package. The method
  `LocalFilters.centered(B)` can be called to yield a kernel or a neighborhood whose index
  ranges are approximately centered. This method is not exported to avoid conflicts (for
  instance, it has a slightly different semantic in `OffsetArrays`).

Version 2 of `LocalFilters` provides more general, more consistent, and better optimized
methods:

* Most filtering operations take an optional ordering argument `ord` right before the
  argument, say `B`, specifying the kernel or the neighborhood. If `ord` is
  `ForwardFilter`, `B` is indexed as in discrete correlations; otherwise, if `ord` is
  `ReverseFilter`, `B` is indexed as in discrete convolutions. The default ordering is
  `ForwardFilter` as this is the most natural for many filters (except discrete
  convolutions of course) and as it yields faster code. For symmetric kernels and
  neighborhoods, the ordering has no incidence on the result. In `LocalFilters` version 1,
  indexing as in discrete convolutions was the only rule.

* The API of `localfilters!` have changed a bit, the syntax is
  `localfilters!(dst,A,ord=ForwardFilter,B,initial,update,final=identity)` with `dst` the
  destination, `A` the source, `ord` the direction of the filter, `B` the kernel or
  neighborhood of the filter, `initial` the value of the initial state variable, `update`
  a method to update the state variable, and `final` a method to yield the result to store
  in the destination `dst` given the value of the state variable at the end of visiting
  the neighborhood.

* Constructor `LocalFilters.Indices` and helper method `LocalFilters.localindices` may be
  used as an alternative to `localfilters!` to build custom filters.

* In all filters, a simple neighborhood that is a hyper-rectangular Cartesian sliding
  window can be specified in many different ways. Such a neighborhood is represented by an
  instance of `CartesianIndices` with unit step ranges. Method
  `LocalFilters.kernel(Dims{N},args...)` can be called to build such a `N`-dimensional
  neighborhood from argument(s) `args...`.

* Non-exported `LocalFilters.ball` method is now type stable. Call
  `LocalFilters.ball(Dims{N},r)` instead of `LocalFilters.ball(N,r)`.

* The `strel` method uses uniform arrays from package
  [`StructuredArrays`](https://github.com/emmt/StructuredArrays.jl) to represent
  structuring elements with the same value for all valid indices.

* In out-of-place filters, the destination array needs not be of the same size as the
  source array. The local filtering operation is applied for all indices of the
  destination, using boundary conditions to extract the corresponding value in the source
  array. Currently only *flat* boundary conditions are implemented but this may change in
  the future.


## Guidelines for porting code from version 1 to version 2

If the high level API was used, there should be almost no changes, except for
non-symmetric kernels or neighborhoods for which `ReverseFilter` ordering must be
specified to mimic the former behavior.

At a lower level, the following changes should be done:

* Non-exported union `LocalFilters.IndexInterval` has been replaced by `LocalFilters.Axis`
  to represent the type of any argument that can be used to define a sliding window axis:
  an integer length or an integer-valued index range.

* Non-exported method `LocalFilters.ismmbox` has been replaced by
  `LocalFilters.is_morpho_math_box`.

* Non-exported method `LocalFilters.cartesian_region` has been replaced by the more
  general and better designed exported method `kernel`.

* Replace `Neighborhood{N}(args...)` by `kernel(Dims{N}, args...)` and `Neighborhood{N}`
  or `RectangularBox{N}` by `LocalFilters.Box{N}`.

* Replace `LocalFilters.Kernel` by `OffsetArrays.OffsetArray`.

* Update the arguments of `localfilters!`: `initial` is no longer a method but the initial
  state value, `update` has the same semantics, and `final` just yields the result of the
  local filter given the last state value. By default, `final` is `identity`.

* Replace `LocalFilters.ball(N,r)` by `LocalFilters.ball(Dims{N},r)` which is type-stable.

---

# New features and user visible changes in branch 1.2

## Version 1.2.2

- Fix an important performance bug related to anonymous functions. Methods `localmean!`
  and `convolve!` are about 60 times faster!

- The documentation has been largely revised and on-line documentation is generated by
  [`Documenter.jl`](https://github.com/JuliaDocs/Documenter.jl) and hosted on
  https://emmt.github.io/LocalFilters.jl/.


## Version 1.2.1

- A default window of width `2*round(Int,3σ)+1` for the spatial filter in the bilateral
  filter if a Gaussian spatial filter of standard deviation `σ` is chosen.

# New features and user visible changes in version 1.2.0

- Scalar and array element type restrictions have been relaxed for most filter methods.
  This is to let users apply these methods to non-standard types.

- Some optimizations.

- Syntax `Kernel(T,G,B)` has been deprecated in favor of `Kernel{T}(G,B)`.

- Rename unexported methods `initialindex`, `finalindex`, `defaultstart`,
  `cartesianregion`, `convertcoefs`, and `strictfloor` respectively as
  `first_cartesian_index` and `last_cartesian_index`, `default_start`, `cartesianregion`,
  `convert_coefs`, and `strictfloor`.


# New features and user visible changes in branch 1.1

## Version 1.1.0

- Drop compatibility with Julia versions < 1.0; `Compat` only needed to run tests.


## Version 1.0.0

- Compatibility with Julia 0.6 to 1.1

- Add fast separable filters with the van Herk-Gil-Werman algorithm. This algorithm is
  applied whenever possible (for `RectangularBox`, or flat `Kernel` whose elements are all
  valid).

- New `strel` function to build *structuring elements*.

- The type of the result of operations like local mean and convolution is more consistent
  (*e.g.*, local mean yields a floating-point type result). Rounding to the nearest
  integer is automatically used when the floating-point result of an operation is stored
  into a array of integers.

- Constructors for `Kernel` basically takes two arguments: the array of coefficients, say
  `A`, and the initial `CartesianIndex` for indexing the kernel. This simplify the
  interface, notably when the array of coefficients `A` has not 1-based indexing.

- Compatibility with Julia versions 0.6, 0.7 and 1.0 without loss of performances. This
  has been achieved thanks to the new `cartesianregion()` method (see below).

- The method `cartesianregion()` is provided to return either a `CartesianIndices{N}` or a
  `CartesianRange{CartesianIndex{N}}` (whichever is the most efficient depending on Julia
  version) to loop over the `N`-dimensional indices of anything whose type belongs to
  `CartesianRegion{N}`. Type `CartesianRegion{N}` is an union of the types of anything
  suitable to define a Cartesian region of indices.

- Methods `initialindex` and `finalindex` are provided to retrieve the first and last
  `CartesianIndex` for indexing their argument.

- Types `CartesianBox` and `CenteredBox` have been merged in a single type named
  `RectangularBox` (to avoid conflicts with the
  [CartesianBoxes](https://github.com/emmt/CartesianBoxes.jl) package). Hence,
  `Neighborhood` has two concrete subtypes: `RectangularBox` and `Kernel`.

- Method `anchor` has been removed because its result depends on the indexing of the
  embedded array of kernel coefficients.

- Add [bilateral filter](https://en.wikipedia.org/wiki/Bilateral_filter).
