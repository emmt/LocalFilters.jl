# New features and user visible changes in version 1.1.0

- Drop compatibility with Julia versions < 1.0; `Compat` only needed to run
  tests.


# New features and user visible changes in version 1.0.0

- Compatibility with Julia 0.6 to 1.1

- Add fast separable filters with the van Herk-Gil-Werman algorithm.  This
  algorithm is applied whenever possible (for `RectangularBox`, or flat
  `Kernel` whose elements are all valid).

- New `strel` function to build *structuring elements*.

- The type of the result of operations like local mean and convolution is more
  consistent (*e.g.*, local mean yields a floating-point type result).
  Rounding to the nearest integer is automatically used when the floating-point
  result of an operation is stored into a array of integers.

- Constructors for `Kernel` basically takes two arguments: the array of
  coefficients, say `A`, and the initial `CartesianIndex` for indexing the
  kernel.  This simplify the interface, notably when the array of coefficients
  `A` has not 1-based indexing.

- Compatibility with Julia versions 0.6, 0.7 and 1.0 without loss of
  performances.  This has been achieved thanks to the new `cartesianregion()`
  method (see below).

- The method `cartesianregion()` is provided to return either a
  `CartesianIndices{N}` or a `CartesianRange{CartesianIndex{N}}` (whichever is
  the most efficient depending on Julia version) to loop over the
  `N`-dimensional indices of anything whose type belongs to
  `CartesianRegion{N}`.  Type `CartesianRegion{N}` is an union of the types of
  anything suitable to define a Cartesian region of indices.

- Methods `initialindex` and `finalindex` are provided to retrieve the first
  and last `CartesianIndex` for indexing their argument.

- Types `CartesianBox` and `CenteredBox` have been merged in a single type
  named `RectangularBox` (to avoid conflicts with the
  [CartesianBoxes](https://github.com/emmt/CartesianBoxes.jl) package).  Hence,
  `Neighborhood` has two concrete subtypes: `RectangularBox` and `Kernel`.

- Method `anchor` has been removed because its result depends on the indexing
  of the embedded array of kernel coefficients.

- Add [bilateral filter](https://en.wikipedia.org/wiki/Bilateral_filter).
