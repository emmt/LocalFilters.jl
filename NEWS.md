* Version 0.2.0:

  - Add fast separable filters with the van Herk-Gil-Werman algorithm.

  - Constructors for `Kernel` basically takes two arguments: the array of
    coefficients, say `A`, and the initial `CartesianIndex` for indexing the
    kernel.  This simplify the interface, notably when the array of
    coefficients `A` has not 1-based indexing.

  - Compatibility with Julia versions 0.6, 0.7 and 1.0 without loss of
    performances.  This has been achieved thanks to the new `cartesianregion()`
    method (see below).

  - The method `cartesianregion()` is provided to return either a
    `CartesianIndices{N}` or a `CartesianRange{CartesianIndex{N}}` (whichever
    is most efficient depending on Julia version) to loop over the
    `N`-dimensional indices of anything whose type belongs to
    `CartesianRegion{N}`.  Type `CartesianRegion{N}` is an union of the types
    of anything suitable to define a Cartesian region of indices.

  - Methods `initialindex` and `finalindex` are provided to retrieve the
    first and last `CartesianIndex` for indexing their argument.

  - Types `CartesianBox` and `CenteredBox` have been merged in a single type
    named `RectangularBox` (to avoid conflicts with the
    [CartesianBoxes](https://github.com/emmt/CartesianBoxes.jl) package).
    Hence, `Neighborhood` has two concrete subtypes: `RectangularBox` and
    `Kernel`.

  - Method `anchor` has been removed because its result depends on the indexing
    of the embedded array of kernel coefficients.

  - Add [bilateral filter](https://en.wikipedia.org/wiki/Bilateral_filter).
