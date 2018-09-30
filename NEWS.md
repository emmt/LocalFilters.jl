* (2018-09-30) many improvements and changes:

  - Compatibility with Julia versions 0.6, 0.7 and 1.0 without loss of
    performances.

  - Constructors for `Kernel` basically takes two arguments: the array of
    coefficients, say `A`, and the initial `CartesianIndex` for indexing the
    kernel.  This simplify the interface, notably when the array of
    coefficients `A` has not 1-based indexing.

  - `CartesianRegion{N}` is an union of the types of anything suitable to
    define a Cartesian region of indices.

  - The method `cartesianregion()` is provided to return either a
    `CartesianIndices{N}` or a `CartesianRange{CartesianIndex{N}}` (whichever
    is most efficient depending on Julia version) to loop over the
    `N`-dimensional indices of anything whose type belongs to
    `CartesianRegion{N}`.

  - Methods `initialindex` and `finalindex` are provided to retrieve the
    first and last `CartesianIndex` for indexing their argument.

  - Method `anchor` has been removed because its result depends on the indexing
    of the embedded array of kernel coefficients.

* (2018-02-19) add [bilateral filter](https://en.wikipedia.org/wiki/Bilateral_filter).
