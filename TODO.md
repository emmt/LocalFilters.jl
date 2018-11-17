* Use `@simd`.

* Add more rules to automatically convert type of kernel coefficients.

* Rename `Kernel` as `FilterKernel` and export.

* Check that the incidence on execution time (`localfilter!` may be faster on
  `CenteredBox`) is negligible.

* A union being slower to dispatch (check this?), consider suppressing
  `CartesianRegion` and `IndexInterval`.

* Use van Herk / Gil & Werman algorithm whenever possible (for
  `RectangularBox`, or `Kernel` whose elements are all equal to 1).

* Add a repeat count to a basic operation, or even better: use `B^n` to repeat
  `n` times basic operations with structuring element, or `...*B2*B1` to apply
  with `B1` then with `B2`, etc.

* Detect zero-width region which are a no-op.

* Define neighborhoods as `OffsetArray` (see
  https://github.com/alsam/OffsetArrays.jl).

* Extend this to other types of *kernels* (convolution, median, *etc.*).

* Implement bound types.

* Implement `correlate`.

* Use `StaticArray` for small kernels.

* Implement morphological gradient operators.

* Multi-threading:
  - Automatically split arrays for multi-threaded processing.
  - Multi-thread separable filtering operations (require one workspace per
    thread).

* Make all neighborhoods iterable and check iterations over neighborhoods.
