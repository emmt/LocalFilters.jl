* Make `CartesianBox` an official package and replace `RectangularBox` by
  `CartesianBox`.

* Allow for complex element type (e.g. in the bilateral filter).

* Use `@checkbounds` etc. for bound checking when indexing neighborhoods
  (cf. doc. in "Methods on a neighborhood").

* Bump to version 2.  Provide `Project.toml`, drop compatibility with Julia <
  0.7 and dependency on `Compat`, remove `REQUIRE` file.

* Automatically use van Herk / Gil & Werman algorithm for separable
  (rectangular) neighborhood and suitable operations (min, max, mean, ...).

* Use `@simd`.

* Add more rules to automatically convert type of kernel coefficients.

* Rename `Kernel` as `FilterKernel` and export.

* Check that the incidence on execution time (`localfilter!` may be faster on
  `CenteredBox`) is negligible.

* A union being slower to dispatch (check this?), consider suppressing
  `CartesianRegion` and `IndexInterval`.

* Rewrite methods for van Herk / Gil & Werman algorithm so that the out-place
  version is called as:

  ```julia
  localfilter([T,] A, dims, op, rngs [, w])
  ```

  with `T` the element type of the result (by default `T = eltype(A)` or `T =
  eltype(w)` if `w` is provided).

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
