* Factorize computations for rectangular regions, see erode/dilate/etc.  in
  `JuliaImages/Images.jl`.  Implement van Herk / Gil & Werman algorithm.

* Add a repeat count to a basic operation, or even better: use `B^n` to repeat
  `n` times basic operations with structuring element, or `...*B2*B1` to apply
  with `B1` then with `B2`, etc.

* Detect zero-width region which are a no-op.

* Define regions as `OffsetArray` (see
  https://github.com/alsam/OffsetArrays.jl).

* Extend this to other types of *kernels* (convolution, median, *etc.*).

* Kernels can be separables of not.

* Implement bound types.

* Use `StaticArray` for small kernels.

* Implement morphological gradient operators.

* Merge `CartesianBox` and `CenteredBox` (unless there is a performance
  regression), make all boxes iterable.

* Automatically split arrays for multi-threaded processing.

* Make all neighborhoods iterable and check iterations over neighborhoods.
