## Possible changes for 1.3 branch

This new minor version aims at better integration in the Julia ecosystem. More
standard structures are used. Some inconsistencies are fixed (which slightly
change the interpretaion of arguments in very specific cases).

    window_range(rng::AbstractUnitRange{Int}) = rng
    window_range(rng::AbstractUnitRange{<:Integer}) =
        Int(first(rng)):Int(last(rng))
    window_range(rng::AbstractRange{<:Integer}) =
        throw(ArgumentError("invalid non-unit step range"))
    function window_range(len::Integer)
        len ≥ 1 || throw(ArgumentError("invalid range length"))
        n = Int(len)
        return -(n >> 1):(n - 1) >> 1
    end
    @test_throws ArgumentError window_range(0)
    @test window_range(1) == 0:0
    @test window_range(4) == -2:1
    @test window_range(5) == -2:2

    cartesian_window(A::AbstractArray) = cartesian_window(axes(A))
    cartesian_window(rngs::Tuple{Vararg{Union{Integer,AbstractUnitRange{<:Integer}}}}) = CartesianIndices(map(window_range, rngs))
    cartesian_window(rngs::Tuple{Vararg{AbstractUnitRange{<:Integer}}}) =
            CartesianIndices(rngs)

* Use more standard data structures:

  * Replace `RectangularBox` by `CartesianIndices` (but restricted to unit-step
    ranges).

  * Replace `Kernel` by `OffsetArray` (see
    https://github.com/alsam/OffsetArrays.jl).

  * Suppress `Neighborhood` type.

* Fix workspace allocation in top-/bottom-hat filters.

* Make `LocalFilters.ball` type stable.

* Factorize kernels such as the Gaussian ones or the flat separable ones.

* Relax types in morphological operations.

* For consistency with the way Cartesian boxes are constructed, a scalar `n` in
  `rngs` argument of `localfilter` of van Herk-Gil-Werman algorithm is
  intepreted as the length of an approximately centered range. Explictely use
  `k:k` instead of `k` to have the old behavior.

* Use the same semantic for `store!` than `setindex!` so that `setindex!` can
  be specified as this argument.

## Other improvements and changes

* Automatically convert a all-true kernel in a simple Cartesian box and a
  all-false kernel in a simple Cartesian box of zero size.

* Do `@inbounds` on the source and kernel operations, not automatically on the
  destination. That is, let `store!` decides of this.

* Provide `@generated` or unrolled versions for small kernel sizes.

* Allow for complex element type (e.g. in the bilateral filter).

* Use `@checkbounds` etc. for bound checking when indexing neighborhoods (cf.
  doc. in "Methods on a neighborhood").

* Bump to version 2. Provide `Project.toml`, drop compatibility with Julia <
  0.7 and dependency on `Compat`, remove `REQUIRE` file.

* Automatically use van Herk / Gil & Werman algorithm for separable
  (rectangular) neighborhood and suitable operations (min, max, mean, ...).

* Add more rules to automatically convert type of kernel coefficients.

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
