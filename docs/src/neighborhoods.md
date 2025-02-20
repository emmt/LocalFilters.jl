# Neighborhoods, structuring elements, and kernels

## Definitions

*Neighborhoods* (a.k.a. *structuring elements* in mathematical morphology) and *kernels*
are central concepts in `LocalFilters`. The neighborhood defines which values are involved
in a local operation for each output value of the filter. Neighborhoods are assumed to be
shift invariant but may have any support shape and may have embedded weights (*e.g.*, to
implement *local convolution*). In this latter case, they are called *kernels*.

In `LocalFilters`, a filtering operation, say

```julia
dst = filter(A, B)
```

involves, at each index `i` of a source array `A`, the values `A[i±k]` of `A` for all
indices `k` of `B` and where:

* `i ± k = i + k` for operations like **correlations** where `A` and `B` can both be
  accessed in **forward order** of their indices (which is generally faster);

* `i ± k = i - k` for operations like **convolutions** where one of `A` or `B` must be
  accessed in **reverse order** of its indices (which is generally slower).

In `LocalFilters`, the following terminology is used for `B`:

* `B` is called a **neighborhood** or a **structuring element** for mathematical
  morphology operations (see Section *[Non-linear morphological filters](@ref)*) if its
  purpose is to define the indices in the source relatively to a given index in the
  destination. Such neighborhoods are represented by arrays with Boolean entries (`true`
  where entries are part of the neighborhood) and with as many dimensions as the source
  array `A`.

* `B` is called an **hyper-rectangular box** if it is a sliding window whose edges are
  aligned with the Cartesian axes of the array. Such simple neighborhoods are like the
  previous ones but with all entries equal to `true`, they are most efficiently
  represented by fast uniform arrays like `FastUniformArray{Bool,N}` instances from the
  [`StructuredArrays`](https://github.com/emmt/StructuredArrays.jl) package. Another
  advantage of hyper-rectangular boxes is that they define a separable structuring element
  which may be exploited for very fast morphological operations by the van Herk-Gil-Werman
  algorithm whose numerical complexity does not depend on the size of the neighborhood
  (see [`localfilter!`](@ref)).

* `B` is called a **kernel** when its values are combined by the filter with those of the
  source. This is typically the case of discrete convolutions and correlations. Kernels
  are represented by `AbstractArray{T,N}` instances, with `T` the numerical type of the
  weights and `N` the number of dimensions.

From the user point of view, `B` whether it is a *neighborhood*, a *structuring element*,
an *hyper-rectangular box*, a *sliding window*, or a *kernel* is thus just an array
(usually of small size) with the same number of dimensions as the source array `A`

In addition, `B` may have arbitrary index offsets, to make it *centered* for example. For
hyper-rectangular neighborhoods, `FastUniformArray` instances (from the
[`StructuredArrays`](https://github.com/emmt/StructuredArrays.jl) package) may directly
have custom index offsets. For other neighborhoods and kernels, offset arrays (from the
[`OffsetArrays`](https://github.com/JuliaArrays/OffsetArrays.jl) package) can be used to
implement such offsets.


## Simple rules for specifying neighborhoods and kernels

To facilitate specifying `N`-dimensional neighborhoods and kernels in the filters provided
by `LocalFilters`, the following rules are applied to convert argument `B` to a suitable
neighborhood or kernel:

* If `B` is an `N`-dimensional array, it is used unchanged. Hyper-rectangular boxes or
  symmetries may however be automatically detected to choose the fastest filtering method.

* If `B` is a 2-tuple of`N`-dimensional Cartesian indices, say, `(I_first,I_last)` each of
  type `CartesianIndex{N}`, then the corresponding neighborhood is an hyper-rectangular
  box whose first and last indices are `I_first` and `I_last`.

* If `B` is a `N`-dimensional Cartesian range of type of type `CartesianIndices{N}`, then
  the corresponding neighborhood is an hyper-rectangular box whose first and last indices
  are given by this Cartesian range.

* If `B` is an integer or an integer-valued unit range, it is interpreted as the length or
  the range of indices of the neighborhood along each dimension. In the case of a simple
  length, say, `len`, the index range of the neighborhood will be approximately centered
  using the same conventions as for `fftshift`: `-i:i` for odd lengths and `-i:i-1` for
  even ones and with `i = len ÷ 2`.

* Otherwise `B` may be specified as a `N`-tuple of lengths or ranges (the two can be
  mixed), one for each dimension and where lengths are interpreted as in the previous
  case.

Note that all these cases except the first one correspond to hyper-rectangular boxes. The
**default neighborhood** is `B = 3` which corresponds to a centered hyper-rectangular
sliding window of width 3 in each of its dimensions.

This assumed conversion may be explicitly performed by calling the [`kernel`](@ref)
method:

```julia
kernel(Dims{N}, B)
```

yields the `N`-dimensional neighborhood or kernel corresponding to `B`. If the number `N`
of dimensions can be inferred from `B`, argument `Dims{N}` can be omitted.


## Simple operations on kernels

If `B` is an `N`-dimensional array representing a neighborhood or a kernel, its indices
may be centered by calling the [`LocalFilters.centered`](@ref) method:

```julia
C = LocalFilters.centered(B)
```

which yields an array `C` with the same dimensions and values as `B` but with offset axes.
This even work for arrays already centered or with offset axes that are not centered. Note
that this method follows the same conventions as for `fftshift` (explained above) and has
thus a slightly different semantic than the `OffsetArrays.centered` method which assumes
that the centered range of an even dimension `1-i:i` instead of `-i:i-1`.

It may be convenient to *reverse* a kernel or a neighborhood, this is done by calling the
[`reverse_kernel`](@ref) method:

```julia
R = reverse_kernel(B)
```

which yields an array `R` with the same dimensions and values as `B` but such that `R[i] =
B[-i]` for any index `i` such that `-i` is in-bounds for `B`. This may be useful to
perform a discrete convolution by the kernel `B` by a discrete correlation by `R` which is
usually faster. Note that the [`reverse_kernel`](@ref) method reverses the order of the
values of `B` as the base `reverse` method but also negates the axis ranges.


## Hyper-balls

To build a neighborhood, or a structuring element that is a `N`-dimensional hyper-ball of
radius `r` call:

```julia
LocalFilters.ball(Dims{N}, r)
```

For instance:

```julia
julia> B = LocalFilters.ball(Dims{2}, 3.5)
7×7 OffsetArray(::Matrix{Bool}, -3:3, -3:3) with eltype Bool with indices -3:3×-3:3:
 0  0  1  1  1  0  0
 0  1  1  1  1  1  0
 1  1  1  1  1  1  1
 1  1  1  1  1  1  1
 1  1  1  1  1  1  1
 0  1  1  1  1  1  0
 0  0  1  1  1  0  0
```

This neighborhood is geometrically centered thanks to offset axes, to have a 1-based
indices, you can do:

```julia
julia> B = LocalFilters.ball(Dims{2}, 3.5).parent
7×7 Matrix{Bool}:
 0  0  1  1  1  0  0
 0  1  1  1  1  1  0
 1  1  1  1  1  1  1
 1  1  1  1  1  1  1
 1  1  1  1  1  1  1
 0  1  1  1  1  1  0
 0  0  1  1  1  0  0
```
