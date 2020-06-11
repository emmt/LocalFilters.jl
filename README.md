# LocalFilters.jl

[![License](http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)](LICENSE.md)
[![Build Status](https://travis-ci.org/emmt/LocalFilters.jl.svg?branch=master)](https://travis-ci.org/emmt/LocalFilters.jl)
[![Build Status](https://ci.appveyor.com/api/projects/status/github/emmt/LocalFilters.jl?branch=master)](https://ci.appveyor.com/project/emmt/LocalFilters-jl/branch/master)
[![Coverage Status](https://coveralls.io/repos/emmt/LocalFilters.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/emmt/LocalFilters.jl?branch=master)
[![codecov.io](http://codecov.io/github/emmt/LocalFilters.jl/coverage.svg?branch=master)](http://codecov.io/github/emmt/LocalFilters.jl?branch=master)

This package implements multi-dimensional local filters for
[Julia](http://julialang.org/) (convolution, mathematical morphology, etc.).
Local filters are defined by specific operations combining each value of a
source array with the values in a local neighborhood which may have any size,
shape and dimensionality.  Predefined local filters are provided as well as
means to simply implement custom filters.

This document is structured as follows:

* [Summary](#summary) provides a quick introduction.

* [Implementation](#implementation) explains how to implement you own filter.

* [Neighborhoods](#neighborhoods) describes the concept of neighborhoods.

* [Installation](#installation) to install the package.

Note that this is a first implementation to define the API.  It is is
reasonably fast (see [benchmarks.jl](src/benchmarks.jl)) but separable kernels
can be made faster.

Packages with overlapping functionalities:

* [ImageFiltering](https://github.com/JuliaImages/ImageFiltering.jl) for local
  filters on multidimensional arrays (not just *images*), also implement
  various boundary conditions;

* [ImageMorphology](https://github.com/JuliaImages/ImageMorphology.jl) for fast
  morphological operations with separable structuring elements;


## Summary

`LocalFilters` implements local filtering operations which combine the values
of an array in a neighborhood of each elements of the array (and possibly the
values of a kernel associated with the neighborhood).  The neighborhood is
defined relatively to a given position by an instance of a type derived from
`Neighborhood`.  For
[mathematical morphology](https://en.wikipedia.org/wiki/Mathematical_morphology)
operations, a neighborhood is called a *structuring element*.

Denoting `A` the source array and `B` the neighborhood or the kernel (by
default `B` is a centered box of size 3 along every dimension), the available
filters are:

* `erode(A, B=3)` performs an erosion (local minimum) of `A` by `B`;

* `dilate(A, B=3)` performs a dilation (local maximum) of `A` by `B`;

* `localextrema(A, B=3)` yields the erosion and the dilation of `A` by `B`;

* `opening(A, B=3)` performs an erosion followed by a dilation;

* `closing(A, B=3)` performs a dilation followed by an erosion;

* `top_hat(A, B=3 [, S])` performs a summit detection (an optional third
  argument `S` may be supplied to pre-smooth `A` by `S`);

* `bottom_hat(A, B=3 [, S])` performs a valley detection (an optional third
  argument `S` may be supplied to pre-smooth `A` by `S`);

* `localmean(A, B=3)` performs a local averaging;

* `convolve(A, B=3)` performs a convolution by the kernel `B` or by the support
  of `B` is `eltype(B)` is `Bool`;

* `bilateralfilter(A, Fr, Gs)` performs a bilateral filtering of array `A` with
  `Fr` the range kernel for smoothing differences in intensities and `Gs` the
  spatial kernel for smoothing differences in coordinates (see:
  https://en.wikipedia.org/wiki/Bilateral_filter).

and many more to come...


## Implementation

### General local filters

The pseudo-code for a local filtering operation `C = filter(A, B)` writes:

```julia
for i ∈ Sup(A)
    v = initial(A[i])
    for j ∈ Sup(B) and such that i-j ∈ Sup(A)
        v = update(v, A[i-j], B[j])
    end
    store(C, i, v)
end
```

where `A` is the source of the operation, `B` is the neighborhood, `C` is the
result of the operation.  Here `Sup(A)` denotes the support of `A` (that is the
set of indices in `A`).  The methods `initial`, `update` and `store` and the
type of the variable `v` are specific to the considered operation.  For maximum
efficiency, the methods `initial` and `update` shall return the same type of
result.  To execute the filter, call the `localfilter!` method as:

```julia
localfilter!(dst, A, B, initial, update, store)
```

`localfilter!` applies the local filter defined by the neighborhood `B` and
methods `initial`, `update` and `store` to the source array `A` and stores the
result in the destination `dst` which is then returned.

For instance, to compute a local maximum (*i.e.* a **dilation** in mathematical
morphology terms):

```julia
initial(a) = typemin(typeof(a))
update(v,a,b) = (b && v < a ? a : v)
store(C,i,v) = C[i] = v
```

with `typeof(a)` the type of the elements of `A`.  Of course, anonymous functions can
be exploited to implement this filter in a single call:

```julia
localfilter!(dst, A, B,
             (a)     -> typemin(typeof(a)),   # initial method
             (v,a,b) -> (b && v < a ? a : v), # update method
             (C,i,v) -> C[i] = v)             # store method
```

Below is another example of the methods needed to implement a local average:

```julia
initial(a) = (0, zero(T))
update(v,a,b) = v[1] + 1, v[2] + (b ? a : zero(T))
store(C,i,v) = C[i] = v[2]/v[1]
```

with `T = typeof(a)`.

The same mechanism can be used to implement other operations such as
convolution, median filtering, *etc.* via the `localfilter!` driver.


### Fast separable local filters

When the filter amounts to combining all elements in a rectangular neighborhood
by an associative binary operation (`+`, `min`, `max`, *etc.*), the van
Herk-Gil-Werman algorithm can be used to implement the filter.  This algorithm
is much faster than a naive implementation (about `3N` operations per element
for a `N`-dimensional array whatever the size of the neighborhood instead of
`p^N - 1` operations for a neighborhood of lenght `p` along all the `N`
dimensions).  Another advantage of the van Herk-Gil-Werman algorithm is that it
can be applied in-place.  Such a filter is said to be *separable* and can be
applied along each dimension, one at a time.

The syntax to apply a separable local filter is:

```julia
localfilter!([dst = A,] A, dims, op, rngs [, w])
```

which overwrites the contents of `dst` with the result of applying van
Herk-Gil-Werman algorithm to filter array `A` along dimension(s) `dims` with
(associative) binary operation `op` and contiguous structuring element(s)
defined by the interval(s) `rngs`.  Optional argument `w` is a workspace array
which is automatically allocated if not provided; otherwise, it must be a
vector of same element type as `A` which is automatically resized as needed.
The destination `dst` must have the same indices as the source `A` (*i.e.*
`axes(dst) == axes(A)` must hold).  Operation can be done in-place; that is,
`dst` and `A` can be the same (this is the implemented behavior if `dst` is not
supplied).

Argument `dims` specifies along which dimension(s) of `A` the filter is to be
applied, it can be a single integer, several integers or a colon `:` to specify
all dimensions.  Dimensions are processed in the order given by `dims` (the
same dimension may appear several times) and there must be a matching interval
in `rngs` to specify the structuring element (except that if `rngs` is a single
interval, it is used for every dimension in `dims`).  An interval is either an
integer or an integer unit range in the form `kmin:kmax` (an interval specified
as a single integer, say `k`, is the same as specifying `k:k`).

Assuming mono-dimensional arrays `A` and `dst`, the single filtering pass:

```julia
localfilter!(dst, A, :, op, rng)
```

yields:

```
dst[j] = A[j-kmax] ⋄ A[j-kmax+1] ⋄ A[j-kmax+2] ⋄ ... ⋄ A[j-kmin]
```

for all `j ∈ [first(axes(A,1)):last(axes(A,1))]`, with `x ⋄ y = op(x, y)`,
`kmin = first(rng)` and `kmax = last(rng)`.  Note that if `kmin = kmax = k`
(which occurs if `rng` is a simple integer), the result of the filter is to
operate a simple shift by `k` along the corresponding dimension and has no
effects if `k = 0`.  This can be exploited to not filter some dimension(s).

For instance:

```julia
localfilter!(A, :, min, -3:3)
```

overwrites `A` with its *morphological erosion* (local minimum) on a
centered structuring element of width 7 in every dimension.

Another example, assuming `A` is a three-dimensional array:

```julia
localfilter!(A, :, max, (-3:3, 0, -4:4))
```

overwrites `A` its *morphological dilation* (*i.e.* local maximum) in a
centered local neighborhood of size `7×1×9` (nothing is done along the second
dimension).  The same result may be obtained with:

```julia
localfilter!(A, (1,3), max, (-3:3, -4:4))
```

where the second dimension is omitted from the list of dimensions.
The out-place version, allocates the destination array and is called as:

```julia
localfilter([T,] A, dims, op, rngs [, w])
```

with `T` the element type of the result (by default `T = eltype(A)`).

### References

* Marcel van Herk, "*A fast algorithm for local minimum and maximum filters on
  rectangular and octagonal kernels*" in Pattern Recognition Letters **13**,
  517-521 (1992).

* Joseph Gil and Michael Werman, "*Computing 2-D Min, Median, and Max Filters*"
  in IEEE Transactions on Pattern Analysis and Machine Intelligence **15**,
  504-507 (1993).

## Neighborhoods

`Neighborhood` (a.k.a. *structuring element* for the adepts of mathematical
morphology) is a central concept in `LocalFilters`.  The neighborhood defines
which values are involved in a local operation for each component of the source
array.  Neighborhoods are assumed to be shift invariant but may have any
support shape and may have embedded weights (*e.g.*, to implement *local
convolution*).


### Types of neighborhoods

From the user point of view, there are three kinds of neighborhoods:

* **Rectangular boxes** are rectangular neighborhoods whose edges are aligned
  with the axes of array indices and which may be centered or have arbitrary
  offsets along the dimensions.  These neighborhoods are represented by
  instances of `LocalFilters.RectangularBox`.

* **Arbitrarily shaped neighborhoods** are neighborhoods with arbitrary shape
  and offset.  These neighborhoods are represented by instances of
  `LocalFilters.Kernel` with boolean element type.  These neighborhoods are
  constructed from an array of booleans and an optional starting index.

* **Kernels** are neighborhoods whose elements are weights and which may have
  arbitrary offset.  These neighborhoods are represented by instances of
  `LocalFilters.Kernel` with numerical element type.  These neighborhoods are
  constructed from an array of weights and an optional starting index.


### Syntax for neighborhoods


* The *default neighborhood* is a centered rectangular box of width 3 in each
  of its dimensions.

* A *scalar* integer `w` yields a centered rectangular box of size `w` along
  all dimensions.  `w` must be at least equal to 1 and the geometrical center
  of the box is defined according to the conventions in `fftshift`.

* A *tuple* `t` of integers yields a centered rectangular box whose size is
  `t[i]` along the `i`-th dimension.  All values of `t` must be larger or equal
  to 1.  Tip: Remember that you can use `v...` to convert a *vector* `v` into a
  tuple.

* An *array* `A` yields a `LocalFilters.Kernel` whose coefficients are the
  values of `A` and whose neighborhood is the centered bounding-box of `A`.

* A *Cartesian region* `R` (an instance of `CartesianIndices`) yields a
  `LocalFilters.RectangularBox` which is a rectangular neighborhood whose
  support contains all relative positions within `first(R)` and `last(R)`.

* A rectangular box neighborhood created by calling
  `LocalFilters.RectangularBox` as:

  ```julia
  LocalFilters.RectangularBox(R)
  LocalFilters.RectangularBox(I1, I2)
  LocalFilters.RectangularBox(dims, offs)
  LocalFilters.RectangularBox(inds)
  ```

  where `R` is an instance of`CartesianIndices`, `I1` and `I2` are two
  `CartesianIndex` specifying the first and last relative position within the
  neighborhood, `dims` and `offs` are tuples of integers specifying the
  dimensions of the neighborhood and its offsets, `inds` are unit ranges.

  Assuming `dim` is an integer, then:

  ```julia
  LocalFilters.RectangularBox{N}(dim)
  ```

  yields an `N`-dimensional rectangular box of size `dim` along all dimensions
  and centered at the geometrical center of the box (with the same conventions
  as `fftshift`).

  Similarly, assuming `i1` and `i2` are integers, then:

  ```julia
  LocalFilters.RectangularBox{N}(i1:i2)
  ```

  yields an `N`-dimensional rectangular box with index range `i1:i2` along all
  dimensions.


### Methods on a neighborhood

The following statements make sense on a neighborhood `B`:

```julia
eltype(B) -> element type of B
ndims(B)  -> number of dimensions of B
length(B) -> number of elements in the bounding-box of B
size(B)   -> size of the bounding-box of B along all dimensions
size(B,d) -> size of the bounding-box of B along d-th dimension
first(B)  -> CartesianIndex of first position in the bounding-box
             of B relative to its anchor
last(B)   -> CartesianIndex of last position in the bounding-box
             of B relative to its anchor
B[i]      -> yields the kernel value of B at index i
```

Note that the index `i` in `B[i]` is assumed to be between `first(B)` and
`last(B)`, for efficiency reasons this is not checked.  The type returned by
`eltype(B)` is `Bool` for a neighborhood which is just defined by its support
(*e.g.* a `LocalFilters.CenteredBox` or a `LocalFilters.RectangularBox`), the
element type of its kernel otherwise.

```julia
CartesianIndices(B)
```

yields the Cartesian indices of relative positions of the bounding-box of
neighborhood `B`.

If the argument `B` which defines a neighborhood (see previous section) is not
an instance of a type derived from `LocalFilters.Neighborhood`, it may be
explicitly converted by:

```julia
convert(LocalFilters.Neighborhood{N}, B)
```

with `N` the number of dimensions of the target array.


## Installation

To install the last official version:

```julia
using Pkg
Pkg.add("LocalFilters")
```

To use the last development version, install with Pkg, the Julia package
manager, as an unregistered Julia package (press the ] key to enter the Pkg
REPL mode):

```julia
... pkg> add https://github.com/emmt/LocalFilters.jl.git
```

The `LocalFilters` package is pure Julia code and there is nothing to build.
