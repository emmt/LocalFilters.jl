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

**LocalFilters** implements local filtering operations which combine the values
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
set of indices in `A`).  The methods `initial`, `update` and `store` are
specific to the considered operation.  To execute the filter, call the
`localfilter!` method as:

```julia
localfilter!(dst, A, B, initial, update, store)
```

`localfilter!` applies the local filter defined by the neighborhood `B` and
methods `initial`, `update` and `store` to the source array `A` and stores the
reseult in the destination `dst` which is then returned.

For instance, to compute a local maximum (*i.e.* a **dilation** in mathematical
morphology terms):

```julia
initial(a) = typemin(T)
update(v,a,b) = (b && v < a ? a : v)
store(c,i,v) = c[i] = v
```

with `T` the type of the elements of `A`.  Of course, anonymous functions can
be exploited to implement this filter in a single call:

```julia
localfilter!(dst, A, B,
             (a)     -> typemin(T),           # initial method
             (v,a,b) -> (b && v < a ? a : v), # update method
             (c,i,v) -> c[i] = v)             # store method
```

Below is another example of the methods needed to implement a local average:

```julia
initial(a) = (0, zero(T))
update(v,a,b) = v[1] + 1, v[2] + (b ? a : zero(T))
store(c,i,v) = c[i] = v[2]/v[1]
```

The same mechanism can be used to implement other operations such as
convolution, median filtering, *etc.* via the `localfilter!` driver.


## Neighborhoods

`Neighborhood` (a.k.a. *structuring element* for the fans of mathematical
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

* **Kernels** are neighborhoods whose elements are weights and which be may
  have arbitrary offset.  These neighborhoods are represented by instances of
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

* A *Cartesian range* `R` (an instance of `CartesianIndices` or of
  `CartesianRange`) yields a `LocalFilters.RectangularBox` which is a
  rectangular neighborhood whose support contains all relative positions within
  `first(R)` and `last(R)`.

* A rectangular box neighborhood created by calling
  `LocalFilters.RectangularBox` as:

  ```julia
  LocalFilters.RectangularBox(R)
  LocalFilters.RectangularBox(I1, I2)
  LocalFilters.RectangularBox(dims, offs)
  LocalFilters.RectangularBox(inds)
  ```

  where `R` is an instance of`CartesianIndices` (or of `CartesianRange` for old
  Julia version), `I1` and `I2` are two `CartesianIndex` specifying the first
  and last relative position within the neighborhood, `dims` and `offs` are
  tuples of integers specifying the dimensions of the neighborhood and its
  offsets, `inds` are unit ranges.

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

The following methods make sense on a neighborhood `B`:

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
CartesianRange(B)
```

yields the Cartesian range of relative positions of the bounding-box of
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

To use the last development version:

```julia
using Pkg
Pkg.clone("https://github.com/emmt/LocalFilters.jl.git")
```

The `LocalFilters` package is pure Julia code and there is nothing to build.
