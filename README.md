# LocalFilters.jl

[![Doc. Dev][doc-dev-img]][doc-dev-url]
[![Doc. Stable][doc-stable-img]][doc-stable-url]
[![License][license-img]][license-url]
[![Build Status][github-ci-img]][github-ci-url]
[![Build Status][appveyor-img]][appveyor-url]
[![Coverage][codecov-img]][codecov-url]

[Julia](http://julialang.org/) package `LocalFilters` implements multi-dimensional local
filters such as discrete convolution, local mean, mathematical morphology, etc., and
provides support to build custom local filters.

This page summarizes the principles and the features of `LocalFilters`, the [Reference
Manual](doc-dev-url) provides more exhaustive documentation. This document is structured
as follows:

* [Available filters](#available-filters) lists ready to use filters.

* [Neighborhoods](#neighborhoods) describes the concept of *neighborhoods*, also known as
  *sliding windows* in image processing or *structuring element* in mathematical
  morphology.

* [Build your own filters](#build-your-own-filters) explains how to implement custom local
  filters.

* [Installation](#installation) gives instructions to install the package.

Packages with overlapping functionalities:

* [ImageFiltering](https://github.com/JuliaImages/ImageFiltering.jl) for local filters on
  multidimensional arrays (not just *images*), also implement various boundary conditions;

* [ImageMorphology](https://github.com/JuliaImages/ImageMorphology.jl) for fast
  morphological operations with separable structuring elements;


## Available filters

`LocalFilters` provides a number of linear and non-linear filters. All methods have an
*in-place* counterpart which can be called to avoid allocations.

### Linear filters

`LocalFilters` provides the following linear filters:

* `localmean(A,B=3)` performs a local averaging of `A` in a neighborhood defined by `B`.

* `correlate(A,B)` performs a discrete correlation of `A` by the kernel `B`. This is the
  most general linear filter.

* `convolve(A,B)` performs a discrete convolution of `A` by the kernel `B`. This is the
  same as a discrete correlation of `A` by the symmetrical of `B`.


### Mathematical morphology

`LocalFilters` implements the following [mathematical
morphology](https://en.wikipedia.org/wiki/Mathematical_morphology) operations:

* `erode(A,B=3)` performs an erosion (local minimum) of `A` by the structuring element
  `B`;

* `dilate(A,B=3)` performs a dilation (local maximum) of `A` by the structuring element
  `B`;

* `localextrema(A,B=3)` yields the erosion and the dilation of `A` by the structuring
  element `B`;

* `opening(A,B=3)` performs an erosion followed by a dilation of `A` by the structuring
  element `B`;

* `closing(A,B=3)` performs a dilation followed by an erosion of `A` by the structuring
  element `B`;

* `top_hat(A,B=3[,S])` performs a summit detection of `A` by the structuring element `B`
  (argument `S` may be optionally supplied to pre-smooth `A` by `S`);

* `bottom_hat(A,B=3[,S])` performs a valley detection of `A` by the structuring element
  `B` (argument `S` may be optionally supplied to pre-smooth `A` by `S`).

In mathematical morphology, the structuring element `B` defines the local neighborhood of
each index in the source array. It can be a sliding hyper-rectangular Cartesian window or
an array of booleans to define a more complex neighborhood shape. If `B` is a single odd
integer (as it is by default), the structuring element is assumed to be a sliding window
of size `B` along every dimension of `A`.


### Other non-linear filters

`LocalFilters` provides an instance of the [bilateral
filter](https://en.wikipedia.org/wiki/Bilateral_filter):

* `bilateralfilter(A,F,G,B)` performs a bilateral filtering of array `A` with `F` the
  range kernel for smoothing differences in values, `G` the spatial kernel for smoothing
  differences in coordinates, and `B` the neighborhood. Alternatively one can specify the
  range and spatial parameters `bilateralfilter(A,σr,σs,B=2*round(Int,3σs)+1)` for using
  Gaussian kernels with standard deviations `σr` and `σs`.


## Build your own filters

In `LocalFilters`, a local filtering operation, say `dst = filter(A, B)` with `A` the
source of the operation and `B` the neighborhood or the kernel associated with the filter,
is implemented by the following pseudo-code:

```julia
for i ∈ indices(dst)
    v = initial
    for j ∈ indices(A) ∩ (indices(B) + i)
        v = update(v, A[j], B[j-i])
    end
    dst[i] = final(v)
end
```

where `indices(A)` denotes the set of indices of `A` while `indices(B) + i` denotes the
set of indices `j` such that `j - i ∈ indices(B)` with `indices(B)` the set of indices of
`B`. In other words, `j ∈ indices(A) ∩ (indices(B) + i)` means all indices `j` such that
`j ∈ indices(A)` and `j - i ∈ indices(B)`, hence `A[j]` and `B[j-i]` are in-bounds. In
`LocalFilters`, indices `i` and `j` are multi-dimensional Cartesian indices, thus
`indices(A)` is the analogous of `CartesianIndices(A)` in Julia.

The behavior of the filter is completely determined by the neighborhood or kernel `B`, by
the type of the state variable `v`, its `initial` value for each entry of the destination,
and by the methods `update` and `final`.

Such a filter can be applied by calling `localfilter!` as:

```julia
localfilter!(dst, A, B, initial, update, final = identity) -> dst
```

As shown by the following examples, this simple scheme allows the implementation of a
variety of linear and non-linear local filters:

* Implementing a **local average** of `A` in a neighborhood defined by an array `B` of
  booleans is done with:

  ```julia
  localfilter!(dst, A, B,
               (zero(a), 0), # initial, state = (sum_of_values, count_of_values)
               (v,a,b) -> ifelse(b, (v[1] + a, v[2] + 1), v), # update
               (v) -> v[1]/v[2]) # final
  ```

* Assuming `T = eltype(dst)` is a suitable element type for the result, a **discrete
  convolution** of `A` by `B` can be implemented with:

  ```julia
  localfilter!(dst, A, B,
               zero(T), # initial
               (v,a,b) -> v + a*b) # update
  ```

  There are no needs to specify the `final` method here, as the default `final=identity`,
  does the job.

* Computing a local maximum (that is, a **dilation** in mathematical morphology terms) of
  array `A` with a kernel `B` whose entries are booleans can be done with:

  ```julia
  localfilter!(dst, A, B,
               typemin(a), # initial
               (v,a,b) -> ((b & (v < a)) ? a : v)) # update
  ```

  As in the above example, there are no needs to specify the `final` method here. Note the
  use of a bitwise `&` instead of a `&&` in the `update` method to avoid branching.


## Installation

To install the last official version:

```julia
using Pkg
pkg"add LocalFilters"
```

To use the last development version, install with Pkg, the Julia package manager, as an
unregistered Julia package (press the ] key to enter the Pkg REPL mode):

```julia
using Pkg
pkg"add https://github.com/emmt/LocalFilters.jl"
```

The `LocalFilters` package is pure Julia code and nothing has to be build.

[doc-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[doc-stable-url]: https://emmt.github.io/LocalFilters.jl/stable

[doc-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[doc-dev-url]: https://emmt.github.io/LocalFilters.jl/dev

[license-url]: ./LICENSE.md
[license-img]: http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat

[github-ci-img]: https://github.com/emmt/LocalFilters.jl/actions/workflows/CI.yml/badge.svg?branch=master
[github-ci-url]: https://github.com/emmt/LocalFilters.jl/actions/workflows/CI.yml?query=branch%3Amaster

[appveyor-img]: https://ci.appveyor.com/api/projects/status/github/emmt/LocalFilters.jl?branch=master
[appveyor-url]: https://ci.appveyor.com/project/emmt/LocalFilters-jl/branch/master

[codecov-img]: http://codecov.io/github/emmt/LocalFilters.jl/coverage.svg?branch=master
[codecov-url]: http://codecov.io/github/emmt/LocalFilters.jl?branch=master

[julia-url]: https://julialang.org/
[julia-pkgs-url]: https://pkg.julialang.org/
