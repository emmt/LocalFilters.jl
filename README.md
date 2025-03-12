# LocalFilters

[![Doc. Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://emmt.github.io/LocalFilters.jl/dev)
[![Doc. Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://emmt.github.io/LocalFilters.jl/stable)
[![License](http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)](./LICENSE.md)
[![Build Status](https://github.com/emmt/LocalFilters.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/emmt/LocalFilters.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![Build Status](https://ci.appveyor.com/api/projects/status/github/emmt/LocalFilters.jl?branch=master)](https://ci.appveyor.com/project/emmt/LocalFilters-jl/branch/master)
[![codecov](https://codecov.io/github/emmt/LocalFilters.jl/graph/badge.svg?token=aA8yUwB2En)](https://codecov.io/github/emmt/LocalFilters.jl)

[Julia](http://julialang.org/) package `LocalFilters` implements multi-dimensional local
filters such as discrete convolution or correlation, local mean, mathematical morphology,
etc., and provides support to build custom local filters.

The [Reference Manual](https://emmt.github.io/LocalFilters.jl/dev) provides more
exhaustive documentation. This page summarizes the principles and the features of
`LocalFilters`:

* [Available filters](#available-filters) lists ready to use filters.

* [Kernels and neighborhoods](#kernels-and-neighborhoods) describes the concept of
  *kernels* and *neighborhoods* which define which (and how) values are involved in a
  local filter.

* [Build your own filters](#build-your-own-filters) explains how to implement custom local
  filters.

* [Local mapping and reduction](#local-mapping-and-reduction) describes methods for
  computing local mappings and reductions.

* [Installation](#installation) gives instructions to install the package.

Packages with overlapping functionalities:

* [ImageFiltering](https://github.com/JuliaImages/ImageFiltering.jl) for local filters on
  multidimensional arrays (not just *images*), also implement various boundary conditions;

* [ImageMorphology](https://github.com/JuliaImages/ImageMorphology.jl) for fast
  morphological operations with separable structuring elements.


## Available filters

`LocalFilters` provides a number of linear and non-linear filters. All methods have an
*in-place* counterpart which can be called to avoid allocations.

### Linear filters

`LocalFilters` provides the following linear filters:

* `localmean(A,B=3)` performs a local averaging of `A` in a neighborhood defined by `B`.

* `correlate(A,B)` performs a discrete correlation of `A` by the kernel `B`. This is the
  most general linear filter.

* `convolve(A,B)` performs a discrete convolution of `A` by the kernel `B`. This is the
  same as a discrete correlation of `A` by `reverse_kernel(B)`.


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
an array of Booleans to define a more complex neighborhood shape. If `B` is a single odd
integer (as it is by default), the structuring element is assumed to be a centered sliding
window of size `B` along every dimension of `A`.


### Other non-linear filters

`LocalFilters` provides an instance of the [bilateral
filter](https://en.wikipedia.org/wiki/Bilateral_filter):

* `bilateralfilter(A,F,G,B)` performs a bilateral filtering of array `A` with `F` the
  range kernel for smoothing differences in values, `G` the spatial kernel for smoothing
  differences in coordinates, and `B` the neighborhood. Alternatively one can specify the
  range and spatial parameters `bilateralfilter(A,σr,σs,B=2*round(Int,3σs)+1)` for using
  Gaussian kernels with standard deviations `σr` and `σs`.


## Kernels and neighborhoods

*Neighborhoods* define which array elements around the element of interest are involved in
the result of the local filter. Neighborhoods can be represented by arrays of Booleans,
they are also known as *sliding windows* in image processing or as *structuring element*
in mathematical morphology. Neighborhoods whose elements are uniformly true are equivalent
to hyper-rectangular sliding wondows whose axes are aligned with the Cartesian axes. A
kernel is similar to a neighborhood but its elements have values which represent the
coefficients or the weights of the filter. Kernels and neighborhoods are built by the
`kernel` function.


## Build your own filters

In `LocalFilters`, a local filtering operation, say `dst = filter(A, B)` with `A` the
source of the operation and `B` the neighborhood or the kernel associated with the filter,
is implemented by the following pseudo-code:

```julia
for i ∈ indices(dst)
    v = initial isa Function ? initial(A[i]) : initial
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
`LocalFilters`, indices `i` and `j` are Cartesian indices for multi-dimensional arrays,
thus `indices(A)` is the analogous of `CartesianIndices(A)` in Julia in that case. For
vectors, indices `i` and `j` are linear indices.

The behavior of the filter is completely determined by the neighborhood or kernel `B`, by
the type of the state variable `v` initialized by `initial` for each entry of the
destination, and by the methods `update` and `final`.

Such a filter can be applied by calling `localfilter!` as:

```julia
localfilter!(dst, A, B, initial, update, final = identity) -> dst
```

When `initial` is a function, `A` and `dst` must have the same indices; `localfilter!`
will throw a `DimensionMismatch` if this is not the case.

As shown by the following examples, this simple scheme allows the implementation of a
variety of linear and non-linear local filters:

* Implementing a **local average** of `A` in a neighborhood defined by an array `B` of
  Booleans is done with:

  ```julia
  localfilter!(dst, A, B,
               #= initial =# (; num = zero(eltype(A)), den = 0),
               #= update  =# (v,a,b) -> ifelse(b, (; num = v.num + a, den = v.den + 1), v),
               #= final   =# (v) -> v.num / v.den)
  ```

* Assuming `T = eltype(dst)` is a suitable element type for the result, a **discrete
  correlation** of `A` by `B` can be implemented with:

  ```julia
  localfilter!(dst, A, B,
               #= initial =# zero(T),
               #= update  =# (v,a,b) -> v + a*b)
  ```

  There are no needs to specify the `final` method here, as the default `final =
  identity`, does the job.

* Computing a local maximum (that is, a **dilation** in mathematical morphology terms) of
  array `A` with a kernel `B` whose entries are Booleans can be done with:

  ```julia
  localfilter!(dst, A, B,
               #= initial =# typemin(eltype(A)),
               #= update  =# (v,a,b) -> ((b & (v < a)) ? a : v))
  ```

  As in the above example, there are no needs to specify the `final` method here. Note the
  use of a bitwise `&` instead of a `&&` in the `update` method to avoid branching.


## Local mapping and reduction

Method `localmap(f,A,B)` yields the result of applying the function `f` to the vector of
values of `A` in neighborhoods defined by `B`. Method `localmap!(f,dst,A,B)` is the
in-place version of `localmap(f,A,B)`.

Method `localreduce(op,A,dims,rngs)` applies the van Herk-Gil-Werman algorithm to compute
the reduction by the associative binary operator `op` of the values of `A` into contiguous
hyper-rectangular neighborhoods defined by the interval(s) `rngs` along dimension(s)
`dims` of `A`. Method `localreduce!(op,dst,A,dims,rngs)` is the in-place version of
`localreduce(op,A,dims,rngs)`.


## Installation

To install the last official version, press the `]` key to enter Julia's `Pkg` REPL mode
and type at the `... pkg>` prompt:

```julia
add LocalFilters
```

To use the last development version, type instead:

```julia
add https://github.com/emmt/LocalFilters.jl
```

The `LocalFilters` package is pure Julia code and nothing has to be build.
