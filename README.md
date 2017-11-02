# LocalFilters.jl

This package implements multi-dimensional local filters for Julia (convolution,
mathematical morphology, etc.).

**Cavehats:** This is a first implementation to define the API.  It is is
reasonably fast but optimizations will come later notably for separable
kernels.


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

and many more to come...


## Implementation

The pseudo-code for a local filtering operation `C = filter(A, B)` writes:

```julia
for i ∈ Sup(A)
    v = initial()
    for j ∈ Sup(A) and i-j ∈ Sup(B)
        v = update(v, A[j], B[i-j])
    end
    store(C, i, v)
end
```

where `A` is the source of the operation, `B` is the neighborhood, `C` is the
result of the operation.  Here `Sup(A)` denotes the support of `A` (that is the
set of indices in `A`).  The methods `initial`,`update` and `store` are
specific to the considered operation.  For instance, to compute a local maximum
(*i.e.* a **dilation** in mathematical morphology terms):

```julia
initial() = typemin(T)
update(v,a,b) = (b && v < a ? a : v)
store(c,i,v) = c[i] = v
```

with `T` the type of the elements of `A`.  To compute a local average:

```julia
initial() = (0, zero(T))
update(v,a,b) = v[1] + 1, v[2] + (b ? a : zero(T))
store(c,i,v) = c[i] = v[2]/v[1]
```

The same mechanism can be used to implement other operations such as
convolution, median filtering, *etc.* via the `localfilter!` driver.


## Types of Neighborhoods

There are many possible types of neighborhood:

* The *default neighborhood* is a `LocalFilters.CenteredBox` of width 3 in each
  of its dimensions.

* A *scalar* integer `w` yields a `LocalFilters.CenteredBox` of size `w` along
  all dimensions.  `w` must be an odd integer at least equal to 1.

* A *tuple* `t` of integers yields a `LocalFilters.CenteredBox` whose size is
  `t[i]` along the `i`-th dimension.  All values of `t` must an odd integers
  larger or equal to 1.  Tip: Remember that you can use `v...` to convert a
  *vector* `v` into a tuple.

* An *array* `A` yields a `LocalFilters.Kernel` whose coefficients are the
  values of `A` and whose neighborhood is the centered bounding-box of `A`.

* A *Cartesian range* `R` (an instance of `Base.CartesianRange`) yields a
  `LocalFilters.CartesianBox` which is a rectangular neighborhood whose support
  contains all relative positions within `first(R)` and `last(R)`.

* A centered rectangular box created with `LocalFilters.CenteredBox(dims)`
  with `dims` a tuple of odd integers (all ≥ 1).

* A rectangular box, possibly off-centered, created with one of:

  ```julia
  LocalFilters.CartesianBox(R)
  LocalFilters.CartesianBox(I0, I1)
  LocalFilters.CartesianBox(dims, offs)
  LocalFilters.CartesianBox(inds)
  ```

  where `R` is a `Base.CartesianRange`, `I0` and `I1` are two
  `Base.CartesianIndex` specifying the first and last relative position within
  the neighborhood, `dims` and `offs` are tuples of integers specifying the
  dimensions of the neighborhood and its anchor, `inds` are unit ranges.


## Methods on a Neighborhood

The following methods make sense on a neighborhood `B`:

```julia
eltype(B) -> element type of B
ndims(B)  -> number of dimensions of B
length(B) -> number of elements in the bounding-box of B
size(B)   -> size of the bounding-box of B along all dimensions
size(B,i) -> size of the bounding-box of B along i-th dimension
first(B)  -> CartesianIndex of first position in the bounding-box
             of B relative to its anchor
last(B)   -> CartesianIndex of last position in the bounding-box
             of B relative to its anchor
B[i]      -> yields the kernel value of `B` at index `i`
```

Note that the index `i` in `B[i]` is assumed to be between `first(B)` and
`last(B)`, for efficiency reasons this is not checked.  The type returned by
`eltype(B)` is `Bool` for a neighborhood which is just defined by its support
(*e.g.* a `LocalFilters.CenteredBox` or a `LocalFilters.CartesianBox`), the
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
