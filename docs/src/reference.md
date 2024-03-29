# Reference

The following summarizes the documentation of types and methods provided by the
`LocalFilters` package. This information is also available from the REPL by
typing `?` followed by the name of a method or a type.


## Index

```@index
```

## Linear filters

`LocalFilters` provides a number of shift-invariant linear filters.

```@docs
correlate
correlate!
convolve
convolve!
localmean
localmean!
```

## Mathematical morphology

```@docs
erode
erode!
dilate
dilate!
localextrema
localextrema!
closing
closing!
opening
opening!
top_hat
LocalFilters.top_hat!
bottom_hat
LocalFilters.bottom_hat!
```

## Other non-linear filters

```@docs
bilateralfilter
bilateralfilter!
```

## Generic driver for custom local filters

```@docs
localfilter
localfilter!
```

## Constants

```@docs
ForwardFilter
ReverseFilter
```

## Neighborhoods and kernels

```@docs
kernel
strel
LocalFilters.ball
LocalFilters.centered
```

## Utilities

```@docs
LocalFilters.Indices
LocalFilters.Yields
LocalFilters.localindices
LocalFilters.nearest
LocalFilters.replicate
LocalFilters.kernel_range
LocalFilters.multiply_add
LocalFilters.multiply_add!
```
