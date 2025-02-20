# Reference

The following summarizes the documentation of types and methods provided by the
`LocalFilters` package. This information is also available from the REPL by typing `?`
followed by the name of a method or a type.


## Index

```@index
```

## Simple linear filters

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

## Methods to build local filters

```@docs
localfilter
localfilter!
```

## Constants

```@docs
FORWARD_FILTER
REVERSE_FILTER
```

## Neighborhoods and kernels

```@docs
kernel
reverse_kernel
strel
LocalFilters.ball
LocalFilters.centered
```

## Utilities

Below are described non-exported types and methods that may be useful in the context of
building local filters.

```@docs
LocalFilters.Indices
LocalFilters.localindices
LocalFilters.kernel_range
```
