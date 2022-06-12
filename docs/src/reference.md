# Reference

The following summarizes the documentation of types and methods provided by the
`LocalFilters` package.  This information is also available from the REPL by
typing `?` followed by the name of a method or a type.

## Linear filters

`LocalFilters` provides convolutive filters which are shift-invariant linear
filters.

```@docs
localmean
localmean!
convolve
convolve!
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
strel
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

## Types

```@docs
Neighborhood
RectangularBox
LocalFilters.Kernel
LocalFilters.ConstantProducer
```

## Utilities

```@docs
LocalFilters.default_start
LocalFilters.ball
```
