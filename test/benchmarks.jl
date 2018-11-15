#
# benchmarks.jl --
#
# Some speed tests for local filters.  Among others:
#
# * Naive (but otherwise fast) versions of filters which does not use the
#   `localfilter!` driver nor the smart algorithms implemented in LocalFilters;

isdefined(Main, :LocalFilters) || include("../src/LocalFilters.jl")
isdefined(Main, :NaiveLocalFilters) || include("NaiveLocalFilters.jl")

module LocalFiltersBenchmarks

const AUTORUN = true

using Compat
using Compat.Printf
using LocalFilters
using LocalFilters: Kernel, limits, cartesianregion

import Base: eltype, ndims, size, length, first, last, tail,
    getindex, setindex!, convert

@static if isdefined(Base, :CartesianIndices)
    import Base: CartesianIndices
else
    import Base: CartesianRange
    import Compat: CartesianIndices
end

replicate(a, n::Integer) = ntuple(i->a, n)
compare(a, b) = maximum(abs(a - b))
samevalues(a, b) = maximum(a == b)

#------------------------------------------------------------------------------

function checkresult(txt, tst::Bool)
    print(txt)
    checkresult(tst)
end

function checkresult(tst::Bool)
    if tst
        printstyled("ok"; color=:green)
    else
        printstyled("no"; color=:red)
    end
end

if AUTORUN
    n = 1000
    a = rand(62,81)
    box = RectangularBox(3,5)
    mask = Kernel(box)
    kern = Kernel(eltype(a), mask)
    a0 = similar(a)
    a1 = similar(a)
    a2 = similar(a)
    a3 = similar(a)
    a4 = similar(a)
    a5 = similar(a)

    tests = (:Base, :NTuple, :NTupleVar, :Map)
    for (name, box) in (("RectangularBox", box),
                        ("boolean kernel", mask),
                        ("non-boolean kernel", kern))
        println("\nErosion on a $name (timings on $n iterations):")
        erode!(Val{:Base}, a0, a, box)
        for v in tests
            erode!(Val{v}, a1, a, box)
            @printf "   Variant %-10s (" v
            checkresult(samevalues(a1, a0))
            print("): ")
            @time for i in 1:n; erode!(Val{v}, a0, a, box); end
        end

        println("\nDilation on a $name (timings on $n iterations):")
        dilate!(Val{:Base}, a0, a, box)
        for v in tests
            dilate!(Val{v}, a1, a, box)
            @printf "   Variant %-10s (" v
            checkresult(samevalues(a1, a0))
            print("): ")
            @time for i in 1:n; dilate!(Val{v}, a1, a, box); end
        end

        println("\nErosion and dilation on a $name (timings on $n iterations):")
        erode!(Val{:Base}, a0, a, box)
        dilate!(Val{:Base}, a1, a, box)
        for v in tests
            localextrema!(Val{v}, a2, a3, a, box)
            @printf "   Variant %-10s (" v
            checkresult(samevalues(a2, a0) && samevalues(a3, a1))
            print("): ")
            @time for i in 1:n; localextrema!(Val{v}, a2, a3, a, box); end
        end
        println()
    end

end

end # module
