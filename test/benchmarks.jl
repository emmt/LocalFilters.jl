#
# benchmarks.jl --
#
# Some speed tests for local filters. Among others:
#
# * Naive (but otherwise fast) versions of filters which does not use the
#   `localfilter!` driver nor the smart algorithms implemented in LocalFilters;

isdefined(Main, :LocalFilters) || include("../src/LocalFilters.jl")
isdefined(Main, :NaiveLocalFilters) || include("NaiveLocalFilters.jl")

module LocalFiltersBenchmarks

const AUTORUN = true

using Compat
using Printf
using LocalFilters
using LocalFilters: Kernel, limits, cartesian_region, axes

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
function similarvalues(A::AbstractArray{T,N}, B::AbstractArray{T,N};
                       atol=0.0, gtol=0.0) where {T,N}
    @assert axes(A) == axes(B)
    local sd2::Float64 = 0.0
    local sa2::Float64 = 0.0
    local sb2::Float64 = 0.0
    @inbounds for i in eachindex(A, B)
        a = Float64(A[i])
        b = Float64(B[i])
        sa2 += a*a
        sb2 += b*b
        sd2 += (a - b)^2
    end
    return sqrt(sd2) ≤ atol + gtol*sqrt(max(sa2, sb2))
end

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
    T = Float64
    dims = (64, 81)
    n = 1000
    a = rand(T, dims)
    box = RectangularBox(3,5)
    rngs = (-1:1, -2:2)
    mask = Kernel(box)
    kern = Kernel{eltype(a)}(ones(size(box)))
    a0 = similar(a)
    a1 = similar(a)
    a2 = similar(a)
    a3 = similar(a)
    a4 = similar(a)
    a5 = similar(a)

    tests = (:Base, :NTuple, :NTupleVar, :Map)
    for (name, B) in (("RectangularBox", box),
                      ("boolean kernel", mask),
                      ("non-boolean kernel", kern))
        println("\nErosion on a $name (timings on $n iterations):")
        erode!(Val(:Base), a0, a, B)
        for v in (B === box ? (tests..., :Naive) : tests)
            erode!(Val(v), a1, a, B)
            @printf "   Variant %-10s (" v
            checkresult(samevalues(a1, a0))
            print("): ")
            @time for i in 1:n; erode!(alg, a0, a, B); end
        end
        if B === box
            erode!(a1, a, :, rngs)
            @printf "   Variant %-10s (" "vHGW"
            checkresult(samevalues(a1, a0))
            print("): ")
            @time for i in 1:n; erode!(a1, a, :, rngs); end
        end

        println("\nDilation on a $name (timings on $n iterations):")
        dilate!(Val(:Base), a0, a, B)
        for v in (B === box ? (tests..., :Naive) : tests)
            dilate!(Val(v), a1, a, B)
            @printf "   Variant %-10s (" v
            checkresult(samevalues(a1, a0))
            print("): ")
            @time for i in 1:n; dilate!(Val(v), a1, a, B); end
        end
        if B === box
            dilate!(a1, a, :, rngs)
            @printf "   Variant %-10s (" "vHGW"
            checkresult(samevalues(a1, a0))
            print("): ")
            @time for i in 1:n; dilate!(a1, a, :, rngs); end
        end

        println("\nErosion and dilation on a $name (timings on $n iterations):")
        erode!(Val(:Base), a0, a, B)
        dilate!(Val(:Base), a1, a, B)
        for v in tests
            localextrema!(Val(v), a2, a3, a, B)
            @printf "   Variant %-10s (" v
            checkresult(samevalues(a2, a0) && samevalues(a3, a1))
            print("): ")
            @time for i in 1:n; localextrema!(Val(v), a2, a3, a, B); end
        end

        println("\nLocal mean on a $name (timings on $n iterations):")
        localmean!(Val(:Base), a0, a, B)
        for v in tests
            localmean!(Val(v), a1, a, B)
            @printf "   Variant %-10s (" v
            checkresult(similarvalues(a1, a0; gtol=4*eps(Float64)))
            print("): ")
            @time for i in 1:n; localmean!(Val(v), a1, a, B); end
        end

        if isa(B, Kernel) && eltype(B) != Bool
            println("\nConvolution by a $name (timings on $n iterations):")
            convolve!(Val(:Base), a0, a, B)
            for v in tests
                convolve!(Val(v), a1, a, B)
                @printf "   Variant %-10s (" v
                checkresult(similarvalues(a1, a0; gtol=4*eps(Float64)))
                print("): ")
                @time for i in 1:n; convolve!(Val(v), a1, a, B); end
            end
        end
        println()
    end

end

end # module
