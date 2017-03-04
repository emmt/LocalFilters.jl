# benchmarks.jl --
#
# Some speed tests for local filters.  Among others:
#
# * "Fast" versions of filters which does not use the `localfilter!`
#   driver;

module LocalFiltersBenchmarks

const AUTORUN = true

using LocalFilters
import LocalFilters: Neighborhood, CenteredBox, CartesianBox, Kernel,
    anchor, limits, coefs
import Base.CartesianRange

replicate(a, n::Integer) = ntuple(i->a, n)
compare(a, b) = maximum(abs(a - b))
samevalues(a, b) = maximum(a == b)

#------------------------------------------------------------------------------
# "Fast" versions fo centered boxes.

function fastlocalmean!{T,N}(dst::AbstractArray{T,N},
                             A::AbstractArray{T,N},
                             B::CenteredBox{N})
    @assert size(dst) == size(A)
    R = CartesianRange(size(A))
    imin, imax = limits(R)
    off = last(B)
    @inbounds for i in R
        n, s = 0, zero(T)
        for j in CartesianRange(max(imin, i - off), min(imax, i + off))
            s += A[j]
            n += 1
        end
        dst[i] = s/n
    end
    return dst
end

function fasterode!{T,N}(Amin::AbstractArray{T,N},
                         A::AbstractArray{T,N},
                         B::CenteredBox{N})
    @assert size(Amin) == size(A)
    R = CartesianRange(size(A))
    imin, imax = limits(R)
    tmax = typemax(T)
    off = last(B)
    @inbounds for i in R
        vmin = tmax
        for j in CartesianRange(max(imin, i - off), min(imax, i + off))
            vmin = min(vmin, A[j])
        end
        Amin[i] = vmin
    end
    return Amin
end

function fastdilate!{T,N}(Amax::AbstractArray{T,N},
                          A::AbstractArray{T,N},
                          B::CenteredBox{N})
    @assert size(Amax) == size(A)
    R = CartesianRange(size(A))
    imin, imax = limits(R)
    tmin = typemin(T)
    off = last(B)
    @inbounds for i in R
        vmax = tmin
        for j in CartesianRange(max(imin, i - off), min(imax, i + off))
            vmax = max(vmax, A[j])
        end
        Amax[i] = vmax
    end
    return Amax
end

function fastlocalextrema!{T,N}(Amin::AbstractArray{T,N},
                                Amax::AbstractArray{T,N},
                                A::AbstractArray{T,N},
                                B::CenteredBox{N})
    @assert size(Amin) == size(Amax) == size(A)
    R = CartesianRange(size(A))
    imin, imax = limits(R)
    tmin, tmax = limits(T)
    off = last(B)
    @inbounds for i in R
        vmin, vmax = tmax, tmin
        for j in CartesianRange(max(imin, i - off), min(imax, i + off))
            vmin = min(vmin, A[j])
            vmax = max(vmax, A[j])
        end
        Amin[i] = vmin
        Amax[i] = vmax
    end
    return Amin, Amax
end

#------------------------------------------------------------------------------
# "Fast" versions for Cartesian boxes.

function fastlocalmean!{T,N}(dst::AbstractArray{T,N},
                             A::AbstractArray{T,N},
                             B::CartesianBox{N})
    @assert size(dst) == size(A)
    R = CartesianRange(size(A))
    imin, imax = limits(R)
    kmin, kmax = limits(B)
    @inbounds for i in R
        n, s = 0, zero(T)
        for j in CartesianRange(max(imin, i - kmax), min(imax, i - kmin))
            s += A[j]
            n += 1
        end
        dst[i] = s/n
    end
    return dst
end

function fasterode!{T,N}(Amin::AbstractArray{T,N},
                         A::AbstractArray{T,N},
                         B::CartesianBox{N})
    @assert size(Amin) == size(A)
    R = CartesianRange(size(A))
    imin, imax = limits(R)
    kmin, kmax = limits(B)
    tmax = typemax(T)
    @inbounds for i in R
        vmin = tmax
        for j in CartesianRange(max(imin, i - kmax), min(imax, i - kmin))
            vmin = min(vmin, A[j])
        end
        Amin[i] = vmin
    end
    return Amin
end

function fastdilate!{T,N}(Amax::AbstractArray{T,N},
                          A::AbstractArray{T,N},
                          B::CartesianBox{N})
    @assert size(Amax) == size(A)
    R = CartesianRange(size(A))
    imin, imax = limits(R)
    kmin, kmax = limits(B)
    tmin = typemin(T)
    @inbounds for i in R
        vmax = tmin
        for j in CartesianRange(max(imin, i - kmax), min(imax, i - kmin))
            vmax = max(vmax, A[j])
        end
        Amax[i] = vmax
    end
    return Amax
end

function fastconvolve!{T<:AbstractFloat,N}(dst::AbstractArray{T,N},
                                           A::AbstractArray{T,N},
                                           B::Kernel{T,N})
    @assert size(dst) == size(A)
    R = CartesianRange(size(A))
    imin, imax = limits(R)
    kmin, kmax = limits(B)
    ker, off = coefs(B), anchor(B)
    @inbounds for i in R
        v = zero(T)
        k = i + off
        for j in CartesianRange(max(imin, i - kmax), min(imax, i - kmin))
            v += A[j]*ker[k-j]
        end
        dst[i] = v
    end
    return dst
end

function fastlocalextrema!{T,N}(Amin::AbstractArray{T,N},
                                Amax::AbstractArray{T,N},
                                A::AbstractArray{T,N},
                                B::CartesianBox{N})
    @assert size(Amin) == size(Amax) == size(A)
    R = CartesianRange(size(A))
    imin, imax = limits(R)
    kmin, kmax = limits(B)
    tmin, tmax = limits(T)
    @inbounds for i in R
        vmin, vmax = tmax, tmin
        for j in CartesianRange(max(imin, i - kmax), min(imax, i - kmin))
                vmin = min(vmin, A[j])
            vmax = max(vmax, A[j])
        end
        Amin[i] = vmin
        Amax[i] = vmax
    end
    return Amin, Amax
end

#------------------------------------------------------------------------------
# "Fast" versions for kernels of booleans.

function fastlocalmean!{T,N}(dst::AbstractArray{T,N},
                             A::AbstractArray{T,N},
                             B::Kernel{Bool,N})
    @assert size(dst) == size(A)
    R = CartesianRange(size(A))
    imin, imax = limits(R)
    kmin, kmax = limits(B)
    ker, off = coefs(B), anchor(B)
    @inbounds for i in R
        n, s = 0, zero(T)
        k = i + off
        for j in CartesianRange(max(imin, i - kmax), min(imax, i - kmin))
            if ker[k-j]
                n += 1
                s += A[j]
            end
        end
        dst[i] = s/n
    end
    return dst
end

function fasterode!{T,N}(Amin::AbstractArray{T,N},
                         A::AbstractArray{T,N},
                         B::Kernel{Bool,N})
    @assert size(Amin) == size(A)
    R = CartesianRange(size(A))
    imin, imax = limits(R)
    kmin, kmax = limits(B)
    ker, off = coefs(B), anchor(B)
    tmax = typemax(T)
    @inbounds for i in R
        vmin = tmax
        k = i + off
        for j in CartesianRange(max(imin, i - kmax), min(imax, i - kmin))
            #if ker[k-j] && A[j] < vmin
            #    vmin = A[j]
            #end
            vmin = ker[k-j] && A[j] < vmin ? A[j] : vmin
        end
        Amin[i] = vmin
    end
    return Amin
end

function fastdilate!{T,N}(Amax::AbstractArray{T,N},
                          A::AbstractArray{T,N},
                          B::Kernel{Bool,N})
    @assert size(Amax) == size(A)
    R = CartesianRange(size(A))
    imin, imax = limits(R)
    kmin, kmax = limits(B)
    ker, off = coefs(B), anchor(B)
    tmin = typemin(T)
    @inbounds for i in R
        vmax = tmin
        k = i + off
        for j in CartesianRange(max(imin, i - kmax), min(imax, i - kmin))
            #if ker[k-j] && A[j] > vmax
            #    vmax = A[j]
            #end
            vmax = ker[k-j] && A[j] > vmax ? A[j] : vmax
        end
        Amax[i] = vmax
    end
    return Amax
end

function fastlocalextrema!{T,N}(Amin::AbstractArray{T,N},
                                Amax::AbstractArray{T,N},
                                A::AbstractArray{T,N},
                                B::Kernel{Bool,N})
    @assert size(Amin) == size(Amax) == size(A)
    R = CartesianRange(size(A))
    imin, imax = limits(R)
    kmin, kmax = limits(B)
    tmin, tmax = limits(T)
    ker, off = coefs(B), anchor(B)
    @inbounds for i in R
        vmin, vmax = tmax, tmin
        k = i + off
        for j in CartesianRange(max(imin, i - kmax), min(imax, i - kmin))
            #if ker[k-j]
            #    vmin = min(vmin, A[j])
            #    vmax = max(vmax, A[j])
            #end
            vmin = ker[k-j] && A[j] < vmin ? A[j] : vmin
            vmax = ker[k-j] && A[j] > vmax ? A[j] : vmax
        end
        Amin[i] = vmin
        Amax[i] = vmax
    end
    return Amin, Amax
end

#------------------------------------------------------------------------------
# "Fast" versions for other kernels.

function fastlocalmean!{T<:AbstractFloat,N}(dst::AbstractArray{T,N},
                                            A::AbstractArray{T,N},
                                            B::Kernel{T,N})
    @assert size(dst) == size(A)
    R = CartesianRange(size(A))
    imin, imax = limits(R)
    kmin, kmax = limits(B)
    ker, off = coefs(B), anchor(B)
    @inbounds for i in R
        s1, s2 = zero(T), zero(T)
        k = i + off
        for j in CartesianRange(max(imin, i - kmax), min(imax, i - kmin))
            w = ker[k-j]
            s1 += w*A[j]
            s2 += w
        end
        dst[i] = s1/s2
    end
    return dst
end

function fasterode!{T<:AbstractFloat,N}(Amin::AbstractArray{T,N},
                                        A::AbstractArray{T,N},
                                        B::Kernel{T,N})
    @assert size(Amin) == size(A)
    R = CartesianRange(size(A))
    imin, imax = limits(R)
    kmin, kmax = limits(B)
    ker, off = coefs(B), anchor(B)
    tmax = typemax(T)
    @inbounds for i in R
        vmin = tmax
        k = i + off
        for j in CartesianRange(max(imin, i - kmax), min(imax, i - kmin))
            vmin = min(vmin, A[j] - ker[k-j])
        end
        Amin[i] = vmin
    end
    return Amin
end

function fastdilate!{T<:AbstractFloat,N}(Amax::AbstractArray{T,N},
                                         A::AbstractArray{T,N},
                                         B::Kernel{T,N})
    @assert size(Amax) == size(A)
    R = CartesianRange(size(A))
    imin, imax = limits(R)
    kmin, kmax = limits(B)
    ker, off = coefs(B), anchor(B)
    tmin = typemin(T)
    @inbounds for i in R
        vmax = tmin
        k = i + off
        for j in CartesianRange(max(imin, i - kmax), min(imax, i - kmin))
            vmax = max(vmax, A[j] + ker[k-j])
        end
        Amax[i] = vmax
    end
    return Amax
end

function fastconvolve!{T<:AbstractFloat,N}(dst::AbstractArray{T,N},
                                           A::AbstractArray{T,N},
                                           B::Kernel{T,N})
    @assert size(dst) == size(A)
    R = CartesianRange(size(A))
    imin, imax = limits(R)
    kmin, kmax = limits(B)
    ker, off = coefs(B), anchor(B)
    @inbounds for i in R
        v = zero(T)
        k = i + off
        for j in CartesianRange(max(imin, i - kmax), min(imax, i - kmin))
            v += A[j]*ker[k-j]
        end
        dst[i] = v
    end
    return dst
end

function fastlocalextrema!{T<:AbstractFloat,N}(Amin::AbstractArray{T,N},
                                               Amax::AbstractArray{T,N},
                                               A::AbstractArray{T,N},
                                               B::Kernel{T,N})
    @assert size(Amin) == size(Amax) == size(A)
    R = CartesianRange(size(A))
    imin, imax = limits(R)
    kmin, kmax = limits(B)
    tmin, tmax = limits(T)
    ker, off = coefs(B), anchor(B)
    @inbounds for i in R
        vmin, vmax = tmax, tmin
        k = i + off
        for j in CartesianRange(max(imin, i - kmax), min(imax, i - kmin))
            vmin = min(vmin, A[j] - ker[k-j])
            vmax = max(vmax, A[j] + ker[k-j])
        end
        Amin[i] = vmin
        Amax[i] = vmax
    end
    return Amin, Amax
end

#------------------------------------------------------------------------------

function checkresult(txt, tst)
    print(txt)
    if tst
        print_with_color(:green, " yes\n")
    else
        print_with_color(:red, " no\n")
    end
end

if AUTORUN
    n = 1000
    a = rand(62,81)
    cbox = CenteredBox(3,5)
    rbox = CartesianBox(cbox)
    mask = Kernel(cbox)
    kern = Kernel(eltype(a), mask)
    a0 = similar(a)
    a1 = similar(a)
    a2 = similar(a)
    a3 = similar(a)
    a4 = similar(a)
    a5 = similar(a)

    println("\nErosion on a CenteredBox:")
    erode!(a0, a, cbox)
    fasterode!(a1, a, cbox)
    checkresult(" - same result: ", samevalues(a1, a0))
    println(" - timings ($n iterations):")
    print("    driver: "); @time for i in 1:n; erode!(a0, a, cbox); end
    print("      fast: "); @time for i in 1:n; fasterode!(a1, a, cbox); end

    println("\nDilation on a CenteredBox:")
    dilate!(a0, a, cbox)
    fastdilate!(a1, a, cbox)
    checkresult(" - same result: ", samevalues(a1, a0))
    println(" - timings ($n iterations):")
    print("    driver: "); @time for i in 1:n; dilate!(a0, a, cbox); end
    print("      fast: "); @time for i in 1:n; fastdilate!(a1, a, cbox); end

    println("\nErosion and dilation on a CenteredBox:")
    erode!(a0, a, cbox)
    dilate!(a1, a, cbox)
    localextrema!(a2, a3, a, cbox)
    fastlocalextrema!(a4, a5, a, cbox)
    checkresult(" - same result: ", samevalues(a0, a2) && samevalues(a1, a3))
    checkresult(" - same result: ", samevalues(a0, a4) && samevalues(a1, a5))
    println(" - timings ($n iterations):")
    print("    driver: "); @time for i in 1:n; localextrema!(a2, a3, a, cbox); end
    print("      fast: "); @time for i in 1:n; fastlocalextrema!(a4, a5, a, cbox); end

    println("\n\nErosion on a CartesianBox:")
    erode!(a0, a, rbox)
    fasterode!(a1, a, rbox)
    checkresult(" - same result: ", samevalues(a1, a0))
    println(" - timings ($n iterations):")
    print("    driver: "); @time for i in 1:n; erode!(a0, a, rbox); end
    print("      fast: "); @time for i in 1:n; fasterode!(a1, a, rbox); end

    println("\nDilation on a CartesianBox:")
    dilate!(a0, a, rbox)
    fastdilate!(a1, a, rbox)
    checkresult(" - same result: ", samevalues(a1, a0))
    println(" - timings ($n iterations):")
    print("    driver: "); @time for i in 1:n; dilate!(a0, a, rbox); end
    print("      fast: "); @time for i in 1:n; fastdilate!(a1, a, rbox); end

    println("\nErosion and dilation on a CartesianBox:")
    erode!(a0, a, rbox)
    dilate!(a1, a, rbox)
    localextrema!(a2, a3, a, rbox)
    fastlocalextrema!(a4, a5, a, rbox)
    checkresult(" - same result: ", samevalues(a0, a2) && samevalues(a1, a3))
    checkresult(" - same result: ", samevalues(a0, a4) && samevalues(a1, a5))
    println(" - timings ($n iterations):")
    print("    driver: "); @time for i in 1:n; localextrema!(a2, a3, a, rbox); end
    print("      fast: "); @time for i in 1:n; fastlocalextrema!(a4, a5, a, rbox); end

    println("\n\nErosion on a boolean kernel:")
    erode!(a0, a, mask)
    fasterode!(a1, a, mask)
    checkresult(" - same result: ", samevalues(a1, a0))
    println(" - timings ($n iterations):")
    print("    driver: "); @time for i in 1:n; erode!(a0, a, mask); end
    print("      fast: "); @time for i in 1:n; fasterode!(a1, a, mask); end

    println("\nDilation on a boolean kernel:")
    dilate!(a0, a, mask)
    fastdilate!(a1, a, mask)
    checkresult(" - same result: ", samevalues(a1, a0))
    println(" - timings ($n iterations):")
    print("    driver: "); @time for i in 1:n; dilate!(a0, a, mask); end
    print("      fast: "); @time for i in 1:n; fastdilate!(a1, a, mask); end

    println("\nErosion and dilation on a boolean Kernel:")
    erode!(a0, a, mask)
    dilate!(a1, a, mask)
    localextrema!(a2, a3, a, mask)
    fastlocalextrema!(a4, a5, a, mask)
    checkresult(" - same result: ", samevalues(a0, a2) && samevalues(a1, a3))
    checkresult(" - same result: ", samevalues(a0, a4) && samevalues(a1, a5))
    println(" - timings ($n iterations):")
    print("    driver: "); @time for i in 1:n; localextrema!(a2, a3, a, mask); end
    print("      fast: "); @time for i in 1:n; fastlocalextrema!(a4, a5, a, mask); end

    println("\n\nErosion on a non-boolean kernel:")
    erode!(a0, a, kern)
    fasterode!(a1, a, kern)
    checkresult(" - same result: ", samevalues(a1, a0))
    println(" - timings ($n iterations):")
    print("    driver: "); @time for i in 1:n; erode!(a0, a, kern); end
    print("      fast: "); @time for i in 1:n; fasterode!(a1, a, kern); end

    println("\nDilation on a non-boolean kernel:")
    dilate!(a0, a, kern)
    fastdilate!(a1, a, kern)
    checkresult(" - same result: ", samevalues(a1, a0))
    println(" - timings ($n iterations):")
    print("    driver: "); @time for i in 1:n; dilate!(a0, a, kern); end
    print("      fast: "); @time for i in 1:n; fastdilate!(a1, a, kern); end

    println("\nErosion and dilation on a non-boolean Kernel:")
    erode!(a0, a, kern)
    dilate!(a1, a, kern)
    localextrema!(a2, a3, a, kern)
    fastlocalextrema!(a4, a5, a, kern)
    checkresult(" - same result: ", samevalues(a0, a2) && samevalues(a1, a3))
    checkresult(" - same result: ", samevalues(a0, a4) && samevalues(a1, a5))
    println(" - timings ($n iterations):")
    print("    driver: "); @time for i in 1:n; localextrema!(a2, a3, a, kern); end
    print("      fast: "); @time for i in 1:n; fastlocalextrema!(a4, a5, a, kern); end

end

end # module
