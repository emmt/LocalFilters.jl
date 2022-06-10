"""

This module is to tests the strong penalty in execution time and allocations of
closures in `localfilters!` driver.

"""
module ClosureBug

using LocalFilters, BenchmarkTools

using LocalFilters: Kernel, check_indices, type_of_sum, store!

import LocalFilters: localmean!, convolve!

#------------------------------------------------------------------------------
# localmean!

localmean!(::Val{:master}, args...; kwds...) =
    localmean!(args...; kwds...)

function localmean!(alg::Val, dst::AbstractArray{<:Any,N},
                    A::AbstractArray{<:Any,N}, B=3) where {N}
    localmean!(alg, dst, A, Neighborhood{N}(B))
end

function localmean!(::Val{:anonymous},
                    dst::AbstractArray{<:Any,N},
                    A::AbstractArray{<:Any,N},
                    B::RectangularBox{N}) where {N}
    check_indices(dst, A)
    T = type_of_sum(eltype(A))
    localfilter!(dst, A, B,
                 (a)     -> (zero(T), 0),
                 (v,a,b) -> (v[1] + a, v[2] + 1),
                 (d,i,v) -> store!(d, i, v[1]/v[2]))
end

function localmean!(::Val{:anonymous},
                    dst::AbstractArray{<:Any,N},
                    A::AbstractArray{<:Any,N},
                    B::Kernel{Bool,N}) where {N}
    check_indices(dst, A)
    T = type_of_sum(eltype(A))
    localfilter!(dst, A, B,
                 (a)     -> (zero(T), 0),
                 (v,a,b) -> b ? (v[1] + a, v[2] + 1) : v,
                 (d,i,v) -> store!(d, i, v[1]/v[2]))
end

function localmean!(::Val{:anonymous},
                    dst::AbstractArray{<:Any,N},
                    A::AbstractArray{<:Any,N},
                    B::Kernel{<:Any,N}) where {N}
    check_indices(dst, A)
    T = type_of_sum(promote_type(eltype(A), eltype(B)))
    localfilter!(dst, A, B,
                 (a)     -> (zero(T), zero(T)),
                 (v,a,b) -> (v[1] + a*b, v[2] + b),
                 (d,i,v) -> store!(d, i, v[1]/v[2]))
end

#------------------------------------------------------------------------------
# convolve!

convolve!!(::Val{:master}, args...; kwds...) =
    convolve!(args...; kwds...)

function convolve!(alg::Val, dst::AbstractArray{<:Any,N},
                   A::AbstractArray{<:Any,N}, B=3) where {N}
    convolve!(alg, dst, A, Neighborhood{N}(B))
end

function convolve!(::Val{:anonymous},
                   dst::AbstractArray{<:Any,N},
                   A::AbstractArray{<:Any,N},
                   B::RectangularBox{N}) where {N}
    check_indices(dst, A)
    T = type_of_sum(eltype(A))
    localfilter!(dst, A, B,
                 (a)     -> zero(T),
                 (v,a,b) -> v + a,
                 store!)
end

function convolve!(::Val{:anonymous},
                   dst::AbstractArray{<:Any,N},
                   A::AbstractArray{<:Any,N},
                   B::Kernel{Bool,N}) where {N}
    check_indices(dst, A)
    T = type_of_sum(eltype(A))
    localfilter!(dst, A, B,
                 (a)     -> zero(T),
                 (v,a,b) -> b ? v + a : v,
                 store!)
end

function convolve!(::Val{:anonymous},
                   dst::AbstractArray{<:Any,N},
                   A::AbstractArray{<:Any,N},
                   B::Kernel{<:Any,N}) where {N}
    check_indices(dst, A)
    T = type_of_sum(promote_type(eltype(A), eltype(B)))
    localfilter!(dst, A, B,
                 (a)     -> zero(T),
                 (v,a,b) -> v + a*b,
                 store!)
end

#------------------------------------------------------------------------------

T = Float32
dims = (400, 300)
A = rand(T, dims);
dst_1 = similar(A);
dst_2 = similar(A);

box = RectangularBox(5,3);
ker1 = Kernel(ones(Bool, (5,3)));
ker2 = Kernel(ones(T, (5,3)));

# Create a weighted window for the bilateral filter to avoid allocations.
σr = 1.2;
σs = 2.5;
W = LocalFilters.Kernel(LocalFilters.BilateralFilter.GaussianWindow(σs),
                        LocalFilters.RectangularBox((-7:7,-7:7)));

println("\nTesting `localmean!` with Cartesian box")
print("  old algorithm:")
@btime localmean!($(Val(:anonymous)), $dst_2,$A,$box);
print("  new algorithm:")
@btime localmean!($dst_1,$A,$box);
println("  max. abs. diff: ", maximum(map(abs, extrema(dst_2 - dst_1))))

println("\nTesting `localmean!` with boolean kernel")
print("  old algorithm:")
@btime localmean!($(Val(:anonymous)), $dst_2,$A,$ker1);
print("  new algorithm:")
@btime localmean!($dst_1,$A,$ker1);
println("  max. abs. diff: ", maximum(map(abs, extrema(dst_2 - dst_1))))

println("\nTesting `localmean!` with floating-point kernel")
print("  old algorithm:")
@btime localmean!($(Val(:anonymous)), $dst_2,$A,$ker2);
print("  new algorithm:")
@btime localmean!($dst_1,$A,$ker2);
println("  max. abs. diff: ", maximum(map(abs, extrema(dst_2 - dst_1))))

println()

println("\nTesting `convolve!` with Cartesian box")
print("  old algorithm:")
@btime convolve!($(Val(:anonymous)), $dst_2,$A,$box);
print("  new algorithm:")
@btime convolve!($dst_1,$A,$box);
println("  max. abs. diff: ", maximum(map(abs, extrema(dst_2 - dst_1))))

println("\nTesting `convolve!` with boolean kernel")
print("  old algorithm:")
@btime convolve!($(Val(:anonymous)), $dst_2,$A,$ker1);
print("  new algorithm:")
@btime convolve!($dst_1,$A,$ker1);
println("  max. abs. diff: ", maximum(map(abs, extrema(dst_2 - dst_1))))

println("\nTesting `convolve!` with floating-point kernel")
print("  old algorithm:")
@btime convolve!($(Val(:anonymous)), $dst_2,$A,$ker2);
print("  new algorithm:")
@btime convolve!($dst_1,$A,$ker2);
println("  max. abs. diff: ", maximum(map(abs, extrema(dst_2 - dst_1))))

println()

println("\nTesting `bilateralfilter!` with weights")
print("  current algorithm:     ")
@btime bilateralfilter!($dst_1,$A,$σr,$W);

println()

println("\nTesting `erode!` with Cartesian box")
print("  current algorithm:     ")
@btime erode!($dst_1,$A,$box);

println("\nTesting `erode!` with boolean kernel")
print("  current algorithm:     ")
@btime erode!($dst_1,$A,$ker1);

println()

println("\nTesting `dilate!` with Cartesian box")
print("  current algorithm:     ")
@btime dilate!($dst_1,$A,$box);

println("\nTesting `dilate!` with boolean kernel")
print("  current algorithm:     ")
@btime dilate!($dst_1,$A,$ker1);

end # module
