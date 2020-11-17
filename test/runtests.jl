isdefined(Main, :LocalFilters) || include("../src/LocalFilters.jl")
isdefined(Main, :NaiveLocalFilters) || include("NaiveLocalFilters.jl")

module LocalFiltersTests

using Compat
using Test
using LocalFilters
using LocalFilters: Neighborhood, RectangularBox, Kernel,
    axes, initialindex, finalindex, limits, cartesianregion, ball, coefs,
    ismmbox, strictfloor, _range

struct Empty{N} <: Neighborhood{N} end

# Selector for reference methods.
const REF = Val(:Base)

const ATOL = 0.0
const GTOL = 4*eps(Float64)

replicate(a, n::Integer) = ntuple(i->a, n)
compare(a::AbstractArray, b::AbstractArray) = maximum(abs(a - b))
samevalues(a::AbstractArray, b::AbstractArray) = minimum(a .== b)
identical(a::RectangularBox{N}, b::RectangularBox{N}) where {N} =
    (initialindex(a) === initialindex(b) &&
     finalindex(a) === finalindex(b))
identical(a::Kernel{Ta,N}, b::Kernel{Tb,N}) where {Ta,Tb,N} =
    nearlysame(a, b; atol=0, gtol=0)
identical(a::Neighborhood, b::Neighborhood) = false

function nearlysame(a::Kernel{Ta,N}, b::Kernel{Tb,N};
                    atol=ATOL, gtol=GTOL) where {Ta,Tb,N}
    initialindex(a) == initialindex(b) || return false
    finalindex(a) == finalindex(b) || return false
    if atol == 0 && gtol == 0
        for i in cartesianregion(a)
            a[i] == b[i] || return false
        end
    else
        for i in cartesianregion(a)
            ai, bi = a[i], b[i]
            abs(ai - bi) ≤ atol + gtol*max(abs(ai), abs(bi)) || return false
        end
    end
    return true
end

function similarvalues(A::AbstractArray{Ta,N}, B::AbstractArray{Tb,N};
                       atol=ATOL, gtol=GTOL) where {Ta,Tb,N}
    @assert axes(A) == axes(B)
    T = float(promote_type(Ta, Tb))
    local sd2::T = 0
    local sa2::T = 0
    local sb2::T = 0
    @inbounds for i in eachindex(A, B)
        a = T(A[i])
        b = T(B[i])
        sa2 += a*a
        sb2 += b*b
        sd2 += (a - b)^2
    end
    return sqrt(sd2) ≤ atol + gtol*sqrt(max(sa2, sb2))
end

function reversealldims(A::Array{T,N}) where {T,N}
    B = Array{T,N}(undef, size(A))
    len = length(A)
    off = len + 1
    @inbounds for i in 1:len
        B[off - i] = A[i]
    end
    return B
end

function checkindexing!(ker::Kernel{T,1}) where {T}
    Imin, Imax = limits(ker)
    i1min, i1max = Imin[1], Imax[1]
    tmp = zero(T)
    for i1 in i1min:i1max
        I = CartesianIndex(i1)
        val = ker[I]
        ker[i1] == val || return false
        ker[i1] = tmp
        ker[i1] == ker[I] == tmp || return false
        ker[I] = val
    end
    return true
end

function checkindexing!(ker::Kernel{T,2}) where {T}
    Imin, Imax = limits(ker)
    i1min, i1max = Imin[1], Imax[1]
    i2min, i2max = Imin[2], Imax[2]
    tmp = zero(T)
    for i2 in i2min:i2max,
        i1 in i1min:i1max
        I = CartesianIndex(i1,i2)
        val = ker[I]
        ker[i1,i2] == val || return false
        ker[i1,i2] = tmp
        ker[i1,i2] == ker[I] == tmp || return false
        ker[I] = val
    end
    return true
end

function checkindexing!(ker::Kernel{T,3}) where {T}
    Imin, Imax = limits(ker)
    i1min, i1max = Imin[1], Imax[1]
    i2min, i2max = Imin[2], Imax[2]
    i3min, i3max = Imin[3], Imax[3]
    tmp = zero(T)
    for i3 in i3min:i3max,
        i2 in i2min:i2max,
        i1 in i1min:i1max
        I = CartesianIndex(i1,i2,i3)
        val = ker[I]
        ker[i1,i2,i3] == val || return false
        ker[i1,i2,i3] = tmp
        ker[i1,i2,i3] == ker[I] == tmp || return false
        ker[I] = val
    end
    return true
end

f1(x) = 1 + x*x
f2(x) = x > 0.5

@testset "LocalFilters" begin

    @testset "Miscellaneous" begin
        for T in (Int, Float32)
            @test limits(T) === (typemin(T), typemax(T))
        end
        for dim in (11, 12)
            @test _range(Int8(dim)) === _range(dim)
        end
        @test_throws ArgumentError _range(0)
        @test strictfloor(Int, 5*(1 + eps(Float64))) == 5
        @test strictfloor(Int, 5.0) == 4
    end

    @testset "Neighborhoods" begin
        for (dims, rngs) in (((3,), (-1:1,)),
                             ((3, 4, 5), (-1:1, -2:1, -2:2)))
            N = length(dims)
            box = Neighborhood(dims)
            fse = strel(Bool, box) # flat structuring element
            Imin, Imax = limits(box)
            A = rand(dims...)
            msk = rand(Bool, dims)
            ker = Kernel(A)

            # Test limits(), initialindex() and finalindex().
            @test initialindex(CartesianIndices(rngs)) === Imin
            @test finalindex(CartesianIndices(rngs)) === Imax
            @test limits(CartesianIndices(rngs)) === (Imin, Imax)
            @test initialindex(box) === Imin
            @test finalindex(box) === Imax
            @test limits(box) === (Imin, Imax)
            @test initialindex(A) === oneunit(CartesianIndex{N})
            @test finalindex(A) === CartesianIndex(size(A))
            @test limits(A) === (oneunit(CartesianIndex{N}),
                                 CartesianIndex(size(A)))

            # Test cartesianregion().
            region = CartesianIndices(rngs)
            @test cartesianregion(Imin,Imax) === region
            @test cartesianregion(CartesianIndices(rngs)) === region
            @test cartesianregion(box) === region
            @test cartesianregion(A) === CartesianIndices(A)

            # Neighborhood constructors.
            @test Neighborhood(box) === box
            @test Neighborhood(ker) === ker
            @test Neighborhood(A) === ker
            @test Neighborhood(dims...) === box
            @test Neighborhood(rngs) === box
            @test Neighborhood(rngs...) === box
            @test Neighborhood(CartesianIndices(rngs)) === box
            @test convert(Neighborhood, box) === box
            @test convert(Neighborhood, ker) === ker
            @test convert(Neighborhood, rngs) === box
            @test convert(Neighborhood, dims) === box

            # RectangularBox constructors.
            @test RectangularBox(box) === box
            @test RectangularBox(dims) === box
            @test RectangularBox(dims...) === box
            @test RectangularBox(rngs) === box
            @test RectangularBox(rngs...) === box
            @test convert(RectangularBox, box) === box
            @test convert(RectangularBox, rngs) === box
            @test convert(RectangularBox, dims) === box
            dim = dims[end]
            rng = rngs[end]
            @test RectangularBox{N}(dim) === RectangularBox(ntuple(d -> dim, N))
            @test RectangularBox{N}(rng) === RectangularBox(ntuple(d -> rng, N))

            # Conversions RectangularBox <-> CartesianIndices.
            @test CartesianIndices(box) === CartesianIndices(rngs)
            @test RectangularBox(CartesianIndices(rngs)) === box
            @test convert(RectangularBox, CartesianIndices(rngs)) === box
            @test convert(RectangularBox{N}, CartesianIndices(rngs)) === box
            @test CartesianIndices(box) === CartesianIndices(rngs)
            @test convert(CartesianIndices, box) === CartesianIndices(rngs)
            @test convert(CartesianIndices{N}, box) === CartesianIndices(rngs)

            # Kernel constructors.
            @test Kernel(ker) === ker
            @test identical(Kernel(box), Kernel(ones(Bool, dims)))
            @test identical(Kernel{Bool}(box), Kernel(ones(Bool, dims)))
            @test Kernel(A, initialindex(ker)) === ker
            @test Kernel(A, rngs) === ker
            @test Kernel(A, rngs...) === ker
            @test Kernel(A, CartesianIndices(ker)) === ker
            off = initialindex(A) - initialindex(ker)
            @test identical(Kernel{eltype(A)}(i -> f1(A[off + i]),
                                              CartesianIndices(ker)),
                            Kernel(map(f1, A)))
            @test identical(Kernel(i -> f1(A[off + i]),
                                   CartesianIndices(ker)), Kernel(map(f1, A)))
            @test identical(Kernel{Bool}(i -> f2(A[off + i]),
                                         CartesianIndices(ker)),
                            Kernel(map(f2, A)))
            @test identical(Kernel(i -> f2(A[off + i]),
                                   CartesianIndices(ker)), Kernel(map(f2, A)))
            @test identical(convert(Kernel, box), fse)
            @test identical(Kernel((0,-Inf), msk), strel(Float64, Kernel(msk)))
            @test identical(strel(Bool, box), Kernel(box))
            @test convert(Kernel, fse) === fse
            @test convert(Kernel, ker) === ker

            @test nearlysame(Float32(ker), ker; atol=0, gtol=eps(Float32))
            @test nearlysame(Float32(ker), Float64(ker);
                             atol=0, gtol=eps(Float32))
            @test checkindexing!(ker)

            # Conversion Neighborhood <-> CartesianIndices.
            @test Neighborhood(CartesianIndices(rngs)) === box
            @test convert(Neighborhood, CartesianIndices(rngs)) === box
            @test convert(Neighborhood{N}, CartesianIndices(rngs)) === box
            @test CartesianIndices(box) === CartesianIndices(rngs)
            @test convert(CartesianIndices, box) === CartesianIndices(rngs)
            @test convert(CartesianIndices{N}, box) === CartesianIndices(rngs)

            # Other basic methods.
            @test length(box) === length(CartesianIndices(rngs))
            @test size(box) === size(CartesianIndices(rngs))
            @test size(box) === ntuple(d -> size(box, d), N)
            @test axes(box) === ntuple(d -> axes(box, d), N)
            @test length(ker) === length(A)
            @test size(ker) === size(A)
            @test size(ker) === ntuple(d -> size(ker, d), N)
            @test axes(ker) === ntuple(d -> axes(ker, d), N)
            @test ismmbox(fse) == true
            @test ismmbox(box) == true
            boolker = Neighborhood(ones(Bool,dims))
            @test ismmbox(boolker) == true
            boolker[zero(CartesianIndex{N})] = false
            @test ismmbox(boolker) == false
            @test ismmbox(Empty{N}()) == false

            # Test reverse().
            revbox = reverse(box)
            revker = reverse(ker)
            @test initialindex(revbox) === -finalindex(box)
            @test finalindex(revbox) === -initialindex(box)
            @test initialindex(revker) === -finalindex(ker)
            @test finalindex(revker) === -initialindex(ker)
            @test samevalues(coefs(revker), reversealldims(coefs(ker)))
        end
    end

    for (arrdims, boxdims) in (((341,),    (3,)),
                               ((62,81),   (5,3)),
                               ((23,28,27),(3,7,5)))
        rank = length(arrdims)
        @testset "$(rank)D test" begin
            a = rand(arrdims...)
            box = RectangularBox(boxdims)
            mask = Kernel(box)
            fse = strel(eltype(a), mask) # flat structuring element
            kern = Kernel((1.0, 0.0), mask) # kernel for convolution
            @testset "erode" begin
                result = erode(REF, a, box)
                @test samevalues(erode(a, box), result)
                @test samevalues(erode(a, mask), result)
                @test samevalues(erode(a, fse), result)
                @test samevalues(erode!(copy(a), box), result)
            end
            @testset "dilate" begin
                result = dilate(REF, a, box)
                @test samevalues(dilate(a, box), result)
                @test samevalues(dilate(a, mask), result)
                @test samevalues(dilate(a, fse), result)
                @test samevalues(dilate!(copy(a), box), result)
            end
            @testset "closing" begin
                result = closing(REF, a, box)
                @test samevalues(closing(a, box), result)
                @test samevalues(closing(a, mask), result)
                @test samevalues(closing(a, fse), result)
            end
            @testset "opening" begin
                result = opening(REF, a, box)
                @test samevalues(opening(a, box), result)
                @test samevalues(opening(a, mask), result)
                @test samevalues(opening(a, fse), result)
            end
            @testset "bottom-hat" begin
                result = bottom_hat(REF, a, box)
                @test samevalues(bottom_hat(a, box), result)
                @test samevalues(bottom_hat(a, mask), result)
                @test samevalues(bottom_hat(a, fse), result)
                result = bottom_hat(REF, a, box, 3)
                @test samevalues(bottom_hat(a, box, 3), result)
                @test samevalues(bottom_hat(a, mask, 3), result)
                @test samevalues(bottom_hat(a, fse, 3), result)
            end
            @testset "top-hat" begin
                result = top_hat(REF, a, box)
                @test samevalues(top_hat(a, box), result)
                @test samevalues(top_hat(a, mask), result)
                @test samevalues(top_hat(a, fse), result)
                result = top_hat(REF, a, box, 3)
                @test samevalues(top_hat(a, box,  3), result)
                @test samevalues(top_hat(a, mask, 3), result)
                @test samevalues(top_hat(a, fse, 3), result)
            end
            @testset "localextrema" begin
                e0, d0 = erode(REF, a, box), dilate(REF, a, box)
                e1, d1 = localextrema(REF, a, box)
                @test samevalues(e0, e1) && samevalues(d0, d1)
                e1, d1 = localextrema(a, box)
                @test samevalues(e0, e1) && samevalues(d0, d1)
                e1, d1 = localextrema!(similar(a), similar(a), a, box)
                @test samevalues(e0, e1) && samevalues(d0, d1)
                e1, d1 = localextrema(a, mask)
                @test samevalues(e0, e1) && samevalues(d0, d1)
                e1, d1 = localextrema(a, fse)
                @test samevalues(e0, e1) && samevalues(d0, d1)
            end
            @testset "localmean" begin
                result = localmean(REF, a, box)
                @test similarvalues(localmean(a, box), result)
                @test similarvalues(localmean(a, mask), result)
                @test similarvalues(localmean(a, kern), result)
            end
            @testset "convolve" begin
                result = convolve(REF, a, kern)
                @test similarvalues(convolve(a, box), result)
                @test similarvalues(convolve(a, mask), result)
                @test similarvalues(convolve(a, kern), result)
            end
        end
    end

    ball3x3 = Bool[1 1 1;
                   1 1 1;
                   1 1 1];
    ball5x5 = Bool[0 1 1 1 0;
                   1 1 1 1 1;
                   1 1 1 1 1;
                   1 1 1 1 1;
                   0 1 1 1 0];
    ball7x7 = Bool[0 0 1 1 1 0 0;
                   0 1 1 1 1 1 0;
                   1 1 1 1 1 1 1;
                   1 1 1 1 1 1 1;
                   1 1 1 1 1 1 1;
                   0 1 1 1 1 1 0;
                   0 0 1 1 1 0 0];
    @test samevalues(ball3x3, ball(2, 1))
    @test samevalues(ball5x5, ball(2, 2))
    @test samevalues(ball7x7, ball(2, 3))

    @testset "2D test with a 5x5 ball" begin
        a = rand(162,181)
        mask = Kernel(ball5x5)
        fse = strel(eltype(a), mask)
        @testset "erode" begin
            result = erode(REF, a, mask)
            @test samevalues(erode(a, mask), result)
            @test samevalues(erode(a, fse), result)
        end
        @testset "dilate" begin
            result = dilate(REF, a, mask)
            @test samevalues(dilate(a, mask), result)
            @test samevalues(dilate(a, fse), result)
        end
    end

    # Test van Herk / Gil & Werman algorithm.
    @testset "van Herk / Gil & Werman algorithm" begin
        # Test erode and dilate.
        for dims in ((45,), (30, 21), (7, 8, 9)),
            T in (Float32, Int32)
            N = length(dims)
            A = rand(T, dims)
            B = Array{eltype(A),ndims(A)}(undef, size(A))
            C = Array{eltype(A),ndims(A)}(undef, size(A))
            box = RectangularBox{N}(5)
            rng = -2:2
            result = erode(REF, A, box)
            @test samevalues(erode(A, :, rng), result)
            @test samevalues(erode!(B, A, :, rng), result)
            @test samevalues(erode!(copyto!(B, A), :, rng), result)
            @test samevalues(localfilter!(B, A, 1:N, min, rng), result)
            if N > 1
                # Reverse order of dimensions.
                @test samevalues(localfilter!(B, A, N:-1:1, min, rng), result)
            end
            result = dilate(REF, A, box)
            @test samevalues(dilate(A, :, rng), result)
            @test samevalues(dilate!(B, A, :, rng), result)
            @test samevalues(dilate!(copyto!(B, A), :, rng), result)
            @test samevalues(localfilter!(B, A, 1:N, max, rng), result)
            if N > 1
                # Reverse order of dimensions.
                @test samevalues(localfilter!(B, A, N:-1:1, max, rng), result)
            end
        end

        # Test shifts.
        A = rand(30)
        B = Array{eltype(A),ndims(A)}(undef, size(A))
        n = length(A)
        for k in (2, 0, -3)
            for i in 1:n
                j = clamp(i - k, 1, n)
                B[i] = A[j]
            end
            @test samevalues(B, localfilter(A,1,min,k:k))
            C = copy(A)
            @test samevalues(B, localfilter!(C,1,min,k:k))
        end
        A = rand(12,13)
        B = Array{eltype(A),ndims(A)}(undef, size(A))
        C = Array{eltype(A),ndims(A)}(undef, size(A))
        n1, n2 = size(A)
        for k1 in (2, 0, -3),
            k2 in (1, 0, -2)
            for i2 in 1:n2
                j2 = clamp(i2 - k2, 1, n2)
                for i1 in 1:n1
                    j1 = clamp(i1 - k1, 1, n1)
                    B[i1,i2] = A[j1,j2]
                end
            end
            @test samevalues(B, localfilter(A,:,min,(k1:k1,k2:k2)))
            @test samevalues(B, localfilter(A,[1,2],min,(k1:k1,k2:k2)))
            @test samevalues(B, localfilter(A,(2,1),min,(k2:k2,k1:k1)))
            @test samevalues(B, localfilter!(copyto!(C,A),:,min,(k1:k1,k2:k2)))
            @test samevalues(B, localfilter!(copyto!(C,A),(1,2),min,(k1:k1,k2:k2)))
            @test samevalues(B, localfilter!(copyto!(C,A),[2,1],min,(k2:k2,k1:k1)))
        end
    end

    # Bilateral filter.
    @testset "Bilateral filter" begin
        A = randn(Float64, 128, 200)
        box = RectangularBox{2}(5)
        B1 = bilateralfilter(A, 4, 3, box)
        B2 = bilateralfilter(Float32, A, 4, 3, 5)
        B3 = bilateralfilter!(similar(Array{Float32}, axes(A)), A, 4, 3, box)
        @test similarvalues(B1, B2; atol=0, gtol=8*eps(Float32))
        @test similarvalues(B2, B3; atol=0, gtol=4*eps(Float32))
    end
end

end # module
