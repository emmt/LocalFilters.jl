isdefined(Main, :LocalFilters) || include("../src/LocalFilters.jl")

module LocalFiltersTests

using Compat
using LocalFilters, Compat.Test
using LocalFilters: Neighborhood, RectangularBox, Kernel,
    axes, initialindex, finalindex, limits, cartesianregion, ball, coefs,
    strictfloor, USE_CARTESIAN_RANGE, _range

replicate(a, n::Integer) = ntuple(i->a, n)
compare(a::AbstractArray, b::AbstractArray) = maximum(abs(a - b))
samevalues(a::AbstractArray, b::AbstractArray) = minimum(a .== b)
identical(a::RectangularBox{N}, b::RectangularBox{N}) where {N} =
    (initialindex(a) === initialindex(b) &&
     finalindex(a) === finalindex(b))
identical(a::Kernel{T,N}, b::Kernel{T,N}) where {T,N} =
    (initialindex(a) === initialindex(b) &&
     finalindex(a) === finalindex(b) &&
     samevalues(coefs(a), coefs(b)))
identical(a::Neighborhood, b::Neighborhood) = false

const trivialerode = erode
const trivialdilate = dilate
const triviallocalextrema = localextrema

function reversealldims(A::Array{T,N}) where {T,N}
    B = Array{T,N}(undef, size(A))
    len = length(A)
    off = len + 1
    @inbounds for i in 1:len
        B[off - i] = A[i]
    end
    return B
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
            I1, I2 = limits(box)
            A = rand(dims...)
            ker = Kernel(A)

            # Test limits(), initialindex() and finalindex().
            @test initialindex(CartesianIndices(rngs)) === I1
            @test finalindex(CartesianIndices(rngs)) === I2
            @test limits(CartesianIndices(rngs)) === (I1, I2)
            @static if USE_CARTESIAN_RANGE
                @test initialindex(CartesianRange(rngs)) === I1
                @test finalindex(CartesianRange(rngs)) === I2
                @test limits(CartesianRange(rngs)) === (I1, I2)
            end
            @test initialindex(box) === I1
            @test finalindex(box) === I2
            @test limits(box) === (I1, I2)
            @test initialindex(A) === one(CartesianIndex{N})
            @test finalindex(A) === CartesianIndex(size(A))
            @test limits(A) === (one(CartesianIndex{N}),
                                 CartesianIndex(size(A)))

            # Test cartesianregion().
            @static if USE_CARTESIAN_RANGE
                region = CartesianRange(rngs)
                @test cartesianregion(I1,I2) === region
                @test cartesianregion(CartesianRange(rngs)) === region
                @test cartesianregion(CartesianIndices(rngs)) === region
                @test cartesianregion(box) === region
                @test cartesianregion(A) === CartesianRange(size(A))
            else
                region = CartesianIndices(rngs)
                @test cartesianregion(I1,I2) === region
                @test cartesianregion(CartesianIndices(rngs)) === region
                @test cartesianregion(box) === region
                @test cartesianregion(A) === CartesianIndices(A)
            end

            # Neighborhood constructors.
            @test Neighborhood(box) === box
            @test Neighborhood(A) === ker
            @test Neighborhood(dims...) === box
            @test Neighborhood(rngs) === box
            @test Neighborhood(rngs...) === box
            @test Neighborhood(CartesianIndices(rngs)) === box
            @static if USE_CARTESIAN_RANGE
                @test Neighborhood(CartesianRange(rngs)) === box
                @test Neighborhood(CartesianRange(I1, I2)) === box
            end
            @test convert(Neighborhood, rngs) === box
            @test convert(Neighborhood, dims) === box

            # RectangularBox constructors.
            @test RectangularBox(box) === box
            @test RectangularBox(dims) === box
            @test RectangularBox(dims...) === box
            @test RectangularBox(rngs) === box
            @test RectangularBox(rngs...) === box
            @static if USE_CARTESIAN_RANGE
                @test CartesianRange(box) === CartesianRange(rngs)
                @test CartesianRange(box) === CartesianRange(I1, I2)
            end
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
            @static if USE_CARTESIAN_RANGE
                # Conversion RectangularBox <-> CartesianRange.
                @test RectangularBox(CartesianRange(rngs)) === box
                @test convert(RectangularBox, CartesianRange(rngs)) === box
                @test convert(RectangularBox{N}, CartesianRange(rngs)) === box
                @test CartesianRange(box) === CartesianRange(rngs)
                @test convert(CartesianRange, box) === CartesianRange(rngs)
                @test convert(CartesianRange{CartesianIndex{N}}, box) ===
                    CartesianRange(rngs)
            end

            # Kernel constructors.
            @test Kernel(A, initialindex(ker)) === ker
            @test Kernel(A, rngs) === ker
            @test Kernel(A, rngs...) === ker
            @test Kernel(A, CartesianIndices(ker)) === ker
            @static if USE_CARTESIAN_RANGE
                @test Kernel(A, CartesianRange(ker)) === ker
            end
            off = initialindex(A) - initialindex(ker)
            @test identical(Kernel(i -> f1(A[off + i]), CartesianIndices(ker)),
                            Kernel(map(f1, A)))
            @test identical(Kernel(i -> f2(A[off + i]), CartesianIndices(ker)),
                            Kernel(map(f2, A)))

            # Conversion Neighborhood <-> CartesianIndices.
            @test Neighborhood(CartesianIndices(rngs)) === box
            @test convert(Neighborhood, CartesianIndices(rngs)) === box
            @test convert(Neighborhood{N}, CartesianIndices(rngs)) === box
            @test CartesianIndices(box) === CartesianIndices(rngs)
            @test convert(CartesianIndices, box) === CartesianIndices(rngs)
            @test convert(CartesianIndices{N}, box) === CartesianIndices(rngs)
            @static if USE_CARTESIAN_RANGE
                # Conversion Neighborhood <-> CartesianRange
                @test Neighborhood(CartesianRange(rngs)) === box
                @test convert(Neighborhood, CartesianRange(rngs)) === box
                @test convert(Neighborhood{N}, CartesianRange(rngs)) === box
                @test CartesianRange(box) === CartesianRange(rngs)
                @test convert(CartesianRange, box) === CartesianRange(rngs)
                @test convert(CartesianRange{CartesianIndex{N}}, box) ===
                    CartesianRange(rngs)
            end

            # Other basic methods.
            @test length(box) === length(CartesianIndices(rngs))
            @test size(box) === size(CartesianIndices(rngs))
            @test size(box) === ntuple(d -> size(box, d), N)
            @test axes(box) === ntuple(d -> axes(box, d), N)
            @test length(ker) === length(A)
            @test size(ker) === size(A)
            @test size(ker) === ntuple(d -> size(ker, d), N)
            @test axes(ker) === ntuple(d -> axes(ker, d), N)

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
        @testset "$(rank)D test with boxes" begin
            a = rand(arrdims...)
            box = RectangularBox(boxdims)
            mask = Kernel(box)
            kern = Kernel(eltype(a), mask)
            @testset "erode" begin
                result = trivialerode(a, box)
                @test samevalues(erode(a, box), result)
                @test samevalues(erode(a, mask), result)
                @test samevalues(erode(a, kern), result)
            end
            @testset "dilate" begin
                result = trivialdilate(a, box)
                @test samevalues(dilate(a, box), result)
                @test samevalues(dilate(a, mask), result)
                @test samevalues(dilate(a, kern), result)
            end
            @testset "localextrema" begin
                e0, d0 = trivialerode(a, box), trivialdilate(a, box)
                e1, d1 = triviallocalextrema(a, box)
                @test samevalues(e0, e1) && samevalues(d0, d1)
                e1, d1 = localextrema(a, box)
                @test samevalues(e0, e1) && samevalues(d0, d1)
                e1, d1 = localextrema(a, mask)
                @test samevalues(e0, e1) && samevalues(d0, d1)
                e1, d1 = localextrema(a, kern)
                @test samevalues(e0, e1) && samevalues(d0, d1)
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
        kern = Kernel(eltype(a), mask)
        @testset "erode" begin
            result = erode(a, mask)
            @test samevalues(erode(a, kern), result)
        end
        @testset "dilate" begin
            result = dilate(a, mask)
            @test samevalues(dilate(a, kern), result)
        end
    end

end

end # module
