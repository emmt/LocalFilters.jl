isdefined(Base, :LocalFilters) || include("../src/LocalFilters.jl")

module LocalFiltersTests

using Compat
using LocalFilters, Compat.Test
using LocalFilters: Neighborhood, RectangularBox, Kernel,
    axes, initialindex, finalindex, limits, cartesianregion,
    USE_CARTESIAN_RANGE

replicate(a, n::Integer) = ntuple(i->a, n)
compare(a, b) = maximum(abs(a - b))
samevalues(a, b) = minimum(a .== b)

const trivialerode = erode
const trivialdilate = dilate
const triviallocalextrema = localextrema

@testset "LocalFilters" begin

    @testset "Miscellaneous" begin
        for T in (Int, Float32)
            @test limits(T) === (typemin(T), typemax(T))
        end
    end

    @testset "Neighborhoods" begin
        for (B1, B2) in (((3,), (-1:1,)),
                         ((3, 4, 5), (-1:1, -2:1, -2:2)))
            N = length(B1)
            box = Neighborhood(B1)
            I1, I2 = limits(box)
            inds = map((i,j) -> i:j, I1.I, I2.I)
            A = Array{Int}(undef, size(box))

            # Test limits(), initialindex() and finalindex().
            @test initialindex(CartesianIndices(inds)) === I1
            @test finalindex(CartesianIndices(inds)) === I2
            @test limits(CartesianIndices(inds)) === (I1, I2)
            @static if USE_CARTESIAN_RANGE
                @test initialindex(CartesianRange(inds)) === I1
                @test finalindex(CartesianRange(inds)) === I2
                @test limits(CartesianRange(inds)) === (I1, I2)
            end
            @test initialindex(box) === I1
            @test finalindex(box) === I2
            @test limits(box) === (I1, I2)

            # Test cartesianregion().
            @static if USE_CARTESIAN_RANGE
                region = CartesianRange(inds)
                @test cartesianregion(I1,I2) === region
                @test cartesianregion(CartesianRange(inds)) === region
                @test cartesianregion(CartesianIndices(inds)) === region
                @test cartesianregion(box) === region
                @test cartesianregion(A) === CartesianRange(size(A))
            else
                region = CartesianIndices(inds)
                @test cartesianregion(I1,I2) === region
                @test cartesianregion(CartesianIndices(inds)) === region
                @test cartesianregion(box) === region
                @test cartesianregion(A) === CartesianIndices(A)
            end

            # Neighborhood constructors.
            @test Neighborhood(B1...) === box
            @test Neighborhood(B2) === box
            @test Neighborhood(B2...) === box
            @test Neighborhood(CartesianIndices(inds)) === box
            @static if USE_CARTESIAN_RANGE
                @test Neighborhood(CartesianRange(inds)) === box
                @test Neighborhood(CartesianRange(I1, I2)) === box
            end

            # RectangularBox constructors.
            @test RectangularBox(B1) === box
            @test RectangularBox(B1...) === box
            @test RectangularBox(B2) === box
            @test RectangularBox(B2...) === box
            @static if USE_CARTESIAN_RANGE
                @test CartesianRange(box) === CartesianRange(inds)
                @test CartesianRange(box) === CartesianRange(I1, I2)
            end

            # Conversion RectangularBox <-> CartesianIndices.
            @test CartesianIndices(box) === CartesianIndices(inds)
            @test RectangularBox(CartesianIndices(inds)) === box
            @test convert(RectangularBox, CartesianIndices(inds)) === box
            @test convert(RectangularBox{N}, CartesianIndices(inds)) === box
            @test CartesianIndices(box) === CartesianIndices(inds)
            @test convert(CartesianIndices, box) === CartesianIndices(inds)
            @test convert(CartesianIndices{N}, box) === CartesianIndices(inds)
            @static if USE_CARTESIAN_RANGE
                # Conversion RectangularBox <-> CartesianRange.
                @test RectangularBox(CartesianRange(inds)) === box
                @test convert(RectangularBox, CartesianRange(inds)) === box
                @test convert(RectangularBox{N}, CartesianRange(inds)) === box
                @test CartesianRange(box) === CartesianRange(inds)
                @test convert(CartesianRange, box) === CartesianRange(inds)
                @test convert(CartesianRange{CartesianIndex{N}}, box) === CartesianRange(inds)
            end

            # Conversion Neighborhood <-> CartesianIndices.
            @test Neighborhood(CartesianIndices(inds)) === box
            @test convert(Neighborhood, CartesianIndices(inds)) === box
            @test convert(Neighborhood{N}, CartesianIndices(inds)) === box
            @test CartesianIndices(box) === CartesianIndices(inds)
            @test convert(CartesianIndices, box) === CartesianIndices(inds)
            @test convert(CartesianIndices{N}, box) === CartesianIndices(inds)
            @static if USE_CARTESIAN_RANGE
                # Conversion Neighborhood <-> CartesianRange
                @test Neighborhood(CartesianRange(inds)) === box
                @test convert(Neighborhood, CartesianRange(inds)) === box
                @test convert(Neighborhood{N}, CartesianRange(inds)) === box
                @test CartesianRange(box) === CartesianRange(inds)
                @test convert(CartesianRange, box) === CartesianRange(inds)
                @test convert(CartesianRange{CartesianIndex{N}}, box) === CartesianRange(inds)
            end

            @test length(box) === length(CartesianIndices(inds))
            @test size(box) === size(CartesianIndices(inds))
            @test size(box) === ntuple(d -> size(box, d), N)
            @test axes(box) === ntuple(d -> axes(box, d), N)
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

    @testset "2D test with a 5x5 ball" begin
        ball5x5 = Bool[0 1 1 1 0;
                       1 1 1 1 1;
                       1 1 1 1 1;
                       1 1 1 1 1;
                       0 1 1 1 0];
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
