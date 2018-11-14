isdefined(Base, :LocalFilters) || include("../src/LocalFilters.jl")

module LocalFiltersTests

using Compat
using LocalFilters, Compat.Test
import LocalFilters: Neighborhood, RectangularBox, Kernel,
    firstindex, lastindex, limits

replicate(a, n::Integer) = ntuple(i->a, n)
compare(a, b) = maximum(abs(a - b))
samevalues(a, b) = minimum(a .== b)

const trivialerode = erode
const trivialdilate = dilate
const triviallocalextrema = localextrema

@testset "LocalFilters" begin

    @testset "Neighborhoods" begin
        for (B1, B2) in (((3,), (-1:1,)),
                         ((3, 4, 5), (-1:1, -2:1, -2:2)))
            box = Neighborhood(B1)
            I1, I2 = limits(box)
            inds = map((i,j) -> i:j, I1.I, I2.I)
            @test Neighborhood(B1...) === box
            @test Neighborhood(B2) === box
            @test Neighborhood(B2...) === box
            @test Neighborhood(CartesianIndices(inds)) === box
            @test CartesianIndices(box) === CartesianIndices(inds)
            @static if !isdefined(Base, :CartesianIndices)
                @test Neighborhood(CartesianRange(inds)) === box
                @test Neighborhood(CartesianRange(I1, I2)) === box
                @test CartesianRange(box) === CartesianRange(inds)
                @test CartesianRange(box) === CartesianRange(I1, I2)
            end
            @test RectangularBox(B1) === box
            @test RectangularBox(B1...) === box
            @test RectangularBox(B2) === box
            @test RectangularBox(B2...) === box
            @test RectangularBox(CartesianIndices(inds)) === box
            @static if !isdefined(Base, :CartesianIndices)
                @test RectangularBox(CartesianRange(inds)) === box
            end
            @test length(box) === length(CartesianIndices(inds))
            @test size(box) === size(CartesianIndices(inds))
            # FIXME: @test axes(box) === axes(CartesianIndices(inds))
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
