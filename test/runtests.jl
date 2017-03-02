module LocalFiltersTests

const DEBUG = false

using LocalFilters, Base.Test
import LocalFilters: Neighborhood, CenteredBox, CartesianBox, Kernel,
    anchor
import Base.CartesianRange

replicate(a, n::Integer) = ntuple(i->a, n)
compare(a, b) = maximum(abs(a - b))
samevalues(a, b) = maximum(a == b)

if DEBUG
    a = rand(62,81)
    cbox = CenteredBox(3,5)
    rbox = CartesianBox(cbox)
    mask = Kernel(cbox)
    kern = Kernel(eltype(a), mask)
    result = erode(a, cbox)
    println("erode 1 -> ", samevalues(erode(a, rbox), result))
    println("erode 2 -> ", samevalues(erode(a, mask), result))
    println("erode 3 -> ", samevalues(erode(a, kern), result))
    result = dilate(a, cbox)
    println("dilate 1 -> ", samevalues(dilate(a, rbox), result))
    println("dilate 2 -> ", samevalues(dilate(a, mask), result))
    println("dilate 3 -> ", samevalues(dilate(a, kern), result))
end

@testset "LocalFilters" begin
    for (arrdims, boxdims) in (((341,),    (3,)),
                               ((62,81),   (5,3)),
                               ((23,28,27),(3,7,5)))
        rank = length(arrdims)
        @testset "  $(rank)D test with boxes" begin
            a = rand(arrdims)
            cbox = CenteredBox(boxdims)
            rbox = CartesianBox(cbox)
            mask = Kernel(cbox)
            kern = Kernel(eltype(a), mask)
            @testset "    erode" begin
                result = erode(a, cbox)
                @test samevalues(erode(a, rbox), result)
                @test samevalues(erode(a, mask), result)
                @test samevalues(erode(a, kern), result)
            end
            @testset "    dilate" begin
                result = dilate(a, cbox)
                @test samevalues(dilate(a, rbox), result)
                @test samevalues(dilate(a, mask), result)
                @test samevalues(dilate(a, kern), result)
            end
        end
    end

    @testset "  2D test with a 5x5 ball" begin
        ball5x5 = [false true true true false;
                   true  true true true true;
                   true  true true true true;
                   true  true true true true;
                   false true true true false];
        a = rand(162,181)
        mask = Kernel(ball5x5)
        kern = Kernel(eltype(a), mask)
        @testset "  erode" begin
            result = erode(a, mask)
            @test samevalues(erode(a, kern), result)
        end
        @testset "  dilate" begin
            result = dilate(a, mask)
            @test samevalues(dilate(a, kern), result)
        end
    end

end

end # module
