#isdefined(Main, :LocalFilters) || include("../src/LocalFilters.jl")
#isdefined(Main, :NaiveLocalFilters) || include("NaiveLocalFilters.jl")

module TestingLocalFilters

#using Compat
using Test
using OffsetArrays
using LocalFilters
using LocalFilters:
    FilterOrdering, ForwardFilterOrdering, ReverseFilterOrdering,
    Box, Indices, kernel_offset, kernel_range, kernel, ball, limits,
    is_morpho_math_box, check_indices, localindices,
    ranges, centered, replicate

# A bit of type-piracy for more readable error messages.
Base.show(io::IO, x::CartesianIndices) =
    print(io, "CartesianIndices($(x.indices))")

#=
# Selector for reference methods.
const REF = Val(:Base)

const ATOL = 0.0
const GTOL = 4*eps(Float64)

compare(a::AbstractArray, b::AbstractArray) = maximum(abs(a - b))
samevalues(a::AbstractArray, b::AbstractArray) = minimum(a .== b)
identical(a::RectangularBox{N}, b::RectangularBox{N}) where {N} =
    (first_cartesian_index(a) === first_cartesian_index(b) &&
     last_cartesian_index(a) === last_cartesian_index(b))
identical(a::Kernel{Ta,N}, b::Kernel{Tb,N}) where {Ta,Tb,N} =
    nearlysame(a, b; atol=0, gtol=0)
identical(a::Neighborhood, b::Neighborhood) = false

function nearlysame(a::Kernel{Ta,N}, b::Kernel{Tb,N};
                    atol=ATOL, gtol=GTOL) where {Ta,Tb,N}
    first_cartesian_index(a) == first_cartesian_index(b) || return false
    last_cartesian_index(a) == last_cartesian_index(b) || return false
    if atol == 0 && gtol == 0
        for i in neighborhood(a)
            a[i] == b[i] || return false
        end
    else
        for i in neighborhood(a)
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
=#

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

@testset "LocalFilters" begin

    @testset "Basics" begin
        # Indices
        let A = 1:4, B = [1,2,3], I = Indices(A, B)
            @test IndexStyle(I) === IndexLinear()
            @test I === Indices(A)
            @test I === Indices(B)
            @test I(A) === eachindex(IndexStyle(I), A)
            @test I(B) === eachindex(IndexStyle(I), B)
        end
        let A = [1 2; 3 4; 5 6], B = [1 2 3; 4 5 6], I = Indices(A, B)
            @test IndexStyle(I) === IndexCartesian()
            @test I === Indices(A)
            @test I === Indices(B)
            @test I(A) === eachindex(IndexStyle(I), A)
            @test I(B) === eachindex(IndexStyle(I), B)
        end

        # kernel_offset
        @test_throws ArgumentError kernel_offset(-1)
        @test kernel_offset(Int16(5)) === -3
        @test kernel_offset(Int16(4)) === -3

        # kernel_range
        @test_throws ArgumentError kernel_range(-1)
        @test isempty(kernel_range(0))
        @test kernel_range(0) === 0:-1
        @test kernel_range(1) === 0:0
        @test kernel_range(4) === -2:1
        @test kernel_range(5) === -2:2
        @test kernel_range(-4:5) === -4:5
        @test kernel_range(-4:1:5) === -4:5
        @test_throws ArgumentError kernel_range(-4:2:7)
        @test kernel_range(1,3) === 1:3
        @test kernel_range(Int16(-3),Int8(5)) === -3:5
        @test kernel_range(Base.OneTo(7)) === Base.OneTo{Int}(7)
        @test kernel_range(Base.OneTo(Int16(7))) === Base.OneTo{Int}(7)

        # Boxes.
        #@test Box(...)

        # kernel
        # FIXME: @test length(kernel()) == 0
        # FIXME: @test kernel(()) == 0
        # FIXME: @test kernel(Dims{0}) == 0
        @test kernel(Dims{2}, 6) === kernel(6, 6)
        @test kernel(Dims{2}, 6) === Box(-3:2,-3:2)
        @test kernel(Dims{2}, 5, 6) === Box(-2:2,-3:2)
        @test kernel(5, 6) === Box(-2:2,-3:2)
        @test kernel(-2:4, 6) === Box(-2:4,-3:2)
        @test kernel(CartesianIndex(-2,1,0), CartesianIndex(4,9,5)) === Box(-2:4, 1:9, 0:5)
        @test kernel(-2:4, 1:9, 0:5) === Box(-2:4, 1:9, 0:5)
        for args in ((6, -1:3, 2:4),
                     (CartesianIndex(-2,1,0), CartesianIndex(4,9,5)))
            @test kernel(args...) === kernel(args)
            @test kernel(Dims{3}, args...) ===
                kernel(Dims{3}, args)
        end
        let R = CartesianIndices((6, -1:3, 2:4))
            @test kernel(R) === Box(R)
            @test kernel(Dims{3}, R) === Box(R)
            @test Box(R) === Box{3}(R)
        end
        if VERSION ≥ v"1.6"
            # Ranges can have a step in CartesianIndices
            @test_throws ArgumentError kernel(CartesianIndices((1:2:6,)))
            @test kernel(CartesianIndices((1:1:6,))) === Box(1:6)
        end

        # ordering
        let B = reshape(collect(1:20), (4,5)), R = Box(CartesianIndices(B))
            @test B[ForwardFilter(CartesianIndex(2,3), CartesianIndex(3,5))] == B[1,2]
            @test B[ReverseFilter(CartesianIndex(2,3), CartesianIndex(-1,1))] == B[3,2]
            @test R[ForwardFilter(CartesianIndex(2,3), CartesianIndex(3,5))] == true
            @test R[ReverseFilter(CartesianIndex(2,3), CartesianIndex(-1,1))] == true
        end

        # centered
        let B = reshape(collect(1:20), (4,5)), R = CartesianIndices(B)
            @test axes(centered(B)) == (-2:1, -2:2)
            @test axes(centered(centered(B))) === axes(centered(B))
            @test ranges(centered(R)) === (-2:1, -2:2)
        end

        # replicate
        @test replicate(NTuple{3}, 'a') === ('a', 'a', 'a')
        @test replicate(NTuple{3,Char}, 'a') === ('a', 'a', 'a')
        @test replicate(NTuple{2,Int}, 'a') === (97, 97)

        # limits
        @test limits(Float32) === (-Float32(Inf), Float32(Inf))
        @test limits(Int8) === (Int8(-128),Int8(127))

        # check_indices
        let dims = (4,5), A = reshape(collect(1:prod(dims)), dims),
            B = ones(dims), C = rand(Float32, dims), D = centered(C)
            @test_throws DimensionMismatch check_indices(A, B, C, D)
            @test check_indices(A) === check_indices(D)
            @test check_indices(A,B) === check_indices(D)
            @test check_indices(A,B,C) === check_indices(D)
            @test check_indices(Bool,A) === true
            @test check_indices(Bool,A,B) === true
            @test check_indices(Bool,A,B,C) === true
            @test check_indices(Bool,A,B,C,D) === false
            @test check_indices(Bool,centered(A),D) === true
            @test check_indices(Bool) === false
            @test check_indices(Bool,axes(A)) === false
            @test check_indices(Bool,axes(A),B) === true
            @test check_indices(Bool,axes(A),B,C) === true
            @test check_indices(Bool,axes(A),B,C,D) === false
        end

        # ForwardFilter/ReverseFilter
        @test reverse(ForwardFilter) === ReverseFilter
        @test reverse(ReverseFilter) === ForwardFilter
        @test (ForwardFilter isa FilterOrdering) == true
        @test (ForwardFilter isa ForwardFilterOrdering) == true
        @test (ForwardFilter isa ReverseFilterOrdering) == false
        @test (ReverseFilter isa FilterOrdering) == true
        @test (ReverseFilter isa ForwardFilterOrdering) == false
        @test (ReverseFilter isa ReverseFilterOrdering) == true
        let i = 3, j = 7
            @test ForwardFilter(i, j) === j - i
            @test ReverseFilter(i, j) === i - j
            @test ForwardFilter(Int16(i), Int16(j)) === j - i
            @test ReverseFilter(Int16(i), Int16(j)) === i - j
        end
        let i = CartesianIndex(3,4,5), j = CartesianIndex(-1,7,3)
            @test ForwardFilter(i, j) === j - i
            @test ReverseFilter(i, j) === i - j
        end

        # result_eltype
        # is_morpho_math_box
        # strel

        # ball
        @test ball(Dims{2}, 1) == ball3x3
        @test ball(Dims{2}, 2) == ball5x5
        @test ball(Dims{2}, 3) == ball7x7
    end

    @testset "Local mean" begin
        let A = ones(Float64, 20)
            @test localmean(A, 3) == A
            @test localmean(A, ForwardFilter, 3) == A
            @test localmean(A, ReverseFilter, 3) == A
        end
    end
#=
    @testset "Neighborhoods" begin
        for (dims, rngs) in (((3,), (-1:1,)),
                             ((3, 4, 5), (-1:1, -2:1, -2:2)))
            N = length(dims)
            box = neighborhood(dims)
            fse = strel(Bool, box) # flat structuring element
            Imin, Imax = first(box), last(box)
            A = rand(dims...)
            msk = rand(Bool, dims)
            ker = centered(A)

            # Test limits(), first_cartesian_index() and last_cartesian_index().
            @test first_cartesian_index(CartesianIndices(rngs)) === Imin
            @test last_cartesian_index(CartesianIndices(rngs)) === Imax
            @test limits(CartesianIndices(rngs)) === (Imin, Imax)
            @test first_cartesian_index(box) === Imin
            @test last_cartesian_index(box) === Imax
            @test limits(box) === (Imin, Imax)
            @test first_cartesian_index(A) === oneunit(CartesianIndex{N})
            @test last_cartesian_index(A) === CartesianIndex(size(A))
            @test limits(A) === (oneunit(CartesianIndex{N}),
                                 CartesianIndex(size(A)))

            # Test neighborhood().
            region = CartesianIndices(rngs)
            @test neighborhood(Imin,Imax) === region
            @test neighborhood(CartesianIndices(rngs)) === region
            @test neighborhood(box) === region
            @test neighborhood(A) === CartesianIndices(A)

            # Neighborhood constructors.
            @test neighborhood(box) === box
            @test neighborhood(ker) === ker
            @test neighborhood(A) === ker
            @test neighborhood(dims...) === box
            @test neighborhood(rngs) === box
            @test neighborhood(rngs...) === box
            @test neighborhood(CartesianIndices(rngs)) === box
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
            @test neighborhood(Dims{N}, dim) === RectangularBox(ntuple(d -> dim, N))
            @test neighborhood(Dims{N}, rng) === RectangularBox(ntuple(d -> rng, N))

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
            @test_deprecated Kernel(eltype(A), ker) === ker
            @test identical(Kernel(box), Kernel(ones(Bool, dims)))
            @test identical(Kernel{Bool}(box), Kernel(ones(Bool, dims)))
            @test Kernel(A, first_cartesian_index(ker)) === ker
            @test Kernel(A, rngs) === ker
            @test Kernel(A, rngs...) === ker
            @test Kernel(A, CartesianIndices(ker)) === ker
            off = first_cartesian_index(A) - first_cartesian_index(ker)
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

            # Other basic methods.
            @test length(box) === length(CartesianIndices(rngs))
            @test size(box) === size(CartesianIndices(rngs))
            @test size(box) === ntuple(d -> size(box, d), N)
            @test axes(box) === ntuple(d -> axes(box, d), N)
            @test length(ker) === length(A)
            @test size(ker) === size(A)
            @test size(ker) === ntuple(d -> size(ker, d), N)
            @test axes(ker) === ntuple(d -> axes(ker, d), N)
            @test is_morpho_math_box(fse) == true
            @test is_morpho_math_box(box) == true
            boolker = neighborhood(ones(Bool,dims))
            @test is_morpho_math_box(boolker) == true
            boolker[zero(CartesianIndex{N})] = false
            @test is_morpho_math_box(boolker) == false
            #@test is_morpho_math_box(Empty{N}()) == false

            # Test reverse().
            revbox = reverse(box)
            revker = reverse(ker)
            @test first_cartesian_index(revbox) === -last_cartesian_index(box)
            @test last_cartesian_index(revbox) === -first_cartesian_index(box)
            @test first_cartesian_index(revker) === -last_cartesian_index(ker)
            @test last_cartesian_index(revker) === -first_cartesian_index(ker)
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
            box = neighborhood(Dims{N}, 5)
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
        A = randn(Float64, 128, 100)
        σr = 1.2
        σs = 2.5
        width = LocalFilters.BilateralFilter.default_width(σs)
        box = neighborhood(Dims{2}, width)
        B0 = bilateralfilter(A, σr, σs)
        B1 = bilateralfilter(A, σr, σs, box)
        B2 = bilateralfilter(Float32, A, σr, σs, width)
        B3 = bilateralfilter!(similar(Array{Float32}, axes(A)), A, σr, σs, box)
        @test similarvalues(B1, B0; atol=0, gtol=8*eps(Float32))
        @test similarvalues(B2, B0; atol=0, gtol=8*eps(Float32))
        @test similarvalues(B3, B0; atol=0, gtol=4*eps(Float32))
    end
=#
end

end # module
