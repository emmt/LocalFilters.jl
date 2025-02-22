module TestingLocalFilters

#using Compat
using Test
using OffsetArrays, StructuredArrays
using LocalFilters
using LocalFilters:
    FilterOrdering, ForwardFilterOrdering, ReverseFilterOrdering,
    Box, box, ball, Indices, kernel_range, kernel, limits,
    is_morpho_math_box, check_axes, localindices,
    ranges, centered, centered_offset, unit_range,
    top_hat!, bottom_hat!,
    Returns

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
=#
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
#=
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


@testset "LocalFilters" begin

    @testset "Utilities" begin

        @testset "Returns" begin
            @test (1,1,1) === @inferred ntuple(Returns(1), Val(3))
            @test (0x1,0x1,0x1) === @inferred ntuple(Returns{UInt8}(1), Val(3))
        end

        @testset "Indices" begin
            A = 1:4
            B = [1,2,3]
            I = @inferred Indices(A, B)
            @test IndexStyle(I) === IndexLinear()
            @test I === Indices(A)
            @test I === Indices(B)
            @test I(A) === eachindex(IndexStyle(I), A)
            @test I(B) === eachindex(IndexStyle(I), B)
            @test_throws DimensionMismatch I(A, B)
            A = [1 2; 3 4; 5 6]
            B = [1 2 3; 4 5 6]
            I = @inferred Indices(A, B)
            @test IndexStyle(I) === IndexCartesian()
            @test I === Indices(A)
            @test I === Indices(B)
            @test I(A) === eachindex(IndexStyle(I), A)
            @test I(B) === eachindex(IndexStyle(I), B)
            @test_throws DimensionMismatch I(A, B)
            for S in (IndexLinear, IndexCartesian)
                I = @inferred Indices{S}()
                @test IndexStyle(I) === S()
                @test I(A) === eachindex(S(), A)
                @test I(B) === eachindex(S(), B)
            end
        end

        @testset "check_axes" begin
            dims = (4,5)
            A = reshape(collect(1:prod(dims)), dims)
            B = ones(dims)
            C = rand(Float32, dims)
            D = centered(C)
            E = ones(dims..., 2)
            @test_throws DimensionMismatch check_axes(A, B, C, D)
            @test_throws DimensionMismatch check_axes(A, E)
            @test_throws DimensionMismatch check_axes(axes(A), B, C, D)
            @test_throws DimensionMismatch check_axes(axes(A), E)
            @test check_axes(A) === check_axes(D)
            @test check_axes(A,B) === check_axes(D)
            @test check_axes(A,B,C) === check_axes(D)
            @test check_axes(Bool,A) === true
            @test check_axes(Bool,A,B) === true
            @test check_axes(Bool,A,B,C) === true
            @test check_axes(Bool,A,B,C,D) === false
            @test check_axes(Bool,centered(A),D) === true
            @test check_axes(Bool) === false
            @test check_axes(Bool,axes(A)) === false
            @test check_axes(Bool,axes(A),B) === true
            @test check_axes(Bool,axes(A),B,C) === true
            @test check_axes(Bool,axes(A),B,C,D) === false
        end

        @testset "unit_range" begin
            @test (@inferred unit_range(-1, 8)) === -1:8
            @test (@inferred unit_range(Int16(-3), Int16(-1))) === -3:-1
            @test (@inferred unit_range(0:3)) === 0:3
            @test (@inferred unit_range(0x1:0x4)) === 1:4
            @test (@inferred unit_range(Base.OneTo(9))) === Base.OneTo(9)
            @test (@inferred unit_range(Base.OneTo{Int8}(6))) === Base.OneTo(6)
            @test (@inferred unit_range(-2:1:7)) === -2:7
            @test (@inferred unit_range(7:-1:-2)) === -2:7
            @test (@inferred unit_range(0x2:0x1:0x20)) === 2:32
            @test_throws ArgumentError unit_range(0x2:0x2:0x20)
            @test (@inferred unit_range(CartesianIndices((-1:6, 0x0:0xa)))) === CartesianIndices((-1:6, 0:10))
            @test (@inferred unit_range(CartesianIndices((Base.OneTo(7), Int16(3):Int16(3))))) === CartesianIndices((Base.OneTo(7), 3:3))
            if VERSION ≥ v"1.6"
                # Ranges can have a step in CartesianIndices
                @test_throws ArgumentError unit_range(CartesianIndices((1:2:6,)))
                @test (@inferred unit_range(CartesianIndices((2:1:6,)))) === CartesianIndices((2:6,))
                @test (@inferred unit_range(CartesianIndices((6:-1:2,)))) === CartesianIndices((2:6,))
            end
        end

        @testset "centered_offset" begin
            @test_throws ArgumentError centered_offset(-1)
            @test centered_offset(Int16(5)) === -3
            @test centered_offset(Int16(4)) === -3
        end

        @testset "kernel_range" begin
            @test_throws ArgumentError kernel_range(-1)
            @test isempty(kernel_range(0))
            @test (@inferred kernel_range(0)) === 0:-1
            @test (@inferred kernel_range(1)) === 0:0
            @test (@inferred kernel_range(4)) === -2:1
            @test (@inferred kernel_range(5)) === -2:2
            @test (@inferred kernel_range(-4:5)) === -4:5
            @test (@inferred kernel_range(-4:1:5)) === -4:5
            @test_throws ArgumentError kernel_range(-4:2:7)
            @test (@inferred kernel_range(1,3)) === 1:3
            @test (@inferred kernel_range(Int16(-3),Int8(5))) === -3:5
            @test (@inferred kernel_range(Base.OneTo(7))) === Base.OneTo{Int}(7)
            @test (@inferred kernel_range(Base.OneTo(Int16(7)))) === Base.OneTo{Int}(7)
            @test (@inferred kernel_range(FORWARD_FILTER, -4:5)) === -4:5
            @test (@inferred kernel_range(REVERSE_FILTER, -4:5)) === -5:4
            @test (@inferred kernel_range(FORWARD_FILTER, -4:1:5)) === -4:5
            @test (@inferred kernel_range(REVERSE_FILTER, -4:1:5)) === -5:4
            @test (@inferred kernel_range(FORWARD_FILTER, 2, 8)) === 2:8
            @test (@inferred kernel_range(REVERSE_FILTER, -1, 4)) === -4:1
        end

        @testset "kernel" begin
            B = FastUniformArray(true)
            @test B === @inferred kernel()
            @test B === @inferred kernel(Dims{0})
            @test B === @inferred kernel(())
            @test B === @inferred kernel(Dims{0}, ())
            B = box(-3:2, -3:2, -3:2)
            @test B === @inferred kernel(6, 6, 6)
            @test B === @inferred kernel(Int16(6), -3:2, Int8(-3):Int8(2))
            @test B === @inferred kernel(Dims{3}, 6, 6, 6)
            @test B === @inferred kernel(Dims{3}, Int8(6))
            B = box(-2:2, -3:2, -3:3)
            @test B === @inferred kernel(5, 6, 7)
            @test B === @inferred kernel(Dims{3}, 5, 6, 7)
            B = box(-2:4, -3:2)
            @test B === @inferred kernel(-2:4, 6)
            @test B === @inferred kernel(-2:4, -3:2)
            @test B === @inferred kernel(Dims{2}, -2:4, 6)
            @test B === @inferred kernel(Dims{2}, -2:4, -3:2)
            B = box(-3:2, 1:9, 0:5)
            I, J = CartesianIndex(-3,1,0), CartesianIndex(2,9,5)
            @test B === @inferred kernel(I, J)
            @test B === @inferred kernel((I, J))
            @test B === @inferred kernel(Dims{3}, I, J)
            @test B === @inferred kernel(Dims{3}, (I, J))
            @test B === @inferred kernel(6, 1:9, 0:5)
            @test B === @inferred kernel(Dims{3}, 6, 1:9, 0:5)
            R = CartesianIndices((6, -1:3, 2:4, -2:2))
            B = box(R)
            @test B === @inferred kernel(R)
            @test B === @inferred kernel(Dims{4}, R)
            if VERSION ≥ v"1.6"
                # Ranges can have a step in CartesianIndices
                @test_throws ArgumentError kernel(CartesianIndices((1:2:6,)))
                @test kernel(CartesianIndices((1:1:6,))) === box(1:6)
            end
            A = reshape(1:24, (2,3,4))
            @test A === @inferred kernel(A)
            @test A === @inferred kernel(Dims{3}, A)
            @test_throws ArgumentError kernel(Dims{1}, A)
            @test_throws ArgumentError kernel(Dims{11})
            @test_throws ArgumentError kernel(Dims{11}, :e)
            @test_throws ArgumentError kernel(Dims{11}, :e, 1)
            @test_throws ArgumentError kernel(Dims{11}, :e, 1, [1,3])
        end

        @testset "reverse_kernel" begin
            A = FastUniformArray(true)
            B = A
            @test B === @inferred reverse_kernel(A)
            @test B === @inferred reverse_kernel()
            @test B === @inferred reverse_kernel(Dims{0})
            @test B === @inferred reverse_kernel(())
            @test B === @inferred reverse_kernel(Dims{0}, ())
            A = box(-3:2, -3:2, -3:2)
            B = box(-2:3, -2:3, -2:3)
            @test B === @inferred reverse_kernel(A)
            @test A === @inferred reverse_kernel(B)
            @test B === @inferred reverse_kernel(6, 6, 6)
            @test B === @inferred reverse_kernel(Int16(6), -3:2, Int8(-3):Int8(2))
            @test B === @inferred reverse_kernel(Dims{3}, 6, 6, 6)
            @test B === @inferred reverse_kernel(Dims{3}, Int8(6))
            A = box(-2:2, -3:2, -3:3)
            B = box(-2:2, -2:3, -3:3)
            @test B === @inferred reverse_kernel(A)
            @test A === @inferred reverse_kernel(B)
            @test B === @inferred reverse_kernel(5, 6, 7)
            @test B === @inferred reverse_kernel(Dims{3}, 5, 6, 7)
            A = box(-2:4, -3:2)
            B = box(-4:2, -2:3)
            @test B === @inferred reverse_kernel(A)
            @test A === @inferred reverse_kernel(B)
            @test B === @inferred reverse_kernel(-2:4, 6)
            @test B === @inferred reverse_kernel(-2:4, -3:2)
            @test B === @inferred reverse_kernel(Dims{2}, -2:4, 6)
            @test B === @inferred reverse_kernel(Dims{2}, -2:4, -3:2)
            A = box(-3:2, 1:9, 0:5)
            B = box(-2:3, -9:-1, -5:0)
            @test B === @inferred reverse_kernel(A)
            @test A === @inferred reverse_kernel(B)
            @test B === @inferred reverse_kernel(CartesianIndex(-3,1,0), CartesianIndex(2,9,5))
            @test B === @inferred reverse_kernel(Dims{3}, CartesianIndex(-3,1,0), CartesianIndex(2,9,5))
            @test B === @inferred reverse_kernel(6, 1:9, 0:5)
            @test B === @inferred reverse_kernel(Dims{3}, 6, 1:9, 0:5)
            R = CartesianIndices((6, -1:3, 2:4, -2:2))
            A = box(R)
            B = if length(R.indices[1]) == 1
                # Prior to Julia 1.6, a Cartesian index range specified as a scalar `i` is
                # assumed to correspond to the single index `i`.
                box(-6:-6, -3:1, -4:-2, -2:2)
            else
                # Starting with Julia 1.6, a Cartesian index range specified as a scalar
                # `n` is assumed to correspond to the range `1:n`,
                box(-6:-1, -3:1, -4:-2, -2:2)
            end
            @test B === @inferred reverse_kernel(A)
            @test A ==  @inferred reverse_kernel(B)
            @test B === @inferred reverse_kernel(R)
            @test B === @inferred reverse_kernel(Dims{ndims(A)}, R)
            if VERSION ≥ v"1.6"
                # Ranges can have a step in CartesianIndices
                @test_throws ArgumentError reverse_kernel(CartesianIndices((1:2:6,)))
                @test reverse_kernel(CartesianIndices((1:1:6,))) === box(-6:-1)
            end
            A = reshape(1:24, (2,3,4))
            B = @inferred reverse_kernel(A)
            @test axes(B) == map(d -> -d:-1, size(A))
            @test A == @inferred reverse_kernel(reverse_kernel(A))
            @test B == @inferred reverse_kernel(Dims{ndims(A)}, A)
            @test_throws ArgumentError reverse_kernel(Dims{1}, A)
            @test_throws ArgumentError reverse_kernel(Dims{11})
            @test_throws ArgumentError reverse_kernel(Dims{11}, :e)
            @test_throws ArgumentError reverse_kernel(Dims{11}, :e, 1)
            @test_throws ArgumentError reverse_kernel(Dims{11}, :e, 1, [1,3])
            # Check that inference works on uniform arrays for `reverse_kernel`.
            A = @inferred Box{3}(7,8,9)
            B = @inferred FastUniformArray{Bool,3,true}(axes(A))
            C = @inferred UniformArray(first(A), axes(A))
            D = @inferred MutableUniformArray(first(A), axes(A))
            @test B === A
            @test C == A
            @test D == A
            Ar = @inferred reverse_kernel(A)
            Br = @inferred reverse_kernel(B)
            Cr = @inferred reverse_kernel(C)
            Dr = @inferred reverse_kernel(D)
            @test Br === Ar
            @test Cr == Ar
            @test Dr == Ar
        end

        @testset "Ordering" begin
            @test reverse(FORWARD_FILTER) === REVERSE_FILTER
            @test reverse(REVERSE_FILTER) === FORWARD_FILTER
            @test (FORWARD_FILTER isa FilterOrdering) == true
            @test (FORWARD_FILTER isa ForwardFilterOrdering) == true
            @test (FORWARD_FILTER isa ReverseFilterOrdering) == false
            @test (REVERSE_FILTER isa FilterOrdering) == true
            @test (REVERSE_FILTER isa ForwardFilterOrdering) == false
            @test (REVERSE_FILTER isa ReverseFilterOrdering) == true
            i, j = 3, 7
            @test FORWARD_FILTER(i, j) === j - i
            @test REVERSE_FILTER(i, j) === i - j
            @test FORWARD_FILTER(Int16(i), Int16(j)) === j - i
            @test REVERSE_FILTER(Int16(i), Int16(j)) === i - j
            i, j = CartesianIndex(3,4,5), CartesianIndex(-1,7,3)
            @test FORWARD_FILTER(i, j) === j - i
            @test REVERSE_FILTER(i, j) === i - j
            B = reshape(collect(1:20), (4,5))
            R = box(CartesianIndices(B))
            @test B[FORWARD_FILTER(CartesianIndex(2,3), CartesianIndex(3,5))] == B[1,2]
            @test B[REVERSE_FILTER(CartesianIndex(2,3), CartesianIndex(-1,1))] == B[3,2]
            @test R[FORWARD_FILTER(CartesianIndex(2,3), CartesianIndex(3,5))] == true
            @test R[REVERSE_FILTER(CartesianIndex(2,3), CartesianIndex(-1,1))] == true
        end

        @testset "centered" begin
            B = reshape(collect(1:20), (4,5))
            @test axes(centered(B)) == (-2:1, -2:2)
            @test axes(centered(centered(B))) === axes(centered(B))
            @test centered(centered(B)) === centered(B)
        end

        @testset "limits" begin
            @test limits(Float32) === (-Float32(Inf), Float32(Inf))
            @test limits(Int8) === (Int8(-128),Int8(127))
        end

        @testset "ball" begin
            ball3x3 = centered(Bool[1 1 1;
                                    1 1 1;
                                    1 1 1]);
            ball5x5 = centered(Bool[0 1 1 1 0;
                                    1 1 1 1 1;
                                    1 1 1 1 1;
                                    1 1 1 1 1;
                                    0 1 1 1 0]);
            ball7x7 = centered(Bool[0 0 1 1 1 0 0;
                                    0 1 1 1 1 1 0;
                                    1 1 1 1 1 1 1;
                                    1 1 1 1 1 1 1;
                                    1 1 1 1 1 1 1;
                                    0 1 1 1 1 1 0;
                                    0 0 1 1 1 0 0]);
            @test ball(Dims{2}, 1.5) == ball3x3
            @test ball(Dims{2}, 2.5) == ball5x5
            @test ball(Dims{2}, 3.5) == ball7x7
            @test ball(Dims{3}, 3) == ball(Dims{3}, 3.0)
        end

        @testset "is_morpho_math_box" begin
            @test  is_morpho_math_box(box(-1:2, 2:4))
            @test  is_morpho_math_box(ones(Bool, 3, 4))
            @test !is_morpho_math_box(ball(Dims{2}, 1.0))
            @test  is_morpho_math_box(ball(Dims{2}, 1.5))
            @test !is_morpho_math_box(ball(Dims{2}, 2.0))
            A = Array{Float32}(undef, 2, 3, 4)
            @test  is_morpho_math_box(fill!(A, 0))
            @test !is_morpho_math_box(fill!(A, 1))
            @test !is_morpho_math_box(fill!(A, -Inf))
            R = CartesianIndices(A)
            @test_throws Exception is_morpho_math_box(R)
        end

        @testset "strel" begin
            T = Float32
            A = @inferred box(-3:3)
            B = FastUniformArray(zero(T), axes(A))
            @test A === @inferred strel(Bool, A)
            @test A === @inferred strel(Bool, CartesianIndices(A))
            @test B === @inferred strel(T, CartesianIndices(A))
            A = @inferred box(2, -1:2)
            B = FastUniformArray(zero(T), axes(A))
            @test A === @inferred strel(Bool, A)
            @test A === @inferred strel(Bool, CartesianIndices(A))
            @test B === @inferred strel(T, CartesianIndices(A))
            A = @inferred centered(ones(Bool, 3, 4, 5))
            S = @inferred box(axes(A))
            B = FastUniformArray(zero(T), axes(A))
            @test A === @inferred strel(Bool, A)
            @test A ==  @inferred strel(Bool, CartesianIndices(A))
            @test S === @inferred strel(Bool, CartesianIndices(A))
            @test B === @inferred strel(T, CartesianIndices(A))
            A = @inferred ball(Dims{3}, 3.5)
            B = map(x -> x ? zero(T) : -T(Inf), A)
            @test A === @inferred strel(Bool, A)
            @test B ==  @inferred strel(T, A)
        end

    end # @testset "Utilities"

    @testset "Local mean" begin
        A = ones(Float64, 20)
        @test localmean(A, 3) == A
        @test localmean(A, FORWARD_FILTER, 3) == A
        @test localmean(A, REVERSE_FILTER, 3) == A
    end

    # See https://github.com/emmt/LocalFilters.jl/issues/6
    @testset "Roughness filter" begin
        rng = -1:1 # neighborhood range (length must be odd for this example to work)
        A = rand(Float32, 20, 30)
        # LocalFilters version.
        B = @inferred localfilter!(
            similar(A), A, length(rng),
            #= initial =# (a) -> (zero(a), a),
            #= update =# (v, a, _) -> (max(v[1], abs(a - v[2])), v[2]),
            #= final =# (v) -> v[1])
        # Slow version.
        C = similar(A)
        I = CartesianIndices(A)
        J = CartesianIndices((rng, rng))
        for i in I
            v = zero(eltype(A))
            for j in J
                k = i + j
                if k ∈ I
                    v = max(v, abs(A[k] - A[i]))
                end
            end
            C[i] = v
        end
        @test C ≈ B
    end

    @testset "Morphology (T = $T)" for (T, dims) in ((UInt8, (12, 13)), (Float32, (4,5,6)))
        A = rand(T, dims);    # source
        N = ndims(A);
        S = ball(Dims{N}, 2.5);   # shaped structuring element
        R = box((-2:2, -1:2, -1:1)[1:N]); # simple hyper-rectangular box
        wrk = similar(A);     # workspace
        B2 = similar(A);      # for in-place operation
        C = copy(A);          # to check that the source is left unchanged
        @testset "$name" for (name, func, func!) in (("Erosion", erode, erode!),
                                                     ("Dilation", dilate, dilate!))
            # ... with a simple rectangular structuring element
            B1 = @inferred func(A, R; slow=true);
            @test C == A   # check that A is left unchanged
            @test B2 === @inferred func!(B2, A, R; slow=true)
            @test C == A   # check that A is left unchanged
            @test B2 == B1 # check if in-place and out-of-place yield the same result
            @test B1 == @inferred func(A, R; slow=false);
            @test C == A   # check that A is left unchanged
            @test B2 === @inferred func!(B2, A, R; slow=false)
            @test C == A   # check that A is left unchanged
            @test B2 == B1 # check if in-place and out-of-place yield the same result
            # FIXME: @test B2 === @inferred func!(copyto!(B2, A), R)
            # FIXME: @test B2 == B1 # check if in-place and out-of-place yield the same result
            if T <: AbstractFloat
                F = @inferred strel(T, R) # flat structuring element like R
                @test B1 == @inferred func(A, F)
                @test C == A   # check that A is left unchanged
                @test B2 === @inferred func!(B2, A, F)
                @test C == A   # check that A is left unchanged
                @test B2 == B1 # check if in-place and out-of-place yield the same result
            end
            # ... with a shaped structuring element
            B1 = @inferred func(A, S);
            @test C == A   # check that A is left unchanged
            @test B2 === @inferred func!(B2, A, S)
            @test C == A   # check that A is left unchanged
            @test B2 == B1 # check if in-place and out-of-place yield the same result
            if T <: AbstractFloat
                F = @inferred strel(T, S) # flat structuring element like S
                @test B1 == @inferred func(A, F)
                @test C == A   # check that A is left unchanged
                @test B2 === @inferred func!(B2, A, F)
                @test C == A   # check that A is left unchanged
                @test B2 == B1 # check if in-place and out-of-place yield the same result
            end
        end
        @testset "Local min. and max." begin
            B1 = similar(A)
            # ... with a simple rectangular structuring element
            A1, A2 = @inferred localextrema(A, R);
            @test C == A   # check that A is left unchanged
            @test A1 == @inferred erode(A, R)  # `erode` also yields local min.
            @test A2 == @inferred dilate(A, R) # `dilate` also yields local max.
            @test (B1, B2) === @inferred localextrema!(B1, B2, A, R)
            @test C == A   # check that A is left unchanged
            @test B1 == A1
            @test B2 == A2
            if T <: AbstractFloat
                F = @inferred strel(T, R) # flat structuring element like R
                @test (A1, A2) == @inferred localextrema(A, F)
                @test C == A   # check that A is left unchanged
                @test (B1, B2) === @inferred localextrema!(B1, B2, A, F)
                @test C == A   # check that A is left unchanged
                @test B1 == A1
                @test B2 == A2
            end
            # ... with a shaped structuring element
            A1, A2 = @inferred localextrema(A, S);
            @test C == A   # check that A is left unchanged
            @test A1 == @inferred erode(A, S)  # `erode` also yields local min.
            @test A2 == @inferred dilate(A, S) # `dilate` also yields local max.
            @test (B1, B2) === @inferred localextrema!(B1, B2, A, S)
            @test C == A   # check that A is left unchanged
            @test B1 == A1
            @test B2 == A2
            if T <: AbstractFloat
                F = @inferred strel(T, S) # flat structuring element like S
                @test (A1, A2) == @inferred localextrema(A, F)
                @test C == A   # check that A is left unchanged
                @test (B1, B2) === @inferred localextrema!(B1, B2, A, F)
                @test C == A   # check that A is left unchanged
                @test B1 == A1
                @test B2 == A2
            end
        end
        @testset "$name" for (name, func, func!) in (("Opening", opening, opening!),
                                                     ("Closing", closing, closing!))
            # ... with a simple rectangular structuring element
            B1 = @inferred func(A, R; slow=true);
            @test C == A   # check that A is left unchanged
            if func === opening
                @test B1 == @inferred dilate(erode(A, R), R) # opening is erosion followed by dilation
            elseif func === closing
                @test B1 == @inferred erode(dilate(A, R), R) # closing is dilation followed by erosion
            end
            @test B2 === @inferred func!(B2, wrk, A, R; slow=true)
            @test C == A   # check that A is left unchanged
            @test B2 == B1 # check if in-place and out-of-place yield the same result
            @test B1 == @inferred func(A, R; slow=false);
            @test C == A   # check that A is left unchanged
            @test B2 === @inferred func!(B2, wrk, A, R; slow=false)
            @test C == A   # check that A is left unchanged
            @test B2 == B1 # check if in-place and out-of-place yield the same result
            if T <: AbstractFloat
                F = @inferred strel(T, R) # flat structuring element like R
                @test B1 == @inferred func(A, F)
                @test C == A   # check that A is left unchanged
                @test B2 === @inferred func!(B2, wrk, A, F)
                @test C == A   # check that A is left unchanged
                @test B2 == B1 # check if in-place and out-of-place yield the same result
            end
            # ... with a shaped structuring element
            B1 = @inferred func(A, S);
            @test C == A   # check that A is left unchanged
            if func === opening
                @test B1 == @inferred dilate(erode(A, S), S) # opening is erosion followed by dilation
            elseif func === closing
                @test B1 == @inferred erode(dilate(A, S), S) # closing is dilation followed by erosion
            end
            @test B2 === @inferred func!(B2, wrk, A, S)
            @test C == A   # check that A is left unchanged
            @test B2 == B1 # check if in-place and out-of-place yield the same result
            if T <: AbstractFloat
                F = @inferred strel(T, S) # flat structuring element like S
                @test B1 == @inferred func(A, F)
                @test C == A   # check that A is left unchanged
                @test B2 === @inferred func!(B2, wrk, A, F)
                @test C == A   # check that A is left unchanged
                @test B2 == B1 # check if in-place and out-of-place yield the same result
            end
        end
        @testset "$name" for (name, func, func!) in (("Top-hat", top_hat, top_hat!),
                                                     ("Bottom-hat", bottom_hat, bottom_hat!))
            # ... with a simple rectangular structuring element
            B1 = @inferred func(A, R; slow=true);
            @test C == A   # check that A is left unchanged
            if func === top_hat
                @test B1 == A .- opening(A, R) # definition of top-hat
            elseif func === bottom_hat
                @test B1 == closing(A, R) .- A # definition of bottom-hat
            end
            @test B2 === @inferred func!(B2, wrk, A, R; slow=true)
            @test C == A   # check that A is left unchanged
            @test B2 == B1 # check that in-place and out-of-place yield the same result
            @test B1 == @inferred func(A, R; slow=false);
            @test C == A   # check that A is left unchanged
            @test B2 === @inferred func!(B2, wrk, A, R; slow=false)
            @test C == A   # check that A is left unchanged
            @test B2 == B1 # check that in-place and out-of-place yield the same result
            if T <: AbstractFloat
                F = @inferred strel(T, R) # flat structuring element like R
                @test B1 == @inferred func(A, F)
                @test C == A   # check that A is left unchanged
                @test B2 === @inferred func!(B2, wrk, A, F)
                @test C == A   # check that A is left unchanged
                @test B2 == B1 # check if in-place and out-of-place yield the same result
            end
            # ... with a shaped structuring element
            B1 = @inferred func(A, S);
            @test C == A   # check that A is left unchanged
            if func === top_hat
                @test B1 == A .- opening(A, S) # definition of top-hat
            elseif func === bottom_hat
                @test B1 == closing(A, S) .- A # definition of bottom-hat
            end
            @test B2 === @inferred func!(B2, wrk, A, S)
            @test C == A   # check that A is left unchanged
            @test B2 == B1 # check if in-place and out-of-place yield the same result
            if T <: AbstractFloat
                F = @inferred strel(T, S) # flat structuring element like S
                @test B1 == @inferred func(A, F)
                @test C == A   # check that A is left unchanged
                @test B2 === @inferred func!(B2, wrk, A, F)
                @test C == A   # check that A is left unchanged
                @test B2 == B1 # check if in-place and out-of-place yield the same result
            end
        end
    end # @testset "Morphology"

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

    # Test van Herk / Gil & Werman algorithm.
    @testset "van Herk / Gil & Werman algorithm" begin
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
=#

    @testset "Bilateral filter" begin
        A = randn(Float64, 128, 100)
        σr = 1.2
        σs = 2.5
        width = LocalFilters.BilateralFilter.default_width(σs)
        @test isodd(width)
        radius = (width - 1) ÷ 2
        ball = @inferred LocalFilters.ball(Dims{2}, radius)
        # FIXME: B4 = @inferred bilateralfilter(A, σr, σs, ball)
        @test axes(ball) == ntuple(_ -> -radius:radius, ndims(ball))
        box = fill!(similar(ball), one(eltype(ball)))
        B0 = @inferred bilateralfilter(A, σr, σs)
        # FIXME: B1 = @inferred bilateralfilter(A, σr, σs, box)
        B2 = @inferred bilateralfilter(Float32, A, σr, σs, width)
        # FIXME: B3 = @inferred bilateralfilter!(similar(Array{Float32}, axes(A)), A, σr, σs, box)
        # FIXME: @test similarvalues(B1, B0; atol=0, gtol=8*eps(Float32))
        @test similarvalues(B2, B0; atol=0, gtol=8*eps(Float32))
        # FIXME: @test similarvalues(B3, B0; atol=0, gtol=4*eps(Float32))
    end
end

end # module
