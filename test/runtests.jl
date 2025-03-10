module TestingLocalFilters

using Test, Random, TypeUtils
using OffsetArrays, StructuredArrays
using LocalFilters
using LocalFilters:
    FilterOrdering, ForwardFilterOrdering, ReverseFilterOrdering,
    Box, box, ball, Indices, kernel_range, kernel, limits,
    is_morpho_math_box, check_axes, localindices,
    ranges, centered, centered_offset, unit_range,
    Returns, reverse, reverse!

zerofill!(A::AbstractArray) = fill!(A, zero(eltype(A)))

# `sum_init(T)` yields the initial value of the sum of elements of type `T`.
# The reasoning is that the sum of `n` identical values `x` is given by `n*x` (with
# `n` an `Int`).
sum_init(::Type{T}) where {T} = zero(T)*one(Int)

# `sumprod_init(A,B)` yields the initial value of the sum of products of elements of type
# `A` and `B`.
sumprod_init(::Type{A}, ::Type{B}) where {A,B} = (zero(A)*zero(B))*one(Int)
sumprod_update(v, a, b::Bool) = ifelse(b, v + a, v)
sumprod_update(v, a, b) = v + a*b
const sumprod_final = identity
sumprod_eltype(::Type{A}, ::Type{B}) where {A,B} = typeof(sumprod_init(A, B))

mean_init(::Type{A}, ::Type{B}) where {A,B} = (sumprod_init(A, B), sum_init(B))
mean_update(v, a, b::Bool) = ifelse(b, (v[1] + a, v[2] + one(v[2])), v)
mean_update(v, a, b) = (v[1] + a*b, v[2] + b)
mean_final(v) = (t = v[1]/v[2]; ifelse(iszero(v[2]), zero(t), t))

# The (local) mean is computed as (Σⱼ A[i±j]*B[j])/(Σⱼ B[j]) so the
# type of the result is that of `A` converted to floating-point.
mean_eltype(::Type{A}, ::Type{B}) where {A,B} = float(A)

erode_init(::Type{A}, ::Type{B}) where {A,B} = typemax(morphology_eltype(A, B))
erode_update(v, a, b::Bool) = ifelse(b, min(v, a), v)
erode_update(v, a, b) = min(v, a - b)
const erode_final = identity

dilate_init(::Type{A}, ::Type{B}) where {A,B} = typemin(morphology_eltype(A, B))
dilate_update(v, a, b::Bool) = ifelse(b, max(v, a), v)
dilate_update(v, a, b) = max(v, a + b)
const dilate_final = identity

# The result of a morphological operation for a source value `A` and a structuring element
# value `B` is of the type of `A` if `B` is Boolean and of the type of `A ± B` otherwise.
morphology_eltype(::Type{A}, ::Type{Bool}) where {A} = A
morphology_eltype(::Type{A}, ::Type{B}) where {A<:AbstractFloat,B<:AbstractFloat} =
    promote_type(A, B) # FIXME For small integers `±` may overflow...

function unsafe_sumprod_filter!(dst::AbstractArray{<:Any,N},
                                A::AbstractArray{<:Any,N},
                                ord::FilterOrdering,
                                B::AbstractArray{<:Any,N},
                                i, J) where {N}
    v = sumprod_init(eltype(A), eltype(B))
    @inbounds begin
        if eltype(B) <: Bool
            @simd for j in J
                a, b = A[j], B[ord(i,j)]
                v = ifelse(b, v + oftype(v, a), v)
            end
        else
            @simd for j in J
                a, b = A[j], B[ord(i,j)]
                v += oftype(v, a*b)
            end
        end
        dst[i] = v
    end
end

function unsafe_mean_filter!(dst::AbstractArray{<:Any,N},
                             A::AbstractArray{<:Any,N},
                             ord::FilterOrdering,
                             B::AbstractArray{<:Any,N},
                             i, J) where {N}
    num, den = mean_init(eltype(A), eltype(B))
    @inbounds begin
        if eltype(B) <: Bool
            @simd for j in J
                a, b = A[j], B[ord(i,j)]
                num = ifelse(b, num + oftype(num, a), num)
                den += oftype(den, b)
            end
        else
            @simd for j in J
                a, b = A[j], B[ord(i,j)]
                num += oftype(num, a*b)
                den += oftype(den, b)
            end
        end
        dst[i] = iszero(den) ? zero(eltype(dst)) : convert(eltype(dst), num/den)
    end
end

function unsafe_erode_filter!(dst::AbstractArray{<:Any,N},
                              A::AbstractArray{<:Any,N},
                              ord::FilterOrdering,
                              B::AbstractArray{<:Any,N},
                              i, J) where {N}
    v = erode_init(eltype(A), eltype(B))
    @inbounds begin
        if eltype(B) <: Bool
            @simd for j in J
                v = ifelse(B[ord(i,j)], min(v, A[j]), v)
            end
        else
            @simd for j in J
                v = min(v, A[j] - B[ord(i,j)])
            end
        end
        dst[i] = v
    end
end

function unsafe_dilate_filter!(dst::AbstractArray{<:Any,N},
                               A::AbstractArray{<:Any,N},
                               ord::FilterOrdering,
                               B::AbstractArray{<:Any,N},
                               i, J) where {N}
    v = dilate_init(eltype(A), eltype(B))
    @inbounds begin
        if eltype(B) <: Bool
            @simd for j in J
                v = ifelse(B[ord(i,j)], max(v, A[j]), v)
            end
        else
            @simd for j in J
                v = max(v, A[j] + B[ord(i,j)])
            end
        end
        dst[i] = v
    end
end

const KernelAxis = Union{Integer,AbstractUnitRange{<:Integer}}

for f in (:erode, :dilate, :sumprod, :mean)
    t = f === :sumprod ? :sumprod_eltype :
        f === :mean    ? :mean_eltype : :morphology_eltype
    @eval begin
        function $(Symbol("$(f)_ref"))(A::AbstractArray{<:Any,N},
                                       B::Union{KernelAxis,NTuple{N,KernelAxis}};
                                       kwds...) where {N}
            return $(Symbol("$(f)_ref"))(A, kernel(Dims{N}, B); kwds...)
        end
        function $(Symbol("$(f)_ref"))(A::AbstractArray{<:Any,N},
                                       B::AbstractArray{<:Any,N}; kwds...) where {N}
            T = $t(eltype(A), eltype(B))
            return $(Symbol("$(f)_ref!"))(similar(A, T), A, B; kwds...)
        end
        function $(Symbol("$(f)_ref!"))(dst::AbstractArray{<:Any,N},
                                        A::AbstractArray{<:Any,N},
                                        B::AbstractArray{<:Any,N}; kwds...) where {N}
            return filter_ref!(dst, A, B,
                               $(Symbol("$(f)_init"))(eltype(A), eltype(B)),
                               $(Symbol("$(f)_update")),
                               $(Symbol("$(f)_final")); kwds...)
        end
    end
end

function filter_ref!(dst::AbstractArray{<:Any,N},
                     A::AbstractArray{<:Any,N},
                     B::AbstractArray{<:Any,N},
                     init, update::Function, final::Function = identity;
                     order::FilterOrdering = FORWARD_FILTER) where {N}
    !(init isa Function) || axes(A) == axes(dst) || error(
        "source and destination have different axes")
    I = CartesianIndices(dst)
    J = CartesianIndices(B)
    @inbounds for i in I
        v = init isa Function ? init(A[i]) : init
        for j in J
            k = order isa ForwardFilterOrdering ? i + j : i - j
            if k ∈ I
                v = update(v, A[k], B[j])
            end
        end
        dst[i] = final(v)
    end
    return dst
end

@testset "LocalFilters" begin

    @testset "Utilities" begin

        @testset "Returns" begin
            @test (1,1,1) === @inferred ntuple(Returns(1), Val(3))
            @test (0x1,0x1,0x1) === @inferred ntuple(Returns{UInt8}(1), Val(3))
        end

        @testset "`reverse` and `reverse!`" begin
            @testset "... basic use" begin
                @test reverse("abcd") == "dcba"
                r = 1:8
                v = Array{Int}(undef, length(r))
                @test reverse(r, 2) == [1,8,7,6,5,4,3,2]
                @test reverse(r, 3, 6) == [1,2,6,5,4,3,7,8]
                @test reverse(r; dims=:) === 8:-1:1
                @test reverse(r; dims=1) === 8:-1:1
                @test reverse(copyto!(v, r); dims=:) == 8:-1:1
                @test_throws ArgumentError reverse(r; dims=0)
                @test v === reverse!(copyto!(v, r), 2) && v == [1,8,7,6,5,4,3,2]
                @test v === reverse!(copyto!(v, r), 3, 6) && v == [1,2,6,5,4,3,7,8]
                @test v === reverse!(copyto!(v, r); dims=:) && v == [8,7,6,5,4,3,2,1]
                @test_throws ArgumentError reverse!(v; dims=0)
            end
            @testset "... with `dims=$d`" for d in (Colon(), 1, 2)
                if d isa Colon || d == 1
                    # Check for vectors.
                    A = 1:12
                    R = 12:-1:1
                    B = @inferred reverse(A; dims=d)
                    @test B == R
                    @test B isa AbstractRange{eltype(A)}
                    @test step(B) == -step(A)
                    @test last(B) == first(A)
                    @test first(B) == last(A)
                    C = Array{eltype(A)}(undef, size(A))
                    @test C === @inferred reverse!(copyto!(C, A); dims=d)
                    @test C == R
                    if d isa Colon
                        @test R == @inferred reverse(A)
                        @test C === @inferred reverse!(copyto!(C, A))
                        @test C == R
                    end
                    A = -2:3:7
                    R = 7:-3:-2
                    B = @inferred reverse(A; dims=d)
                    @test B == R
                    @test B isa AbstractRange{eltype(A)}
                    @test step(B) == -step(A)
                    @test last(B) == first(A)
                    @test first(B) == last(A)
                    C = Array{eltype(A)}(undef, size(A))
                    @test C === @inferred reverse!(copyto!(C, A); dims=d)
                    @test C == R
                    if d isa Colon
                        @test R == @inferred reverse(A)
                        @test C === @inferred reverse!(copyto!(C, A))
                        @test C == R
                    end
                end
                if d isa Colon || d ≤ 2
                    # Check for 2-dimensional arrays.
                    A = reshape(1:12, 3,4)
                    R = d isa Colon ? reshape(12:-1:1, size(A)) : Base.reverse(A; dims=d)
                    C = Array{eltype(A)}(undef, size(A))
                    @test R == @inferred reverse(A; dims=d)
                    @test C === @inferred reverse!(copyto!(C, A); dims=d)
                    @test C == R
                    if d isa Colon
                        @test R == @inferred reverse(A)
                        @test C === @inferred reverse!(copyto!(C, A))
                        @test C == R
                    end
                    A = reshape(1:12, 4,3)
                    C = Array{eltype(A)}(undef, size(A))
                    R = d isa Colon ? reshape(12:-1:1, size(A)) : Base.reverse(A; dims=d)
                    @test R == @inferred reverse(A; dims=d)
                    @test C === @inferred reverse!(copyto!(C, A); dims=d)
                    @test C == R
                    if d isa Colon
                        @test R == @inferred reverse(A)
                        @test C === @inferred reverse!(copyto!(C, A))
                        @test C == R
                    end
                end
                if d isa Colon || d ≤ 3
                    # Check for 3-dimensional arrays.
                    A = reshape(-4:5:111, 2,3,4)
                    R = d isa Colon ? reshape(111:-5:-4, size(A)) : Base.reverse(A; dims=d)
                    C = Array{eltype(A)}(undef, size(A))
                    @test R == @inferred reverse(A; dims=d)
                    @test C === @inferred reverse!(copyto!(C, A); dims=d)
                    @test C == R
                    if d isa Colon
                        @test R == @inferred reverse(A)
                        @test C === @inferred reverse!(copyto!(C, A))
                        @test C == R
                    end
                end
            end
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
            @test ndims(A) == 0
            B = A
            @test B === @inferred reverse_kernel(A)
            @test B === @inferred reverse_kernel(Dims{ndims(A)}, A)
            @test B === @inferred reverse_kernel()
            @test B === @inferred reverse_kernel(Dims{ndims(A)})
            @test B === @inferred reverse_kernel(())
            @test B === @inferred reverse_kernel(Dims{ndims(A)}, ())
            A = box(-3:2, -3:2, -3:2)
            B = box(-2:3, -2:3, -2:3)
            @test B === @inferred reverse_kernel(A)
            @test B === @inferred reverse_kernel(Dims{ndims(A)}, A)
            @test A === @inferred reverse_kernel(B)
            @test A === @inferred reverse_kernel(Dims{ndims(B)}, B)
            @test B === @inferred reverse_kernel(6, 6, 6)
            @test B === @inferred reverse_kernel(Int16(6), -3:2, Int8(-3):Int8(2))
            @test B === @inferred reverse_kernel(Dims{3}, 6, 6, 6)
            @test B === @inferred reverse_kernel(Dims{3}, Int8(6))
            A = box(-2:2, -3:2, -3:3)
            B = box(-2:2, -2:3, -3:3)
            @test B === @inferred reverse_kernel(A)
            @test B === @inferred reverse_kernel(Dims{ndims(A)}, A)
            @test A === @inferred reverse_kernel(B)
            @test A === @inferred reverse_kernel(Dims{ndims(B)}, B)
            @test B === @inferred reverse_kernel(5, 6, 7)
            @test B === @inferred reverse_kernel(Dims{3}, 5, 6, 7)
            A = box(-2:4, -3:2)
            B = box(-4:2, -2:3)
            @test B === @inferred reverse_kernel(A)
            @test B === @inferred reverse_kernel(Dims{ndims(A)}, A)
            @test A === @inferred reverse_kernel(B)
            @test A === @inferred reverse_kernel(Dims{ndims(B)}, B)
            @test B === @inferred reverse_kernel(-2:4, 6)
            @test B === @inferred reverse_kernel(-2:4, -3:2)
            @test B === @inferred reverse_kernel(Dims{2}, -2:4, 6)
            @test B === @inferred reverse_kernel(Dims{2}, -2:4, -3:2)
            A = box(-3:2, 1:9, 0:5)
            B = box(-2:3, -9:-1, -5:0)
            @test B === @inferred reverse_kernel(A)
            @test B === @inferred reverse_kernel(Dims{ndims(A)}, A)
            @test A === @inferred reverse_kernel(B)
            @test A === @inferred reverse_kernel(Dims{ndims(B)}, B)
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
            C = @inferred centered(B)
            @test eltype(C) === eltype(B)
            @test ndims(C) === ndims(B)
            @test size(C) === size(B)
            @test axes(C) == (-2:1, -2:2)
            @test collect(C) == collect(B)
            @test C === @inferred centered(C) # `centered` is idempotent
            B = UniformArray(true, 4,2:7,11)
            C = @inferred centered(B)
            @test eltype(C) === eltype(B)
            @test ndims(C) === ndims(B)
            @test size(C) === size(B)
            @test axes(C) == (-2:1, -3:2, -5:5)
            @test C isa AbstractUniformArray
            @test first(C) === first(B)
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
            @test A === @inferred strel(Bool, axes(A))
            @test A === @inferred strel(Bool, axes(A)...)
            @test A === @inferred strel(Bool, CartesianIndices(A))
            @test B === @inferred strel(T, axes(A))
            @test B === @inferred strel(T, axes(A)...)
            @test B === @inferred strel(T, CartesianIndices(A))
            A = @inferred box(2, -1:2)
            B = FastUniformArray(zero(T), axes(A))
            @test A === @inferred strel(Bool, A)
            @test A === @inferred strel(Bool, axes(A))
            @test A === @inferred strel(Bool, axes(A)...)
            @test A === @inferred strel(Bool, CartesianIndices(A))
            @test B === @inferred strel(T, axes(A))
            @test B === @inferred strel(T, axes(A)...)
            @test B === @inferred strel(T, CartesianIndices(A))
            A = @inferred centered(ones(Bool, 3, 4, 5))
            S = @inferred box(axes(A))
            B = FastUniformArray(zero(T), axes(A))
            @test A === @inferred strel(Bool, A)
            @test A ==  @inferred strel(Bool, axes(A))
            @test A ==  @inferred strel(Bool, axes(A)...)
            @test A ==  @inferred strel(Bool, CartesianIndices(A))
            @test S === @inferred strel(Bool, CartesianIndices(A))
            @test B === @inferred strel(T, axes(A))
            @test B === @inferred strel(T, axes(A)...)
            @test B === @inferred strel(T, CartesianIndices(A))
            A = @inferred ball(Dims{3}, 3.5)
            B = map(x -> x ? zero(T) : -T(Inf), A)
            @test A === @inferred strel(Bool, A)
            @test B ==  @inferred strel(T, A)
        end

    end # @testset "Utilities"

    rng = MersenneTwister(31415)
    @testset "Linear filters (dims = $dims)" for dims in ((25,), (13,20), (17,12,8))
        # Pseudo-image, forward and reverse kernels.
        # NOTE Use random integers in a small range for exact results.
        Aint = rand(rng, -3:20, dims)
        Kint = @inferred centered(rand(rng, -1:8, map(x -> round(Int, x/3), dims)))

        @testset "... and \"$name\" kernel" for name in (:integer, :float, :ones, :box)
            # Convert source and build kernels.
            if name === :integer
                A = Aint
                Kf = Kint
                Kr = reverse_kernel(Kf)
            elseif name === :float
                # Convert to floating-point and make sure kernels have no negative values.
                T = Float64
                A = T.(Aint)
                Kf = T.(Kint .- minimum(Kint))
                Kr = reverse_kernel(Kf)
            elseif name === :ones
                # Use arrays of ones as kernels.
                T = Float64
                A = T.(Aint)
                Kf = centered(ones(T, size(Kint)))
                Kr = reverse_kernel(Kf)
            elseif name === :box
                # Will use simple boxes as kernels.
                T = Float64
                A = T.(Aint)
                Kf = axes(Kint)
                Kr = axes(reverse_kernel(Kint))
            end

            # Copy of source for checking that it is left untouched.
            C = copy(A)

            # Reference results.
            B0f = @inferred sumprod_ref(A, Kf; order=FORWARD_FILTER)
            @test C == A  # check that A is left unchanged
            B0r = @inferred sumprod_ref(A, Kf; order=REVERSE_FILTER)
            @test C == A  # check that A is left unchanged

            # Check reverse/forward consistency.
            @test typeof(B0f) === typeof(B0r)
            @test B0f == @inferred sumprod_ref(A, Kr; order=REVERSE_FILTER)
            @test C == A  # check that A is left unchanged
            @test B0r == @inferred sumprod_ref(A, Kr; order=FORWARD_FILTER)
            @test C == A  # check that A is left unchanged

            # Workspace for in-place operations.
            B1 = similar(B0f)
            T = eltype(B0f)

            # Test `correlate`.
            @test B0f == @inferred correlate(A, Kf)
            @test C == A  # check that A is left unchanged
            @test B0r == @inferred correlate(A, Kr)
            @test C == A  # check that A is left unchanged

            # Test `correlate!`.
            @test B1 === @inferred correlate!(zerofill!(B1), A, Kf)
            @test C == A   # check that A is left unchanged
            @test B0f == B1 # check result
            @test B1 === @inferred correlate!(zerofill!(B1), A, Kr)
            @test C == A   # check that A is left unchanged
            @test B0r == B1 # check result

            # Test `convolve`.
            @test B0r == @inferred convolve(A, Kf)
            @test C == A  # check that A is left unchanged
            @test B0f == @inferred convolve(A, Kr)
            @test C == A  # check that A is left unchanged

            # Test `convolve!`.
            @test B1 === @inferred convolve!(zerofill!(B1), A, Kf)
            @test C == A   # check that A is left unchanged
            @test B0r == B1 # check result
            @test B1 === @inferred convolve!(zerofill!(B1), A, Kr)
            @test C == A   # check that A is left unchanged
            @test B0f == B1 # check result

            # Test `localfilter`.
            @test B0f == @inferred localfilter(T, A, Kf, unsafe_sumprod_filter!)
            @test C == A   # check that A is left unchanged
            @test B0r == @inferred localfilter(T, A, Kr, unsafe_sumprod_filter!)
            @test C == A   # check that A is left unchanged
            @test B0r == @inferred localfilter(T, A, Kf, unsafe_sumprod_filter!; order=REVERSE_FILTER)
            @test C == A   # check that A is left unchanged
            @test B0f == @inferred localfilter(T, A, Kr, unsafe_sumprod_filter!; order=REVERSE_FILTER)
            @test C == A   # check that A is left unchanged

            # Test `localfilter!`.
            @test B1 === @inferred localfilter!(zerofill!(B1), A, Kf, unsafe_sumprod_filter!)
            @test C == A   # check that A is left unchanged
            @test B0f == B1 # check result
            @test B1 === @inferred localfilter!(zerofill!(B1), A, Kr, unsafe_sumprod_filter!)
            @test C == A   # check that A is left unchanged
            @test B0r == B1 # check result
            @test B1 === @inferred localfilter!(zerofill!(B1), A, Kf, unsafe_sumprod_filter!; order=REVERSE_FILTER)
            @test C == A   # check that A is left unchanged
            @test B0r == B1 # check result
            @test B1 === @inferred localfilter!(zerofill!(B1), A, Kr, unsafe_sumprod_filter!; order=REVERSE_FILTER)
            @test C == A   # check that A is left unchanged
            @test B0f == B1 # check result

            # Reference results.
            B0f = @inferred mean_ref(A, Kf; order=FORWARD_FILTER)
            @test C == A  # check that A is left unchanged
            B0r = @inferred mean_ref(A, Kf; order=REVERSE_FILTER)
            @test C == A  # check that A is left unchanged

            # Check reverse/forward consistency.
            @test typeof(B0f) === typeof(B0r)
            @test B0f == @inferred mean_ref(A, Kr; order=REVERSE_FILTER)
            @test C == A  # check that A is left unchanged
            @test B0r == @inferred mean_ref(A, Kr; order=FORWARD_FILTER)
            @test C == A  # check that A is left unchanged

            # Workspace for in-place operations.
            B1 = similar(B0f)
            T = eltype(B0f)

            # Test `localmean`.
            @test B0f == @inferred localmean(A, Kf)
            @test C == A # check that A is left unchanged
            @test B0r == @inferred localmean(A, Kr)
            @test C == A # check that A is left unchanged
            @test B0f == @inferred localmean(A, Kf; order=FORWARD_FILTER)
            @test C == A # check that A is left unchanged
            @test B0r == @inferred localmean(A, Kr; order=FORWARD_FILTER)
            @test C == A # check that A is left unchanged
            @test B0r == @inferred localmean(A, Kf; order=REVERSE_FILTER)
            @test C == A # check that A is left unchanged
            @test B0f == @inferred localmean(A, Kr; order=REVERSE_FILTER)
            @test C == A # check that A is left unchanged

            # Test `localmean!`.
            @test B1 === @inferred localmean!(zerofill!(B1), A, Kf)
            @test C == A # check that A is left unchanged
            @test B0f == B1
            @test B1 === @inferred localmean!(zerofill!(B1), A, Kr)
            @test C == A # check that A is left unchanged
            @test B0r == B1
            @test B1 === @inferred localmean!(zerofill!(B1), A, Kf; order=FORWARD_FILTER)
            @test C == A # check that A is left unchanged
            @test B0f == B1
            @test B1 === @inferred localmean!(zerofill!(B1), A, Kr; order=FORWARD_FILTER)
            @test C == A # check that A is left unchanged
            @test B0r == B1
            @test B1 === @inferred localmean!(zerofill!(B1), A, Kf; order=REVERSE_FILTER)
            @test C == A # check that A is left unchanged
            @test B0r == B1
            @test B1 === @inferred localmean!(zerofill!(B1), A, Kr; order=REVERSE_FILTER)
            @test C == A # check that A is left unchanged
            @test B0f == B1

            # Test `localfilter`.
            @test B0f == @inferred localfilter(T, A, Kf, unsafe_mean_filter!)
            @test C == A # check that A is left unchanged
            @test B0r == @inferred localfilter(T, A, Kr, unsafe_mean_filter!)
            @test C == A # check that A is left unchanged
            @test B0f == @inferred localfilter(T, A, Kf, unsafe_mean_filter!; order=FORWARD_FILTER)
            @test C == A # check that A is left unchanged
            @test B0r == @inferred localfilter(T, A, Kr, unsafe_mean_filter!; order=FORWARD_FILTER)
            @test C == A # check that A is left unchanged
            @test B0r == @inferred localfilter(T, A, Kf, unsafe_mean_filter!; order=REVERSE_FILTER)
            @test C == A # check that A is left unchanged
            @test B0f == @inferred localfilter(T, A, Kr, unsafe_mean_filter!; order=REVERSE_FILTER)
            @test C == A # check that A is left unchanged

            # Test `localfilter!`.
            @test B1 === @inferred localfilter!(zerofill!(B1), A, Kf, unsafe_mean_filter!)
            @test C == A # check that A is left unchanged
            @test B0f == B1
            @test B1 === @inferred localfilter!(zerofill!(B1), A, Kr, unsafe_mean_filter!)
            @test C == A # check that A is left unchanged
            @test B0r == B1
            @test B1 === @inferred localfilter!(zerofill!(B1), A, Kf, unsafe_mean_filter!; order=FORWARD_FILTER)
            @test C == A # check that A is left unchanged
            @test B0f == B1
            @test B1 === @inferred localfilter!(zerofill!(B1), A, Kr, unsafe_mean_filter!; order=FORWARD_FILTER)
            @test C == A # check that A is left unchanged
            @test B0r == B1
            @test B1 === @inferred localfilter!(zerofill!(B1), A, Kf, unsafe_mean_filter!; order=REVERSE_FILTER)
            @test C == A # check that A is left unchanged
            @test B0r == B1
            @test B1 === @inferred localfilter!(zerofill!(B1), A, Kr, unsafe_mean_filter!; order=REVERSE_FILTER)
            @test C == A # check that A is left unchanged
            @test B0f == B1

        end

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

    @testset "Morphology (T = $T, $name)" for (T, dims) in ((UInt8, (12, 13)), (Float32, (4,5,6))),
        (name, W) in (("box(Dims{N},5)", 5), ("box(Dims{N},-1:2)", -1:2),
                      ("ball", 2.5), ("box(...)", (-2:2, -1:2, -1:1)))
        A = rand(T, dims); # source
        wrk = similar(A);  # workspace
        B1 = similar(A);   # for in-place operation
        B2 = similar(A);   # for in-place operation
        C = copy(A);       # to check that the source is left unchanged
        N = ndims(A);
        R = if W isa AbstractFloat
            @inferred ball(Dims{N}, W)
        elseif W isa Tuple
            W[1:N]
        else
            W
        end
        K = @inferred kernel(Dims{N}, R)
        @testset "$name" for (name, func, func!, ref_func, filter!) in (("Erosion", erode, erode!, erode_ref, unsafe_erode_filter!),
                                                                        ("Dilation", dilate, dilate!, dilate_ref, unsafe_dilate_filter!))
            B0 = ref_func(A, R)
            @test B0 == @inferred func(A, R; slow=true);
            @test C == A   # check that A is left unchanged
            @test B1 === @inferred func!(zerofill!(B1), A, R; slow=true)
            @test C == A   # check that A is left unchanged
            @test B0 == B1 # check if in-place and out-of-place yield the same result
            @test B0 == @inferred localfilter(A, R, filter!)
            @test C == A   # check that A is left unchanged
            @test B1 === @inferred localfilter!(zerofill!(B1), A, R, filter!)
            @test C == A   # check that A is left unchanged
            @test B0 == B1 # check result
            @test B0 == @inferred func(A, R; slow=false);
            @test C == A   # check that A is left unchanged
            @test B1 === @inferred func!(zerofill!(B1), A, R; slow=false)
            @test C == A   # check that A is left unchanged
            @test B0 == B1 # check if in-place and out-of-place yield the same result
            # FIXME: @test B2 === @inferred func!(copyto!(B2, A), R)
            # FIXME: @test B2 == B0 # check if in-place and out-of-place yield the same result
            if T <: AbstractFloat
                S = @inferred strel(T, K) # flat structuring element like K
                @test B0 == @inferred func(A, S)
                @test C == A   # check that A is left unchanged
                @test B1 === @inferred func!(zerofill!(B1), A, S)
                @test C == A   # check that A is left unchanged
                @test B0 == B1 # check if in-place and out-of-place yield the same result
                @test B0 == @inferred localfilter(A, S, filter!)
                @test C == A   # check that A is left unchanged
                @test B1 === @inferred localfilter!(zerofill!(B1), A, S, filter!)
                @test C == A   # check that A is left unchanged
                @test B0 == B1 # check result
            end
        end
        @testset "Local min. and max." begin
            B01 = @inferred erode(A, R)  # `erode` also yields local min.
            B02 = @inferred dilate(A, R) # `dilate` also yields local max.
            @test (B01, B02) == @inferred localextrema(A, R);
            @test C == A   # check that A is left unchanged
            @test (B1, B2) === @inferred localextrema!(zerofill!(B1), zerofill!(B2), A, R)
            @test C == A   # check that A is left unchanged
            @test B01 == B1
            @test B02 == B2
            if T <: AbstractFloat
                S = @inferred strel(T, K) # flat structuring element like K
                @test (B01, B02) == @inferred localextrema(A, S)
                @test C == A   # check that A is left unchanged
                @test (B1, B2) === @inferred localextrema!(zerofill!(B1), zerofill!(B2), A, S)
                @test C == A   # check that A is left unchanged
                @test B01 == B1
                @test B02 == B2
            end
        end
        @testset "$name" for (name, func, func!) in (("Opening", opening, opening!),
                                                     ("Closing", closing, closing!))
            B0 = if func === opening
                @inferred dilate(erode(A, R), R) # opening is erosion followed by dilation
            else
                @inferred erode(dilate(A, R), R) # closing is dilation followed by erosion
            end
            @test B0 == @inferred func(A, R; slow=true);
            @test C == A   # check that A is left unchanged
            @test B1 === @inferred func!(zerofill!(B1), wrk, A, R; slow=true)
            @test C == A   # check that A is left unchanged
            @test B0 == B1 # check if in-place and out-of-place yield the same result
            @test B0 == @inferred func(A, R; slow=false);
            @test C == A   # check that A is left unchanged
            @test B1 === @inferred func!(zerofill!(B1), wrk, A, R; slow=false)
            @test C == A   # check that A is left unchanged
            @test B0 == B1 # check if in-place and out-of-place yield the same result
            if T <: AbstractFloat
                S = @inferred strel(T, K) # flat structuring element like K
                @test B0 == @inferred func(A, S)
                @test C == A   # check that A is left unchanged
                @test B1 === @inferred func!(zerofill!(B1), wrk, A, S)
                @test C == A   # check that A is left unchanged
                @test B0 == B1 # check if in-place and out-of-place yield the same result
            end
        end
        @testset "$name" for (name, func, func!) in (("Top-hat", top_hat, top_hat!),
                                                     ("Bottom-hat", bottom_hat, bottom_hat!))
            B0 = if func === top_hat
                A .- opening(A, R) # definition of top-hat
            else
                closing(A, R) .- A # definition of bottom-hat
            end
            @test B0 == @inferred func(A, R; slow=true);
            @test C == A   # check that A is left unchanged
            @test B1 === @inferred func!(zerofill!(B1), wrk, A, R; slow=true)
            @test C == A   # check that A is left unchanged
            @test B0 == B1 # check that in-place and out-of-place yield the same result
            @test B0 == @inferred func(A, R; slow=false);
            @test C == A   # check that A is left unchanged
            @test B1 === @inferred func!(zerofill!(B1), wrk, A, R; slow=false)
            @test C == A   # check that A is left unchanged
            @test B0 == B1 # check that in-place and out-of-place yield the same result
            if T <: AbstractFloat
                S = @inferred strel(T, K) # flat structuring element like K
                @test B0 == @inferred func(A, S)
                @test C == A   # check that A is left unchanged
                @test B1 === @inferred func!(zerofill!(B1), wrk, A, S)
                @test C == A   # check that A is left unchanged
                @test B0 == B1 # check if in-place and out-of-place yield the same result
            end
        end
    end # @testset "Morphology"

    @testset "`localmap` and `localmap!` ($win, $func)" for win in (:box, :ball, :win),
        func in (:min, :max)
        T = Float32
        dims = (30, 20)
        A = rand(T, dims)
        Aref = copy(A)
        W = win === :box ? 3 :
            win === :ball ? ball(Dims{ndims(A)}, 2.5) :
            centered(rand(Bool, ntuple(Returns(6), ndims(A))))
        f, f_ref = func === :minimum ? (minimum, erode) : (maximum, dilate)
        Bref = @inferred f_ref(A, W)
        B = @inferred localmap(f, A, W)
        @test A == Aref # check that A is left unchanged
        @test B == Bref # check result
        @test B === @inferred localmap!(f, zerofill!(B), A, W)
        @test A == Aref # check that A is left unchanged
        @test B == Bref # check result
        if win === :win
            Bref = @inferred f_ref(A, W; order=REVERSE_FILTER)
            B = @inferred localmap(f, A, W; order=REVERSE_FILTER)
            @test A == Aref # check that A is left unchanged
            @test B == Bref # check result
            @test B === @inferred localmap!(f, zerofill!(B), A, W; order=REVERSE_FILTER)
            @test A == Aref # check that A is left unchanged
            @test B == Bref # check result
        end
    end

    @testset "van Herk / Gil & Werman algorithm" begin
        A = rand(Float32, 40)
        Aref = copy(A)
        Bref = similar(A)
        n = length(A)
        @testset "... shift by $k" for k in (2, 0, -3)
            for i in 1:n
                j = clamp(i + k, 1, n)
                Bref[i] = A[j]
            end
            B = @inferred localfilter(A, 1, min, k:k)
            @test A == Aref # check that A is left unchanged
            @test B == Bref # check result
            @test B === @inferred localfilter!(copyto!(B, A), 1, min, k:k)
            @test B == Bref # check result
        end
        A = rand(Float64, 12, 13)
        Aref = copy(A)
        Bref = similar(A)
        n1, n2 = size(A)
        @testset "... shift by ($k1, $k2)" for k1 in (2, 0, -3), k2 in (1, 0, -2)
            for i2 in 1:n2
                j2 = clamp(i2 + k2, 1, n2)
                for i1 in 1:n1
                    j1 = clamp(i1 + k1, 1, n1)
                    Bref[i1,i2] = A[j1,j2]
                end
            end
            B = @inferred localfilter(A, :, min, (k1:k1, k2:k2))
            @test A == Aref # check that A is left unchanged
            @test B == Bref # check result
            @test B === @inferred localfilter!(copyto!(B, A), :, min, (k1:k1, k2:k2))
            @test B == Bref # check result
            B = @inferred localfilter(A, [1,2], min, (k1:k1, k2:k2))
            @test A == Aref # check that A is left unchanged
            @test B == Bref # check result
            @test B === @inferred localfilter!(copyto!(B, A), [1,2], min, (k1:k1, k2:k2))
            @test B == Bref # check result
            B = @inferred localfilter(A, (2,1), min, [k2:k2, k1:k1])
            @test A == Aref # check that A is left unchanged
            @test B == Bref # check result
            @test B === @inferred localfilter!(copyto!(B, A), (2,1), min, (k2:k2, k1:k1))
            @test B == Bref # check result
            B = @inferred localfilter(A, [1,2], min, (k1:k1, k2:k2))
            @test A == Aref # check that A is left unchanged
            @test B == Bref # check result
            B = @inferred localfilter(A, (2,1), min, [k2:k2, k1:k1])
            @test A == Aref # check that A is left unchanged
            @test B == Bref # check result
            if iszero(k2)
                B = @inferred localfilter(A, 1, min, k1:k1)
                @test A == Aref # check that A is left unchanged
                @test B == Bref # check result
                B = @inferred localfilter(A, [1], min, (k1:k1,))
                @test A == Aref # check that A is left unchanged
                @test B == Bref # check result
                B = @inferred localfilter(A, (1,), min, [k1:k1])
                @test A == Aref # check that A is left unchanged
                @test B == Bref # check result
            end
            if iszero(k1)
                B = @inferred localfilter(A, 2, min, k2:k2)
                @test A == Aref # check that A is left unchanged
                @test B == Bref # check result
                B = @inferred localfilter(A, [2], min, (k2:k2,))
                @test A == Aref # check that A is left unchanged
                @test B == Bref # check result
                B = @inferred localfilter(A, (2,), min, [k2:k2])
                @test A == Aref # check that A is left unchanged
                @test B == Bref # check result
            end
            #@test samevalues(B, localfilter(A,[1,2],min,(k1:k1,k2:k2)))
            #@test samevalues(B, localfilter(A,(2,1),min,(k2:k2,k1:k1)))
            #@test samevalues(B, localfilter!(copyto!(C,A),:,min,(k1:k1,k2:k2)))
            #@test samevalues(B, localfilter!(copyto!(C,A),(1,2),min,(k1:k1,k2:k2)))
            #@test samevalues(B, localfilter!(copyto!(C,A),[2,1],min,(k2:k2,k1:k1)))
        end
    end

    @testset "Bilateral filter" begin
        A = randn(Float64, 128, 100)
        σr = 1.2
        σs = 2.5
        width = LocalFilters.BilateralFilter.default_width(σs)
        ball = @inferred LocalFilters.ball(Dims{2}, width/2)
        @test isodd(width)
        radius = (width - 1)÷2
        @test axes(ball) == ntuple(Returns(-radius:radius), ndims(ball))
        B4 = @inferred bilateralfilter(A, σr, σs, ball)
        box = fill!(similar(ball), one(eltype(ball)))
        B0 = @inferred bilateralfilter(A, σr, σs)
        B1 = @inferred bilateralfilter(A, σr, σs, box)
        @test maximum(abs.(B0 - B1)) ≤ 1e-12*maximum(abs.(B0))
        B2 = @inferred bilateralfilter(Float32, A, σr, σs, width)
        @test maximum(abs.(B0 - B2)) ≤ 1e-6*maximum(abs.(B0))
        B3 = @inferred bilateralfilter!(similar(Array{Float32}, axes(A)), A, σr, σs, box)
        @test maximum(abs.(B0 - B3)) ≤ 1e-6*maximum(abs.(B0))
    end
end

end # module
