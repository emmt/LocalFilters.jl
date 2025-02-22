"""
    @public args...

declares `args...` as being `public` even though they are not exported. For Julia version
< 1.11, this macro does nothing. Using this macro also avoid errors with CI and coverage
tools.

"""
macro public(args::Union{Symbol,Expr}...)
    VERSION ≥ v"1.11.0-DEV.469" ? esc(Expr(:public, args...)) : nothing
end

if VERSION < v"1.7.0-beta1"
    # `Returns` is not defined prior to Julia 1.7.
    """
        f = Returns(val)

    yields a callable object `f` such that `f(args...; kwds...) === val` always
    holds. This is similar to `Returns` which appears in Julia 1.7.

    You may call:

        f = Returns{T}(val)

    to force the returned value to be of type `T`.

    """
    struct Returns{T}
        value::T
        Returns{T}(value) where {T} = new{T}(value)
        Returns(value::T) where {T} = new{T}(value)
    end
    (obj::Returns)(@nospecialize(args...); @nospecialize(kwds...)) = getfield(obj, :value)
    eval(:(export Returns))
end

if VERSION < v"1.6.0-beta1"
    # `reverse` for all dimensions is not defined prior to Julia 1.6.
    reverse(args...; kwds...) = Base.reverse(args...; kwds...)
    reverse(A::AbstractVector; kwds...) = Base.reverse(A; kwds...)
    function reverse(A::AbstractArray; dims = :)
        if !(dims isa Colon)
            return Base.reverse(A; dims=dims)
        elseif A isa AbstractUniformArray
            return Base.reverse(A)
        else
            B = Base.copymutable(A)
            I = eachindex(B)
            k = last(I) + first(I)
            for i in I
                j = k - i
                j > i || break
                @inbounds begin
                    Bi = B[i]
                    Bj = B[j]
                    B[i] = Bj
                    B[j] = Bi
                end
            end
            return B
        end
    end
end
