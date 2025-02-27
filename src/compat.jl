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
end

if VERSION < v"1.6.0-beta1"
    # Prior to Julia 1.6, `reverse` for all dimensions is not defined and `reverse!` does
    # not accept keywords. NOTE Helper functions `_reverse` and `_reverse!` are introduced
    # for inference to work.
    reverse(args...; kwds...) = Base.reverse(args...; kwds...)
    reverse!(args...; kwds...) = Base.reverse!(args...; kwds...)
    reverse(A::AbstractArray; dims = :) = _reverse(A, dims)
    reverse!(A::AbstractArray; dims = :) = _reverse!(A, dims)
    _reverse(A::AbstractVector, d::Integer) =
        isone(d) ? Base.reverse(A) : throw(ArgumentError("invalid dimension $d ≠ 1"))
    _reverse!(A::AbstractVector, d::Integer) =
        isone(d) ? Base.reverse!(A) : throw(ArgumentError("invalid dimension $d ≠ 1"))
    _reverse(A::AbstractArray, d) = Base.reverse(A; dims=d)
    _reverse!(A::AbstractArray, d) = copyto!(A, Base.reverse(A; dims=d))
    _reverse(A::AbstractVector, ::Colon) = Base.reverse(A)
    _reverse!(A::AbstractVector, ::Colon) = Base.reverse!(A)
    _reverse(A::AbstractArray, ::Colon) = _reverse!(Base.copymutable(A), :)
    function _reverse!(A::AbstractArray, ::Colon)
        I = eachindex(A)
        k = last(I) + first(I)
        @inbounds for i in I
            (j = k - i) > i || break
            Ai = A[i]
            Aj = A[j]
            A[i] = Aj
            A[j] = Ai
        end
        return A
    end
end

if VERSION < v"1.11.0-alpha1"
    # Wrappers to get rid of newly introduced keywords:
    for type in (:BitSet, :Vector)
        @eval sizehint!(x::$type, sz::Integer; shrink::Bool=true, first::Bool=false) =
            Base.sizehint!(x, sz)
    end
    for type in (:Dict, :Set, :WeakKeyDict)
        @eval sizehint!(x::$type, sz; shrink::Bool=true) = Base.sizehint!(x, sz)
    end
    # Fallback only called when `shrink` keyword explicitly specified:
    sizehint!(x, sz; shrink::Bool) = Base.sizehint!(x, sz)
    # Fallback for any other calls:
    sizehint!(args...; kwds...) = Base.sizehint!(args...; kwds...)
end
