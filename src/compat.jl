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
