"""
    @public args...

declares `args...` as being `public` even though they are not exported. For Julia version
< 1.11, this macro does nothing. Using this macro also avoid errors with CI and coverage
tools.

"""
macro public(args::Union{Symbol,Expr}...)
    if VERSION ≥ v"1.11.0-DEV.469"
        esc(Expr(:public, args...))
    else
        nothing
    end
end
