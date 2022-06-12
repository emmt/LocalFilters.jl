using Documenter

push!(LOAD_PATH, "../src/")
using LocalFilters

DEPLOYDOCS = (get(ENV, "CI", nothing) == "true")

makedocs(
    sitename = "LocalFilters.jl Package",
    format = Documenter.HTML(
        prettyurls = DEPLOYDOCS,
    ),
    authors = "Éric Thiébaut and contributors",
    pages = ["index.md", "generic.md", "neighborhoods.md", "linear.md",
             "nonlinear.md", "morphology.md", "separable.md", "reference.md"]
)

if DEPLOYDOCS
    deploydocs(
        repo = "github.com/emmt/LocalFilters.jl.git",
    )
end
