SOURCES = \
    make.jl \
    src/generic.md \
    src/index.md \
    src/linear.md \
    src/math.md \
    src/morphology.md \
    src/neighborhoods.md \
    src/nonlinear.md \
    src/reference.md \
    src/separable.md

MAKEDOC = julia --project --color=yes make.jl

default: build/index.html

clean:
	rm -f *~ */*~ */*/*~

build/index.html: $(SOURCES)
	$(MAKEDOC)

rebuild:
	$(MAKEDOC)

.PHOONY: default clean rebuild
