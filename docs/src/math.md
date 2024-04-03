# Convolution, correlation, and Fourier transform

## Fourier transform

The continuous Fourier transform of ``a(x)`` is defined by:

```math
\hat{a}(u) = \int a(x)\,\mathrm{e}^{-\mathrm{i}\,2\,\pi\,u\,x}\,\mathrm{d}x.
```

The inverse Fourier transform of ``\hat{a}(u)`` then writes:

```math
a(x) = \int \hat{a}(u)\,\mathrm{e}^{+\mathrm{i}\,2\,\pi\,u\,x}\,\mathrm{d}u.
```

## Convolution product

The convolution product of ``a(x)`` by ``b(x)`` is defined by:

```math
c(x) = \mathrm{Conv}(a,b)(x)
= \int a(y)\,b(x - y)\,\mathrm{d}y
= \int b(z)\,a(x - z)\,\mathrm{d}z,
```

with ``z = x - y``. This also shows that the convolution product is
symmetrical:

```math
\mathrm{Conv}(b,a) = \mathrm{Conv}(a,b).
```

Taking ``z = x - y``, the Fourier transform of the convolution product can be
expanded as follows:

```math
\begin{align*}
\hat{c}(u) &= \int c(x)\,\mathrm{e}^{-\mathrm{i}\,2\,\pi\,u\,x}\,\mathrm{d}x\\
&= \iint a(y)\,b(x - y)\,\mathrm{e}^{-\mathrm{i}\,2\,\pi\,u\,x}\,\mathrm{d}x\,\mathrm{d}y\\
&= \iint a(y)\,b(z)\,\mathrm{e}^{-\mathrm{i}\,2\,\pi\,u\,(y + z)}\,\mathrm{d}y\,\mathrm{d}z\\
&= \int a(y)\,\mathrm{e}^{-\mathrm{i}\,2\,\pi\,u\,y}\,\mathrm{d}y
   \int b(z)\,\mathrm{e}^{-\mathrm{i}\,2\,\pi\,u\,z}\,\mathrm{d}z\\
&= \hat{a}(u)\,\hat{b}(u).
\end{align*}
```


## Correlation product

The correlation product of ``a(x)`` by ``b(x)`` is defined by:

```math
r(x) = \mathrm{Corr}(a,b)(x)
= \int a(x + y)\,{b}^\star(y)\,\mathrm{d}y
= \int {b}^\star(z - x)\,a(z)\,\mathrm{d}z,
```

where ``{b}^\star(y)`` denotes the complex conjugate of ``b(y)`` and with ``z =
x + y``. From this follows that:

```math
\mathrm{Corr}(b,a)(x) = {\mathrm{Corr}(a,b)}^\star(-x).
```

Taking ``z = x + y``, the Fourier transform of the correlation product can be
expanded as follows:

```math
\begin{align*}
\hat{r}(u) &= \int r(x)\,\mathrm{e}^{-\mathrm{i}\,2\,\pi\,u\,x}\,\mathrm{d}x\\
&= \iint a(x + y)\,{b}^\star(y)\,\mathrm{e}^{-\mathrm{i}\,2\,\pi\,u\,x}\,\mathrm{d}x\,\mathrm{d}y\\
&= \iint a(z)\,b^\star(y)\,\mathrm{e}^{-\mathrm{i}\,2\,\pi\,u\,(z - y)}\,\mathrm{d}y\,\mathrm{d}z\\
&= \int a(z)\,\mathrm{e}^{-\mathrm{i}\,2\,\pi\,u\,z}\,\mathrm{d}z
   \left(\int b(y)\,\mathrm{e}^{-\mathrm{i}\,2\,\pi\,u\,y}\,\mathrm{d}y\right)^\star\\
&= \hat{a}(u)\,{\hat{b}}^\star(u).
\end{align*}
```

## Discrete convolution and correlation

Following the continuous definition, the discrete convolution of ``a`` by ``b``
is given by:

```math
c[i] = \sum_j a[j]\,b[i - j] = \sum_k b[k]\,a[i - k],
```

with ``k = i - j`` and where the sums are taken for all possible valid indices.

Similarly, following the continuous definition, the discrete correlation of
``a`` by ``b`` is given by:

```math
r[i] = \sum_k a[i + k]\,{b}^\star[k] = \sum_k {b}^\star[j - i]\,a[j],
```

with ``j = i + k`` and where the sums are taken for all possible valid indices.
