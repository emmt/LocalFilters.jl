# Wish list and remaining issues

- Implement other boundary conditions.

- Consistency: more possibilities for `strel` (like building a kernel) and
  `is_morpho_math_box` should yield `true` for any acceptable arguments for `box`.

- Check doc. for math. morphology and bilateral filter.

- Write doc. for van Herk-Gil-Werman algorithm and make it more direct to found.

- Homogenize names: `LocalFilters.reversed` and `LocalFilters.centered` or
  `reverse_kernel` and `center_kernel`. The former makes more sense since the
  methods can be applied to other things than a kernel. There should be a
  `LocalFilters.reversed_conjugate`?
