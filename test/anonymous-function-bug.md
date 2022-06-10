The `localmean!` and `convolve!` methods have an important perfomance issue illustrating by the following timings:

- Testing `localmean!` with Cartesian box
  old algorithm:  61.523 ms (3941224 allocations: 91.12 MiB)
  new algorithm:   1.133 ms (0 allocations: 0 bytes)
  max. abs. diff: 0.0

- Testing `localmean!` with boolean kernel
  old algorithm:  62.735 ms (3941224 allocations: 91.12 MiB)
  new algorithm:   1.614 ms (0 allocations: 0 bytes)
  max. abs. diff: 0.0

- Testing `localmean!` with floating-point kernel
  old algorithm:  81.449 ms (5731836 allocations: 118.45 MiB)
  new algorithm:   1.898 ms (0 allocations: 0 bytes)
  max. abs. diff: 0.0


- Testing `convolve!` with Cartesian box
  old algorithm:  58.158 ms (3581224 allocations: 54.65 MiB)
  new algorithm: 947.109 μs (0 allocations: 0 bytes)
  max. abs. diff: 0.0

- Testing `convolve!` with boolean kernel
  - old algorithm:  59.879 ms (3581224 allocations: 54.65 MiB)
  - new algorithm:   1.156 ms (0 allocations: 0 bytes)
  - max. abs. diff: 0.0

- Testing `convolve!` with floating-point kernel
  - old algorithm:  68.385 ms (5371836 allocations: 81.97 MiB)
  - new algorithm:   1.652 ms (0 allocations: 0 bytes)
  - max. abs. diff: 0.0

The issue seems to be related to the `init` method when it is an anonymous
function which depends on a local variable.  The new code (commit #) fix this
issue.  For the above tests, the speed-up is about a factor of 60 for an array
of 400×400 `Float64` entries and a neighborhood of 5×3 cells.  On an AMD Ryzen
Threadripper 2950X 16-Core processor at 2.2 GHz, the new versions deliver about
1.6 Gflops for `localmean!` and about 2.2 Gflops for `convolve!`.  These speeds
are not much improved in single precision, so loop vectorization is not
effective.

The bilateral filter and morphological filters do not have the same issue:

- Testing `bilateralfilter!` with weights
  - current algorithm:       163.107 ms (0 allocations: 0 bytes)

- Testing `erode!` with Cartesian box
  - current algorithm:       773.768 μs (6 allocations: 3.42 KiB)

- Testing `erode!` with boolean kernel
  - current algorithm:       768.507 μs (5 allocations: 3.38 KiB)

- Testing `dilate!` with Cartesian box
  - current algorithm:       808.188 μs (5 allocations: 3.38 KiB)

- Testing `dilate!` with boolean kernel
  - current algorithm:       809.718 μs (5 allocations: 3.38 KiB)
