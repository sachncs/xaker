# Limitations

- The PCG iterative solver assumes the regularized kernel matrix
  `(K + lam I)` is positive-definite. The package does not enforce this;
  random Gaussian kernels often fail to converge and fall back to
  `torch.linalg.solve` (a single dense solve).
- The CCCP preconditioner is O(n^3) per iteration; for sequences beyond
  ~512 tokens, prefer `precond="fast"` or `precond="diagonal"`.
- Gradient backpropagation through the PCG loop is not implemented as a
  custom `torch.autograd.Function`; gradients flow to the kernel and
  preconditioner parameters only via the direct-solve fallback path.
- Mixed precision (`fp16`, `bf16`) is not exercised end-to-end; some
  paths may numerically underflow in lower-precision dtypes.
- The v1 attention classes (e.g. `KernelAttentionRegression`) have been
  hard-deleted; checkpoints saved against them cannot be loaded directly
  through `Model.load_state_dict`.