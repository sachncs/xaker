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
- The `Linear` (Katharopoulos et al.) baseline cannot represent
  positional structure; it fails on tasks that require position
  awareness (LRA copy at length=32 yields 14% accuracy vs 87-91%
  for `Standard`, `Xsa`, `Fused`). Use `Standard` / `Xsa` / `Fused`
  whenever the task needs positional recall.
- On Apple Silicon (MPS) the PyTorch linalg kernels have shape bugs
  for batched 4-D `linalg.solve`, `linalg.lu_solve`, and `linalg.eigh`.
  The benchmark suite defaults to CPU; override with
  `XAKER_DEVICE=cuda` on a CUDA host.
- The benchmark suite runs on CPU with `dim=64, heads=4` for
  reproducibility. Real workloads benefit from larger dims and a
  CUDA-capable GPU; the relative ordering across attention variants
  holds at all sizes tested but absolute wall-clock numbers will
  differ.