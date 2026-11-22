## ----setup, include=FALSE-----------------------------------------------------
# Executed locally (NOT_CRAN=true), not on CRAN — same policy as the other
# vignettes. On top of that, every chunk needing two or more GPUs is marked
# eval=FALSE individually: neither CRAN nor a typical workstation has them,
# and a half-executed parallel demo is worse than a printed one. What still
# runs here only queries the build and the driver.
knitr::opts_chunk$set(collapse = TRUE, comment = "#>",
                      eval = identical(Sys.getenv("NOT_CRAN"), "true"))

## -----------------------------------------------------------------------------
# library(ggmlR)

## ----eval=FALSE---------------------------------------------------------------
# # llamaR — whole-LLM inference; the split happens inside the model loader
# model <- llama_load_model("model.gguf", n_gpu_layers = -1L,
#                           devices = c("Vulkan0", "Vulkan1"),
#                           split_mode = "layer")   # or "tensor"

## ----eval=FALSE---------------------------------------------------------------
# # Loopback sanity check: does the transport work at all on device 0?
# r <- ggml_vulkan_p2p_selftest(0L, 0L)
# cat(r$report)
# 
# # The real question: does a cross-device copy carry the bytes?
# if (ggml_vulkan_device_count() >= 2) {
#   r <- ggml_vulkan_p2p_selftest(0L, 1L, transport = "opaque-fd")
#   cat(r$report)          # "verified" vs. a mismatch tells you immediately
# }
# 
# # Are there any multi-device (LDA) groups?
# ggml_vulkan_device_groups()

## ----eval=FALSE---------------------------------------------------------------
# set.seed(1)
# W <- matrix(rnorm(2048 * 64), nrow = 2048)   # [N x K] weights
# X <- matrix(rnorm(4 * 64),    nrow = 4)      # [M x K] activations
# 
# Y <- ggml_vulkan_split_mul_mat(W, X, n_devices = 2)
# 
# # Contract: identical to the single-device result, up to f32 rounding.
# max(abs(Y - X %*% t(W)))                     # ~3.8e-6 on 2 devices

## ----eval=FALSE---------------------------------------------------------------
# Y <- ggml_vulkan_split_mul_mat(W, X, device_ids = c(0L, 1L))
# Y <- ggml_vulkan_split_mul_mat(W, X, n_devices = 2, weights = c(0.7, 0.3))
# 
# # Inspect the row ranges the split math will use (0-based, half-open):
# ggml_vulkan_split_row_ranges(nrows = 2048, n_devices = 2)

## ----eval=FALSE---------------------------------------------------------------
# n_in <- 512L; n_hid <- 2048L; n_out <- 512L; batch <- 16L
# W1 <- array(rnorm(n_in * n_hid),  c(n_in,  n_hid))
# W2 <- array(rnorm(n_hid * n_out), c(n_hid, n_out))
# X  <- array(rnorm(n_in * batch),  c(n_in,  batch))
# 
# mlp <- function(ctx, input, w)          # w: the weight tensors, by name
#   ggml_mul_mat(ctx, w$w2, ggml_relu(ctx, ggml_mul_mat(ctx, w$w1, input)))
# 
# Y <- ggml_tp_forward(
#   weights = list(w1 = W1, w2 = W2),            # names are what `split` refers to
#   forward = mlp,
#   x       = X,                                 # the input tensor is named "input"
#   devices = c(0L, 1L),
#   split   = list(w1 = "rows", w2 = "cols"))
# 
# max(abs(Y - t(W2) %*% pmax(t(W1) %*% X, 0)))

## ----eval=FALSE---------------------------------------------------------------
# Y <- ggml_tp_forward(list(w1 = W1, w2 = W2), mlp, X,
#                      devices = c("CPU", "CPU"),
#                      split = list(w1 = "rows", w2 = "cols"))

## ----eval=FALSE---------------------------------------------------------------
# meta  <- ggml_meta_backend(c(0L, 1L), split = list(w1 = "rows", w2 = "cols"))
# sched <- ggml_backend_sched_new(list(meta, ggml_backend_cpu_init()))
# # Weights (and inputs) in a context allocated on `meta`:
# #   buf <- ggml_backend_alloc_ctx_tensors(ctx_w, meta)
# # The graph in a SEPARATE context, allocated through the scheduler:
# #   ggml_backend_sched_alloc_graph(sched, gf); ggml_backend_sched_graph_compute(sched, gf)

## ----eval=FALSE---------------------------------------------------------------
# K <- 64L; M <- 8L
# W1 <- matrix(rnorm(K * K), nrow = K)
# W2 <- matrix(rnorm(K * K), nrow = K)
# X  <- matrix(rnorm(K * M), nrow = K)     # ggml ne = c(K, M): column m is sample m
# 
# make_stage <- function(dev, Wt, relu) {
#   list(
#     device   = dev,
#     in_shape = c(K, M),
#     build = function(ctx, input) {
#       w <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, K)
#       z <- ggml_mul_mat(ctx, w, input)     # == t(Wt) %*% input in R terms
#       list(
#         output      = if (relu) ggml_relu(ctx, z) else z,
#         set_weights = function() ggml_backend_tensor_set_data(w, as.numeric(Wt))
#       )
#     })
# }
# 
# stages <- list(make_stage(0L, W1, relu = TRUE),
#                make_stage(1L, W2, relu = FALSE))
# 
# y <- ggml_pp_forward(stages, x = as.numeric(X), out_shape = c(K, M))
# Y <- matrix(y, nrow = K, ncol = M)
# 
# max(abs(Y - t(W2) %*% pmax(t(W1) %*% X, 0)))   # ~1.8e-5

## ----eval=FALSE---------------------------------------------------------------
# # Replica A = {GPU0, GPU1}, replica B = {GPU2, GPU3}.
# # Each replica takes half the batch; no traffic crosses between replicas.
# Y <- ggml_tp_dp_forward(W, X, replicas = list(c(0L, 1L), c(2L, 3L)))
# 
# # The PP equivalent takes a factory: make_stages(devices, m_shard) -> stage list
# Y <- ggml_pp_dp_forward(make_stages, x, replicas = list(c(0L, 1L), c(2L, 3L)),
#                         out_ncol = M)

## ----eval=FALSE---------------------------------------------------------------
# ggml_vulkan_shutdown()     # safe anywhere; releases devices

## ----eval=FALSE---------------------------------------------------------------
# ggml_vulkan_shutdown(hard = TRUE, status = 0L)   # never returns

## -----------------------------------------------------------------------------
# ggml_vulkan_hard_exit_available()

## ----eval=FALSE---------------------------------------------------------------
# list.files(system.file("examples", package = "ggmlR"),
#            pattern = "^(tp_|pp_|multi_gpu|dp_)")

## -----------------------------------------------------------------------------
# ggml_vulkan_status()

