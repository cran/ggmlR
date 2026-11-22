## ----setup, include=FALSE-----------------------------------------------------
# Vignette code is executed locally (NOT_CRAN=true) but not on CRAN, where
# the CPU fallback would multi-thread and trip the "CPU time > elapsed" NOTE.
knitr::opts_chunk$set(eval = identical(Sys.getenv("NOT_CRAN"), "true"))

## -----------------------------------------------------------------------------
# library(ggmlR)
# 
# run_gelu_back <- function(x, dy, backend = c("cpu", "vulkan")) {
#   backend <- match.arg(backend)
# 
#   ctx <- ggml_init(16 * 1024 * 1024)
#   ggml_set_no_alloc(ctx, TRUE)
#   on.exit(ggml_free(ctx), add = TRUE)
# 
#   if (backend == "vulkan") {
#     b <- ggml_vulkan_init(0L)
#   } else {
#     b <- ggml_backend_cpu_init()
#     ggml_backend_cpu_set_n_threads(b, 2L)
#   }
#   on.exit(ggml_backend_free(b), add = TRUE)
# 
#   g <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, length(dy))
#   x_t <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, length(x))
#   ggml_set_input(g)
#   ggml_set_input(x_t)
# 
#   out <- ggml_gelu_back(ctx, g, x_t)
#   ggml_set_output(out)
# 
#   # Does this device run the node itself, or would it fall back to the CPU?
#   on_device <- ggml_backend_dev_supports_op(ggml_backend_get_device(b), out)
# 
#   buf <- ggml_backend_alloc_ctx_tensors(ctx, b)
#   ggml_backend_tensor_set_data(g, dy)
#   ggml_backend_tensor_set_data(x_t, x)
#   ggml_backend_graph_compute(b, ggml_build_forward_expand(ctx, out))
# 
#   list(value = ggml_backend_tensor_get_data(out), on_device = on_device)
# }
# 
# set.seed(1)
# x  <- rnorm(10000, sd = 3)
# dy <- rnorm(10000)
# 
# cpu <- run_gelu_back(x, dy, "cpu")
# cpu$on_device
# 
# if (ggml_vulkan_available()) {
#   gpu <- run_gelu_back(x, dy, "vulkan")
#   print(gpu$on_device)                # TRUE: the shader runs, not a fallback
#   max(abs(gpu$value - cpu$value))     # float32 rounding only
# }

