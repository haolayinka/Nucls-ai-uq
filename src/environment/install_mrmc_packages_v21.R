args <- commandArgs(trailingOnly=TRUE)
lib <- if (length(args) >= 1) args[[1]] else "environments/R_libs_v21"
dir.create(lib, recursive=TRUE, showWarnings=FALSE)
.libPaths(c(lib, .libPaths()))
needed <- c("iMRMC","MRMCaov")
repos <- c(CRAN="https://cloud.r-project.org")
cat("library path:", lib, "\n")
for (p in needed) {
  if (!requireNamespace(p, quietly=TRUE)) {
    cat("Installing", p, "from CRAN...\n")
    install.packages(p, lib=lib, repos=repos, dependencies=TRUE)
  }
  if (!requireNamespace(p, quietly=TRUE)) stop("PACKAGE_INSTALL_FAILED: ", p)
  cat(p, "version", as.character(packageVersion(p)), "\n")
}
cat("MRMC_V21_INSTALL_COMPLETE\n")
