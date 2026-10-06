options(warn=1)
cat("===== R SESSION =====\n")
print(R.version.string)
print(sessionInfo())
pkgs <- c("iMRMC","MRMCaov","lme4","glmmTMB","brms","rstanarm")
cat("\n===== PACKAGE AVAILABILITY =====\n")
for (p in pkgs) {
  ok <- requireNamespace(p, quietly=TRUE)
  cat(p, " available=", ok, "\n", sep="")
  if (ok) cat("  version=", as.character(packageVersion(p)), "\n", sep="")
}
if (requireNamespace("iMRMC", quietly=TRUE)) {
  suppressPackageStartupMessages(library(iMRMC))
  cat("\n===== iMRMC EXPORTED FUNCTIONS OF INTEREST =====\n")
  nms <- getNamespaceExports("iMRMC")
  keep <- nms[grepl("IMRMC|success|binary|uStat11|Design|ScoreMatrix|MRMC", nms, ignore.case=TRUE)]
  cat(paste(sort(keep), collapse="\n"), "\n")
  cat("\n===== FORMALS =====\n")
  for (fn in intersect(c("doIMRMC","successDFtoROCdf","uStat11.conditional","uStat11.jointD","convertDFtoDesignMatrix","convertDFtoScoreMatrix"), nms)) {
    cat("\n--", fn, "--\n")
    print(formals(get(fn, asNamespace("iMRMC"))))
  }
}
cat("\nMRMC_V21_ENV_PROBE_COMPLETE\n")
