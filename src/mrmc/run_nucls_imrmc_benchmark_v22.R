args <- commandArgs(trailingOnly=TRUE)
project <- if (length(args)>=1) args[[1]] else Sys.getenv("NUCLS_PROJECT_ROOT", unset=getwd())
outdir <- file.path(project,"outputs","mrmc_benchmark_v22")
rlib <- file.path(project,"environments","R_libs_v22")
dir.create(outdir,recursive=TRUE,showWarnings=FALSE)
dir.create(rlib,recursive=TRUE,showWarnings=FALSE)
.libPaths(c(rlib,.libPaths()))

cat("R_VERSION=",R.version.string,"\n",sep="")
cat("R_LIBS_USER=",rlib,"\n",sep="")

if (!requireNamespace("iMRMC",quietly=TRUE)) {
  cat("iMRMC_NOT_FOUND_INSTALLING_FROM_CRAN\n")
  install.packages("iMRMC",lib=rlib,repos="https://cloud.r-project.org",dependencies=NA)
}
if (!requireNamespace("iMRMC",quietly=TRUE)) stop("IMRMC_INSTALL_FAILED")
library(iMRMC)
cat("IMRMC_VERSION=",as.character(packageVersion("iMRMC")),"\n",sep="")
if (packageVersion("iMRMC") < "2.1.0") warning("iMRMC version older than benchmarked 2.1.0")

run_one <- function(path,label,expect_cases,fully_crossed=FALSE) {
  d <- read.csv(path,stringsAsFactors=FALSE)
  req <- c("readerID","caseID","modalityID","score")
  if (!all(req %in% names(d))) stop(paste("Missing required columns in",label))
  d$readerID <- factor(d$readerID)
  d$caseID <- factor(d$caseID)
  d$modalityID <- factor(d$modalityID,levels=c("U","E"))
  d$score <- as.numeric(d$score)
  if (length(unique(d$caseID)) != expect_cases) stop(paste("Unexpected case count",label))
  if (any(!d$score %in% c(0,1))) stop("Score must be binary agreement")
  if (anyDuplicated(d[c("readerID","caseID","modalityID")])) stop("Duplicate reader-case-modality rows")
  direct <- aggregate(score~modalityID,d,mean)
  cat("\n=== ",label," ===\n",sep="")
  print(direct)
  
  # conditionalD is the preferred arbitrary-design estimator in the current iMRMC package.
  rc <- iMRMC::uStat11.conditionalD(
    d, modalitiesToCompare=c("E","U"), kernelFlag=1,
    keyColumns=c("readerID","caseID","modalityID","score"))
  rj <- iMRMC::uStat11.jointD(
    d, modalitiesToCompare=c("E","U"), kernelFlag=1,
    keyColumns=c("readerID","caseID","modalityID","score"))
  
  extract <- function(res,method) {
    est <- as.numeric(res$mean)
    vv <- as.numeric(res$var)
    if (length(est)<3 || length(vv)<3) stop("Unexpected uStat11 output")
    se <- sqrt(pmax(vv,0))
    z <- qnorm(.975)
    data.frame(
      analysis=label, method=method,
      target=c("theta_E_reader_case_avg","theta_U_reader_case_avg","delta_E_minus_U_reader_case_avg"),
      estimate=est[1:3], variance=vv[1:3], se=se[1:3],
      ci_low=est[1:3]-z*se[1:3], ci_high=est[1:3]+z*se[1:3],
      n_readers=as.integer(res$nR), n_cases=as.integer(res$nC),
      stringsAsFactors=FALSE)
  }
  a <- extract(rc,"iMRMC_uStat11_conditionalD")
  b <- extract(rj,"iMRMC_uStat11_jointD")
  
  if (fully_crossed) {
    maxdiff <- max(abs(a$estimate-b$estimate),abs(a$variance-b$variance))
    if (!is.finite(maxdiff) || maxdiff > 1e-10) stop(paste("FULLY_CROSSED_CONDITIONAL_JOINT_MISMATCH",maxdiff))
    cat("FULLY_CROSSED_CONDITIONAL_JOINT_CHECK_OK\n")
  }
  list(table=rbind(a,b),conditional=rc,joint=rj)
}

full <- run_one(file.path(outdir,"nucls_mrmc_five_reader_1144_incomplete.csv"),
                "five_reader_1144_incomplete",1144,FALSE)
core <- run_one(file.path(outdir,"nucls_mrmc_five_reader_679_fully_crossed.csv"),
                "five_reader_679_fully_crossed",679,TRUE)
res <- rbind(full$table,core$table)
write.csv(res,file.path(outdir,"nucls_imrmc_benchmark_results_v22.csv"),row.names=FALSE)

# Keep variance components and other package-native outputs for audit.
saveRDS(list(full_conditional=full$conditional,full_joint=full$joint,
             core_conditional=core$conditional,core_joint=core$joint),
        file.path(outdir,"nucls_imrmc_native_outputs_v22.rds"))

cat("\nV22_IMRMC_BENCHMARK_RESULTS\n")
print(res,row.names=FALSE)
cat("V22_IMRMC_BENCHMARK_COMPLETE\n")
