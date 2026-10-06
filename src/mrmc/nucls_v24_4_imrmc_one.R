args <- commandArgs(trailingOnly=TRUE)
if (length(args) != 5) stop("Usage: input.csv output.csv scenario rep n_selected")
input_path <- args[[1]]
output_path <- args[[2]]
scenario <- args[[3]]
rep <- as.integer(args[[4]])
n_selected <- as.integer(args[[5]])

project <- Sys.getenv("NUCLS_PROJECT_ROOT", unset=Sys.getenv("NUCLS_PROJECT", unset=getwd()))
rlib <- file.path(project,"environments","R_libs_v22")
.libPaths(c(rlib,.libPaths()))

if (!requireNamespace("iMRMC",quietly=TRUE)) stop("iMRMC package not found")
ver <- packageVersion("iMRMC")
if (ver < "2.1.0") stop(paste("iMRMC version is older than required 2.1.0:",ver))
library(iMRMC)

d <- read.csv(input_path,stringsAsFactors=FALSE)
req <- c("readerID","caseID","modalityID","score")
if (!all(req %in% names(d))) stop("Missing required iMRMC columns")
d$readerID <- factor(d$readerID)
d$caseID <- factor(d$caseID)
d$modalityID <- factor(d$modalityID,levels=c("U","E"))
d$score <- as.numeric(d$score)

if (length(unique(d$caseID)) != n_selected) stop("Unexpected case count")
if (length(unique(d$readerID)) != 5) stop("Unexpected reader count")
if (any(!d$score %in% c(0,1))) stop("Score must be binary agreement")
if (anyDuplicated(d[c("readerID","caseID","modalityID")])) stop("Duplicate reader-case-modality rows")

direct <- aggregate(score~modalityID,d,mean)
direct_U <- direct$score[direct$modalityID=="U"]
direct_E <- direct$score[direct$modalityID=="E"]

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
  if (any(!is.finite(est[1:3])) || any(!is.finite(vv[1:3]))) stop("Non-finite iMRMC output")
  se <- sqrt(pmax(vv[1:3],0))
  z <- qnorm(.975)
  # iMRMC order for modalitiesToCompare=c("E","U") is E, U, E-U.
  data.frame(
    method=method,
    target=c("theta_E_selected","theta_U_selected","delta_E_minus_U_selected"),
    estimate=est[1:3],
    variance=vv[1:3],
    se=se,
    ci_low=est[1:3]-z*se,
    ci_high=est[1:3]+z*se,
    interval_width=2*z*se,
    n_readers=as.integer(res$nR),
    n_cases=as.integer(res$nC),
    n_input_rows=nrow(d),
    direct_observed_U=direct_U,
    direct_observed_E=direct_E,
    R_version=R.version.string,
    iMRMC_version=as.character(ver),
    stringsAsFactors=FALSE
  )
}

res <- rbind(
  extract(rc,"iMRMC_uStat11_conditionalD"),
  extract(rj,"iMRMC_uStat11_jointD")
)

tmp <- paste0(output_path,".tmp")
write.csv(res,tmp,row.names=FALSE)
if (!file.rename(tmp,output_path)) stop("Failed atomic output rename")

cat(sprintf("IMRMC_OK scenario=%s rep=%03d ncases=%d rows=%d version=%s\n",
            scenario,rep,n_selected,nrow(d),as.character(ver)))
