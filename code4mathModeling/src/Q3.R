##
usa <- read.csv("../dataset/usa.csv", header=FALSE)
zh <- read.csv("../dataset/zh.csv", header=FALSE)
library(psych)
uscov<-cov(usa[,2:6])
fa.parallel(cov2cor(uscov), n.obs = 100, fa="both", n.iter = 20, show.legend = FALSE)
fausa<-fa(cov2cor(uscov), n.obs=100, nfactors = 2, rotate = "varimax", fm="lm", n.iter = 20)

zhcov<-cov(zh[,2:7])
fa.parallel(cov2cor(zhcov), n.obs = 100, fa="both", n.iter = 20, show.legend = FALSE)

fazh<-fa(cov2cor(zhcov), n.obs=100, nfactors = 2, rotate = "varimax", fm="ml", n.iter = 20)

principal(uscov)
principal(zhcov)