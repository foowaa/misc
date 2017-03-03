# 相关系数
fensheng_xiaomai <- read.csv("../dataset/fensheng_xiaomai.csv")
fensheng_daogu <- read.csv("../dataset/fensheng_daogu.csv")
fensheng_yumi <- read.csv("../dataset/fensheng_yumi.csv")
ah_shouru <- read.csv("../dataset/ah_shouru.csv")
hn_shouru <- read.csv("../dataset/hn_shouru.csv")
cov(fensheng_yumi)
cov(fensheng_daogu)
cov(fensheng_xiaomai)
cov(fensheng_yumi, method='sperman')
cov(fensheng_daogu, method='spearman')
cov(fensheng_xiaomai, method='spearman)
cov(fensheng_yumi, method='kendall')
cov(fensheng_daogu, method='kendall')
cov(fensheng_xiaomai, method='kendall')
cov(ah_shouru)
cov(hn_shouru)
cov(ah_shouru, method='spearman')
cov(hn_shouru, method='spearman')
cov(ah_shouru, method='kendall')
cov(hn_shouru, method='kendall')



# 柯布-道格拉斯函数拟合
nn <- read.csv("../dataset/d22.csv", header=FALSE)
nn1<-nn[-1,]   #删除题头
mnn<as.matrix(nn1)
colnames(mnn)=NULL
rownames（mnn)=NULL
b=matrix(as.numeric(mnn),nrow=nrow(mnn))
xnn<-scale(b)  #归一化

axnn<-xnn[,2]+xnn[,3]+xnn[,4]
naxnn<scale(axnn)
mdf<-cbind(xnn[,1],naxnn, xnn[,5], xnn[,6],xnn[,7],xnn[,8])
df<-data.frame(mdf)

# fit

fit<-lm(log(X1)~log(X2)+log(X3)+log(X4), data=df)
df1<-df[1:6,]
df2<-df[7:15,]
fit1<-lm(log(-X1)~log(-X2)+log(-X3)+log(-X4), data=df1)
fit2<-lm(log(X1)~log(X2)+log(X3)+log(X4), data=df2)
summary(fit)
summary(fit1)
summary(fit2)

par(mfrow=c(2,2))
plot(fit1)
plot(fit2)

# ANOVA
d31 <- read.csv("../dataset/d31.csv", header=FALSE)
d32 <- read.csv("../dataset/d32.csv", header=FALSE)
fita<-aov(d31$total~d31$prv+d31$LP)
summary(fita)

fitb<-aov(d32$total~d32$pz+d32$LP)
summary(fitb)



