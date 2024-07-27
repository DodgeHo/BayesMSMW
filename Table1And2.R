
# Remove everything from this R environment, to speed up
rm(list = ls())

######################################
# Installing the packages
######################################

# Define a function to check and install missing packages
install_if_missing <- function(packages) {
  # Get the list of already installed packages
  installed_packages <- rownames(installed.packages())
  
  # Loop through each package and install if it's not already installed
  for (pkg in packages) {
    if (!pkg %in% installed_packages) {
      install.packages(pkg)
    }
  }
}

# List of required packages
required_packages <- c("expm", "MASS", "truncnorm", "dplyr", "abind","parallel")

# Call the function to install missing packages
install_if_missing(required_packages)

source("BayesMSMW_Code_1.2.R")

######################################
# Prepare General Parameters
######################################

N <- 100
P <- 6
P1 <- 3
P2 <- 3
d <- 5
errvar <- 1
outcome <- "binary"

# True values of parameters
true_tau <- 1/rgamma(1,1,1)
true_tau1 <- 1/rgamma(1,1,1)
ratio <- .5
true_v <- rnorm(d,mean = 0, sd = sqrt(true_tau))  
true_w1 <- rnorm(P1,mean = 0, sd = sqrt(true_tau1))
true_w2 <- rnorm(P2,mean = 0, sd = ratio*sqrt(true_tau1))  
true_w <- c(true_w1,true_w2)
true_B <- as.vector(true_w %*% t(true_v))


######################################
# Generate Data
######################################

# Generate data X
x <- rnorm((N*(P)*(d)),0,1)
X <- array(c(x), dim = c(N,P,d))
Xlist1 <- X[,c(1:P1),]
Xlist2 <- X[,c((P1+1):P),]
Xlistfull <- list(Xlist1,Xlist2)
Xsingle <- Xlist1  # Also prepare single source

# Generate observation data y
mu <- c()
for(n in 1:N){
  mu[n] <- (t(true_w) %*% X[n,,] %*% t(t(true_v)))
}
y <- rnorm(N,mean=mu,sd=rep(errvar,N))
Y <- ifelse(y >= 0.5, 1, 0) # Binary

# Generate Test data
xx0 <- rnorm((N*(P)*(d)),0,1)
Xtest <- array(c(xx0), dim = c(N,P,d))
mutest <- c()
for(n in 1:N){
  mutest[n] <- (t(true_w) %*% Xtest[n,,] %*% t(t(true_v)))
}
yy <- rnorm(N,mean=mutest,sd=rep(errvar,N))
Ytest <- ifelse(yy >= 0.5, 1, 0) # Binary

# Generate Single Test data
Xtest_single <- Xtest
Xtest_single[, 4:6, ] <- Xtest[, 1:3, ]
mutest_single  <- c()
for(n in 1:N){
  mutest_single[n] <- (t(c(true_w1,true_w1)) %*% Xtest_single[n,,] %*% t(t(true_v)))
}
yy_single <- rnorm(N,mean=mutest_single,sd=rep(errvar,N))
Ytest_single <- ifelse(yy_single >= 0.5, 1, 0) # Binary

######################################
# Group Parameters
######################################
full_rank <- min(P, d)
Ranks <- c(2,2,1,1,full_rank,full_rank)
Flag_MS <- c("yes","no","yes","no","yes","no")

# Set a 6x6 Container
Predictions <- matrix(list(), nrow = 6, ncol = 6)

for (i in 1:6) {
  func.test <- BayesMSMW(X.list=Xlistfull, Y.list=Y, multi.source=Flag_MS[i], rank=Ranks[i], outcome=outcome)

  for (j in 1:6) {
    if (Flag_MS[j] == "yes"){
      func.pred <- predict.MSMW(Xtest=Xtest, R=Ranks[j], Ws=func.test$Ws, Vs=func.test$Vs, outcome=outcome)
    }else{
      func.pred <- predict.MSMW(Xtest=Xtest_single, R=Ranks[j], Ws=func.test$Ws, Vs=func.test$Vs, outcome=outcome)
    }
    Predictions[[i, j]] <- func.pred
  }
}

Misclassifications <- matrix(list(), nrow = 6, ncol = 6)
Correlations <- matrix(list(), nrow = 6, ncol = 6)
for (i in 1:6){
  for (j in 1:6){
    data <- Predictions[[i, j]]$prediction
    if (Flag_MS[j] == "yes"){
      Misclassifications[[i, j]] <- mean(Ytest != data)
      Correlations[[i, j]] <- cor(Ytest, data)
    }else{
      Misclassifications[[i, j]] <- mean(Ytest_single != data)
      Correlations[[i, j]] <- cor(Ytest_single, data)
    }
  }
}
