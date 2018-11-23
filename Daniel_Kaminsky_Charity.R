###########################################
#### Charity Project - Daniel Kaminsky ####
###########################################

# OBJECTIVE: A charitable organization wishes to develop a machine learning
# model to improve the cost-effectiveness of their direct marketing campaigns
# to previous donors.

# 1) Develop a classification model using data from the most recent campaign that
# can effectively capture likely donors so that the expected net profit is maximized.

# 2) Develop a prediction model to predict donation amounts for donors - the data
# for this will consist of the records for donors only.

# Cleaning the workspace
rm(list = ls())

# Loading the libraries
library(caret)
library(corrplot)
library(elasticnet)
library(foreach)
library(flexmix)
library(ggplot2)
library(glmnet)
library(gridExtra)
library(kernlab)
library(lars)
library(MASS)
library(Matrix)
library(Metrics)
library(mgcv)
library(moments)
library(neuralnet)
library(nnet)
library(plyr)
library(pscl) # For Zero Inflated Poisson and Negative Binomial Regressions
library(randomForest)
library(rpart)
library(rattle)

# load the data
charity <- read.csv(file.choose()) # load the "charity.csv" file
str(charity)
head(charity)
tail(charity)

###############################
## Exploratory Data Analysis ##
###############################

# Structure
str(charity, list.len = ncol(charity)) # 'data.frame':	8009 obs. of  24 variables

# Training dataset
data.train <- charity[charity$part=="train",]
str(data.train, list.len = ncol(data.train)) # 'data.frame':	3984 obs. of  24 variables

# Subsetting to remove Factor variables
myvars <- names(data.train) %in% c("ID",
                                   "reg1",
                                   "reg2",
                                   "reg3",
                                   "reg4",
                                   "home",
                                   "chld",
                                   "hinc",
                                   "genf",
                                   "wrat",
                                   "part")
chari.tr.NoFactor <- data.train[!myvars]

str(chari.tr.NoFactor, list.len = ncol(chari.tr.NoFactor)) # 'data.frame':	3984 obs. of  13 variables
head(chari.tr.NoFactor)

# Correlation Matrix
#  library(corrplot)
Correlation1 <- cor(chari.tr.NoFactor)
corrplot(Correlation1, method="circle", type="lower")
corrplot(Correlation1, method="number", type="lower")

# Scatterplot Matrix
jittered_x <- sapply(chari.tr.NoFactor[, c(1:3, 9:10, 13:13)], jitter)
pairs(jittered_x, names(chari.tr.NoFactor[, c(1:3, 9:10, 13:13)]), col=(chari.tr.NoFactor$donr)+1)

###########################################
### Histograms, Q-Q Plots and Box Plots ###
###########################################

# damt
par(mfrow = c(2, 2), mar = c(5.1, 6.1, 4.1, 2.1))
hist(charity$damt, col = "deepskyblue3", main = "Histogram of damt", xlab = "damt",
     cex = 2, cex.axis = 1.5, cex.lab = 1.5, cex.main = 1.5, cex.sub = 1.5)
qqnorm(charity$damt, col = "deepskyblue3", pch = 'o', main = "Normal Q-Q Plot",
       cex = 2, cex.axis = 1.5, cex.lab = 1.5, cex.main = 1.5, cex.sub = 1.5)
qqline(charity$damt, col = "darkred", lty = 2, lwd = 3)
boxplot(charity$damt[charity$damt], col = "red", pch = 16,
        main = "damt", cex = 2.0, cex.axis = 1.5, cex.lab = 1.5, cex.main = 1.5)
par(mfrow = c(1, 1), mar = c(5.1, 4.1, 4.1, 2.1))

# avhv
par(mfrow = c(2, 2), mar = c(5.1, 6.1, 4.1, 2.1))
hist(charity$avhv, col = "deepskyblue3", main = "Histogram of avhv", xlab = "avhv",
     cex = 2, cex.axis = 1.5, cex.lab = 1.5, cex.main = 1.5, cex.sub = 1.5)
qqnorm(charity$avhv, col = "deepskyblue3", pch = 'o', main = "Normal Q-Q Plot",
       cex = 2, cex.axis = 1.5, cex.lab = 1.5, cex.main = 1.5, cex.sub = 1.5)
qqline(charity$avhv, col = "darkred", lty = 2, lwd = 3)
boxplot(charity$avhv[charity$avhv], col = "red", pch = 16,
        main = "avhv", cex = 2.0, cex.axis = 1.5, cex.lab = 1.5, cex.main = 1.5)
par(mfrow = c(1, 1), mar = c(5.1, 4.1, 4.1, 2.1))

# incm
par(mfrow = c(2, 2), mar = c(5.1, 6.1, 4.1, 2.1))
hist(charity$incm, col = "deepskyblue3", main = "Histogram of incm", xlab = "incm",
     cex = 2, cex.axis = 1.5, cex.lab = 1.5, cex.main = 1.5, cex.sub = 1.5)
qqnorm(charity$incm, col = "deepskyblue3", pch = 'o', main = "Normal Q-Q Plot",
       cex = 2, cex.axis = 1.5, cex.lab = 1.5, cex.main = 1.5, cex.sub = 1.5)
qqline(charity$incm, col = "darkred", lty = 2, lwd = 3)
boxplot(charity$incm[charity$incm], col = "red", pch = 16,
        main = "incm", cex = 2.0, cex.axis = 1.5, cex.lab = 1.5, cex.main = 1.5)
par(mfrow = c(1, 1), mar = c(5.1, 4.1, 4.1, 2.1))

# inca
par(mfrow = c(2, 2), mar = c(5.1, 6.1, 4.1, 2.1))
hist(charity$inca, col = "deepskyblue3", main = "Histogram of inca", xlab = "inca",
     cex = 2, cex.axis = 1.5, cex.lab = 1.5, cex.main = 1.5, cex.sub = 1.5)
qqnorm(charity$inca, col = "deepskyblue3", pch = 'o', main = "Normal Q-Q Plot",
       cex = 2, cex.axis = 1.5, cex.lab = 1.5, cex.main = 1.5, cex.sub = 1.5)
qqline(charity$inca, col = "darkred", lty = 2, lwd = 3)
boxplot(charity$inca[charity$inca], col = "red", pch = 16,
        main = "inca", cex = 2.0, cex.axis = 1.5, cex.lab = 1.5, cex.main = 1.5)
par(mfrow = c(1, 1), mar = c(5.1, 4.1, 4.1, 2.1))

# Checking skewness and kurtosis helps to reveal more about distribution shape.  
# A normal distribution has a skewness of zero and kurtosis of 3.0.
moments::skewness(charity$avhv) # 1.539032
moments::kurtosis(charity$avhv) # 7.488119
moments::skewness(charity$incm) # 2.051239
moments::kurtosis(charity$incm) # 11.30928
moments::skewness(charity$inca) # 1.937378
moments::kurtosis(charity$inca) # 10.87691

### predictor transformations ###
charity.t <- charity

# Log avhv
charity.t$avhv <- log(charity.t$avhv)
head(charity$avhv)
head(charity.t$avhv)

# Log incm
charity.t$incm <- log(charity.t$incm)
head(charity$incm)
head(charity.t$incm)

# Log inca
charity.t$inca <- log(charity.t$inca)
head(charity$inca)
head(charity.t$inca)

### Feature Engineering ###
charity.t$rgip <- charity.t$tgif/charity.t$npro # Ratio of "Total Gifts" to "Total Promotions"
head(charity.t)

# log.avhv
par(mfrow = c(2, 2), mar = c(5.1, 6.1, 4.1, 2.1))
hist(charity.t$avhv, col = "deepskyblue3", main = "Histogram of Log avhv", xlab = "Log avhv",
     cex = 2, cex.axis = 1.5, cex.lab = 1.5, cex.main = 1.5, cex.sub = 1.5)
qqnorm(charity.t$avhv, col = "deepskyblue3", pch = 'o', main = "Normal Q-Q Plot",
       cex = 2, cex.axis = 1.5, cex.lab = 1.5, cex.main = 1.5, cex.sub = 1.5)
qqline(charity.t$avhv, col = "darkred", lty = 2, lwd = 3)
boxplot(charity.t$avhv[charity.t$avhv], col = "red", pch = 16,
        main = "Log avhv", cex = 2.0, cex.axis = 1.5, cex.lab = 1.5, cex.main = 1.5)
par(mfrow = c(1, 1), mar = c(5.1, 4.1, 4.1, 2.1))

# Log.incm
par(mfrow = c(2, 2), mar = c(5.1, 6.1, 4.1, 2.1))
hist(charity.t$incm, col = "deepskyblue3", main = "Histogram of Log incm", xlab = "incm",
     cex = 2, cex.axis = 1.5, cex.lab = 1.5, cex.main = 1.5, cex.sub = 1.5)
qqnorm(charity.t$incm, col = "deepskyblue3", pch = 'o', main = "Normal Q-Q Plot",
       cex = 2, cex.axis = 1.5, cex.lab = 1.5, cex.main = 1.5, cex.sub = 1.5)
qqline(charity.t$incm, col = "darkred", lty = 2, lwd = 3)
boxplot(charity.t$incm[charity.t$incm], col = "red", pch = 16,
        main = "Log incm", cex = 2.0, cex.axis = 1.5, cex.lab = 1.5, cex.main = 1.5)
par(mfrow = c(1, 1), mar = c(5.1, 4.1, 4.1, 2.1))

# Log.inca
par(mfrow = c(2, 2), mar = c(5.1, 6.1, 4.1, 2.1))
hist(charity.t$inca, col = "deepskyblue3", main = "Histogram of Log inca", xlab = "inca",
     cex = 2, cex.axis = 1.5, cex.lab = 1.5, cex.main = 1.5, cex.sub = 1.5)
qqnorm(charity.t$inca, col = "deepskyblue3", pch = 'o', main = "Normal Q-Q Plot",
       cex = 2, cex.axis = 1.5, cex.lab = 1.5, cex.main = 1.5, cex.sub = 1.5)
qqline(charity.t$inca, col = "darkred", lty = 2, lwd = 3)
boxplot(charity.t$inca[charity.t$inca], col = "red", pch = 16,
        main = "Log inca", cex = 2.0, cex.axis = 1.5, cex.lab = 1.5, cex.main = 1.5)
par(mfrow = c(1, 1), mar = c(5.1, 4.1, 4.1, 2.1))

###########################################################
### Frequency Table for Categorical Variables Selection ###
###########################################################

# Structure charity.t set
str(charity.t, list.len = ncol(charity.t))

f1 <- as.data.frame(table(charity.t$reg1))
names(f1)[1] = 'reg1'
f1
#   reg1 Freq
# 1    0 6404
# 2    1 1605

f2 <- as.data.frame(table(charity.t$reg2))
names(f2)[1] = 'reg2'
f2
#   reg2 Freq
# 1    0 5454
# 2    1 2555

f3 <- as.data.frame(table(charity.t$reg3))
names(f3)[1] = 'reg2'
f3
#   reg2 Freq
# 1    0 6938
# 2    1 1071

f4 <- as.data.frame(table(charity.t$reg4))
names(f4)[1] = 'reg2'
f4
#   reg2 Freq
# 1    0 6892
# 2    1 1117

f5 <- as.data.frame(table(charity.t$home))
names(f5)[1] = 'home'
f5
#   home Freq
# 1    0 1069
# 2    1 6940

f6 <- as.data.frame(table(charity.t$hinc))
names(f6)[1] = 'hinc'
f6
#   hinc Freq
# 1    1  522
# 2    2 1021
# 3    3  822
# 4    4 3462
# 5    5 1152
# 6    6  544
# 7    7  486

f7 <- as.data.frame(table(charity.t$wrat))
names(f7)[1] = 'wrat'
f7
#    wrat Freq
# 1     0  222
# 2     1  198
# 3     2  250
# 4     3  333
# 5     4  443
# 6     5  408
# 7     6  516
# 8     7  480
# 9     8 3021
# 10    9 2138

# Boxplot of damt by genf 
boxplot(damt~genf,data=charity.t, main="genf vs. damt", 
        xlab="Gender", ylab="Amount Donated",col=(c("gold","lightgreen")))

# Boxplot of damt by chld 
boxplot(damt~chld,data=charity.t, main="chld vs. damt", 
        xlab="Number of Children", ylab="Amount Donated",col=(c("gold","lightgreen")))

# Decision Tree
DTree<- rpart(formula = damt~.-part-donr-ID, data=charity.t,
              method="anova",parms=list(split="information"),
              control=rpart.control(maxdepth=10,usersurrogate=0,
                            maxsurrogate=0))
fancyRpartPlot(DTree)
summary(DTree)
# Variable importance
# chld hinc home reg2 wrat tdon tlag 
#  53   13   11   11    7    4    2 

################################
### set up data for modeling ###
################################

# Training Dataset
data.train <- charity.t[charity$part=="train",]
x.train <- data.train[, c(2:21, 25:25)]
c.train <- data.train[,22] # donr
n.train.c <- length(c.train) # 3984
y.train <- data.train[c.train==1,23] # damt for observations with donr=1
n.train.y <- length(y.train) # 1995
head(x.train)
head(y.train)

# Missing Values - Count per Variable
sapply(data.train, function(data.train) sum(is.na(data.train)))
summary(data.train)

# Validation dataset
data.valid <- charity.t[charity$part=="valid",]
x.valid <- data.valid[, c(2:21, 25:25)]
c.valid <- data.valid[,22] # donr
n.valid.c <- length(c.valid) # 2018
y.valid <- data.valid[c.valid==1,23] # damt for observations with donr=1
n.valid.y <- length(y.valid) # 999

# Missing Values - Count per Variable
sapply(data.valid, function(data.valid) sum(is.na(data.valid)))
summary(data.valid)

# Test Dataset
data.test <- charity.t[charity$part=="test",]
n.test <- dim(data.test)[1] # 2007
x.test <- data.test[, c(2:21, 25:25)]

# Missing Values - Count per Variable
sapply(data.test, function(data.test) sum(is.na(data.test)))
summary(data.test)

###############################################
### Standardizing Train and Validation sets ###
###############################################
x.train.mean <- apply(x.train, 2, mean)
x.train.sd <- apply(x.train, 2, sd)
x.train.std <- t((t(x.train)-x.train.mean)/x.train.sd) # standardize to have zero mean and unit sd
apply(x.train.std, 2, mean) # check zero mean
apply(x.train.std, 2, sd) # check unit sd
data.train.std.c <- data.frame(x.train.std, donr=c.train) # to classify donr
data.train.std.y <- data.frame(x.train.std[c.train==1,], damt=y.train) # to predict damt when donr=1

x.valid.std <- t((t(x.valid)-x.train.mean)/x.train.sd) # standardize using training mean and sd
data.valid.std.c <- data.frame(x.valid.std, donr=c.valid) # to classify donr
data.valid.std.y <- data.frame(x.valid.std[c.valid==1,], damt=y.valid) # to predict damt when donr=1

x.test.std <- t((t(x.test)-x.train.mean)/x.train.sd) # standardize using training mean and sd
data.test.std <- data.frame(x.test.std)

####################################
##### CLASSIFICATION MODELING ######
####################################

# Removing the ID, damt and part columns
Train.donr <- data.train.std.c # data.train
Train.donr$ID <- NULL
Train.donr$damt <- NULL
Train.donr$part <- NULL
head(Train.donr)

#################################################################
########## Modeling the Train Dataset - Classification ##########
#################################################################

# Recode the class (donr) label to Yes/No (required when using classProbs=TRUE in trainControl)
required.labels <- Train.donr['donr']
recoded.labels <- car::recode(required.labels$donr, "0='No'; 1 = 'Yes'")
Train.donr$donr <- recoded.labels
Train.donr$donr  <-as.factor(Train.donr$donr) # Make the donr variable a factor
str(Train.donr)
head(Train.donr)

# ?trainControl
# Methods available
names(getModelInfo())

# Baseline Models
# 10-fold cross validation with 3 repeats
trainControl <- trainControl(method="repeatedcv", summaryFunction=defaultSummary,
                             number=10, repeats=3, classProbs=TRUE)

# Train using Logistic Regression (glm), Linear Discriminant Analysis (lda),
# Quadratic Discriminant Analysis (qda), Regularized Logistic Regression (glmnet),
# Classification and Regression Trees (rpart), Support Vector Machines with
# Radial Basis Functions (svmRadial), Artificial Neural Networks (nnet),
# and K-Nearest Neighbor (knn) using the caret package.
set.seed(7)
fit.glm <- train(donr~., data=Train.donr, method="glm", 
                 trControl=trainControl, family=binomial("logit")) # GLM

set.seed(7)
fit.lda <- train(donr~., data=Train.donr, method="lda", 
                 trControl=trainControl) # LDA

set.seed(7)
fit.qda <- train(donr~., data=Train.donr, method="qda", 
                 trControl=trainControl) # QDA

set.seed(7)
fit.glmnet <- train(donr~., data=Train.donr, method="glmnet", 
                    family = "binomial", trControl=trainControl)  # GLMNET (Regularized Logit)

set.seed(7)
fit.cart <- train(donr~., data=Train.donr, method="rpart", 
                  trControl=trainControl)  # CART

set.seed(7)
fit.svm <- train(donr~., data=Train.donr, method="svmRadial", 
                 trControl=trainControl)  # SVM

set.seed(7)
fit.nnet <- train(donr~., data=Train.donr, method="nnet", 
                 trControl=trainControl)  # ANN

set.seed(7)
fit.knn <- train(donr~., data=Train.donr, method="knn", 
                  trControl=trainControl)  # KNN

# Compare algorithms
results <- resamples(list(LG=fit.glm, LDA=fit.lda, QDA=fit.qda,
                          GLMNET=fit.glmnet, CART=fit.cart, 
                          SVM=fit.svm, ANN=fit.nnet, KNN=fit.knn))
summary(results)
dotplot(results, main=" Accuracy and Kappa Chart - Baseline Models")

##########################################################################
### Models with variables selected by a Decision Tree - Classification ###
##########################################################################

# Decision Tree Variable importance
# chld hinc home reg2 wrat tdon tlag 
#  53   13   11   11    7    4    2 

set.seed(7)
fit.glm2 <- train(donr~chld+hinc+home+reg1+reg2+reg3+reg4+wrat+tdon+tlag+incm+rgip,
                 data=Train.donr, method="glm", trControl=trainControl, family=binomial("logit")) # GLM

set.seed(7)
fit.lda2 <- train(donr~chld+hinc+home+reg1+reg2+reg3+reg4+wrat+tdon+tlag+incm+rgip,
                 data=Train.donr, method="lda", trControl=trainControl) # LDA

set.seed(7)
fit.qda2 <- train(donr~chld+hinc+home+reg1+reg2+reg3+reg4+wrat+tdon+tlag+incm+rgip,
                 data=Train.donr, method="qda", trControl=trainControl) # QDA

set.seed(7)
fit.glmnet2 <- train(donr~chld+hinc+home+reg1+reg2+reg3+reg4+wrat+tdon+tlag+incm+rgip,
                    data=Train.donr, method="glmnet", family = "binomial",
                    trControl=trainControl)  # GLMNET

set.seed(7)
fit.cart2 <- train(donr~chld+hinc+home+reg1+reg2+reg3+reg4+wrat+tdon+tlag+incm+rgip,
                  data=Train.donr, method="rpart", trControl=trainControl)  # CART

set.seed(7)
fit.svm2 <- train(donr~chld+hinc+home+reg1+reg2+reg3+reg4+wrat+tdon+tlag+incm+rgip,
                 data=Train.donr, method="svmRadial", trControl=trainControl)  # SVM

set.seed(7)
fit.nnet2 <- train(donr~chld+hinc+home+reg1+reg2+reg3+reg4+wrat+tdon+tlag+incm+rgip,
                  data=Train.donr, method="nnet", trControl=trainControl)  # ANN

set.seed(7)
fit.knn2 <- train(donr~chld+hinc+home+reg1+reg2+reg3+reg4+wrat+tdon+tlag+incm+rgip,
                    data=Train.donr, method="knn", trControl=trainControl)  # KNN

# Compare algorithms
results2 <- resamples(list(LG=fit.glm2, LDA=fit.lda2, QDA=fit.qda2,
                          GLMNET=fit.glmnet2, CART=fit.cart2, 
                          SVM=fit.svm2, ANN=fit.nnet2, KNN=fit.knn2))
summary(results2)
dotplot(results2, main=" Accuracy and Kappa Chart - Decision Tree Variables Models")

#################################################################
####### Modeling the Validation Dataset - Classification ########
#################################################################

# Reproducing the data cleansing performed in the training dataset
# Removing the ID, damt and part columns
Valid.donr <- data.valid.std.c # data.valid
Valid.donr$ID <- NULL
Valid.donr$damt <- NULL
Valid.donr$part <- NULL
head(Valid.donr)

# Decision Tree Variable importance
# chld hinc home reg2 wrat tdon tlag 
#  53   13   11   11    7    4    2 
head(Valid.donr[, c(1:7, 9:9, 11:11, 18:19, 21:22)])

### Validating GLM2
set.seed(7)
Valid.pred1 <- predict(fit.glm2, Valid.donr[, c(1:7, 9:9, 11:11, 18:19, 21:22)], type="prob")
summary(Valid.pred1)
head(Valid.pred1[,2])

# calculate ordered profit function using average donation = $14.50 and mailing cost = $2
profit.glm2 <- cumsum(14.5*c.valid[order(Valid.pred1[,2], decreasing=T)]-2)
n.mail.valid1 <- which.max(profit.glm2) # number of mailings that maximizes profits
plot(profit.glm2, xlab = "Number of Mailings Sent", col="darkblue",
     ylab = "Profits",
     main="GLM2 Model - Profits Change per Number of Mailings") # see how profits change as more mailings are made
points(n.mail.valid1, max(profit.glm2), pch=19, col="red")
c(n.mail.valid1, max(profit.glm2)) # report number of mailings and maximum profit
# 1426.0 11256.5
# 1417.0 11332.5 with incm
# 1431.0 11333.5 with incm and rgip

cutoff.glm2 <- sort(Valid.pred1[,2], decreasing=T)[n.mail.valid1+1] # set cutoff based on n.mail.valid
chat.valid.glm2 <- ifelse(Valid.pred1[,2]>cutoff.glm2, 1, 0) # mail to everyone above the cutoff
table(chat.valid.glm2, c.valid) # classification table
#               c.valid
#chat.valid.log1   0   1
#              0 567  20
#              1 452 979
# check n.mail.valid = 452+979 = 1,431
# check profit = 14.5*979-2*1431 = 11,333.5

# Results
# n.mail Profit  Model
# 1431   11333.5 GLM2

# MSE
mse(Valid.donr$donr,Valid.pred1[[2]]) # 0.1244016 (with incm and rgip = 0.1206083)

# Multiple R-Squared
R2.0 <- 1-(sum((Valid.donr$donr-Valid.pred1[[2]])^2)/sum((Valid.donr$donr-mean(Valid.donr$donr))^2))
R2.0 # 0.5023446 (with incm and rgip = 0.5175195)

### Validating LDA2
set.seed(7)
Valid.pred3 <- predict(fit.lda2, Valid.donr[, c(1:7, 9:9, 11:11, 18:19, 21:22)], type="prob")
summary(Valid.pred3)
head(Valid.pred3[,2])

# calculate ordered profit function using average donation = $14.50 and mailing cost = $2
profit.lda2 <- cumsum(14.5*c.valid[order(Valid.pred3[,2], decreasing=T)]-2)
n.mail.valid3 <- which.max(profit.lda2) # number of mailings that maximizes profits
plot(profit.lda2, xlab = "Number of Mailings Sent", col="darkblue",
     ylab = "Profits",
     main="LDA2 Model - Profits Change per Number of Mailings") # see how profits change as more mailings are made
points(n.mail.valid3, max(profit.lda2), pch=19, col="red")
c(n.mail.valid3, max(profit.lda2)) # report number of mailings and maximum profit
# 1425 11302

### Validating QDA2
set.seed(7)
Valid.pred4 <- predict(fit.qda2, Valid.donr[, c(1:7, 9:9, 11:11, 18:19, 21:22)], type="prob")
summary(Valid.pred4)
head(Valid.pred4[,2])

# calculate ordered profit function using average donation = $14.50 and mailing cost = $2
profit.qda2 <- cumsum(14.5*c.valid[order(Valid.pred4[,2], decreasing=T)]-2)
n.mail.valid4 <- which.max(profit.qda2) # number of mailings that maximizes profits
plot(profit.qda2, xlab = "Number of Mailings Sent", col="darkblue",
     ylab = "Profits",
     main="QDA2 Model - Profits Change per Number of Mailings") # see how profits change as more mailings are made
points(n.mail.valid4, max(profit.qda2), pch=19, col="red")
c(n.mail.valid4, max(profit.qda2)) # report number of mailings and maximum profit
# 1403.0 11186.5

### Validating GLMNET2
set.seed(7)
Valid.pred5 <- predict(fit.glmnet2, Valid.donr[, c(1:7, 9:9, 11:11, 18:19, 21:22)], type="prob")
summary(Valid.pred5)
head(Valid.pred5[,2])

# calculate ordered profit function using average donation = $14.50 and mailing cost = $2
profit.glmnet2 <- cumsum(14.5*c.valid[order(Valid.pred5[,2], decreasing=T)]-2)
n.mail.valid5 <- which.max(profit.glmnet2) # number of mailings that maximizes profits
plot(profit.glmnet2, xlab = "Number of Mailings Sent", col="darkblue",
     ylab = "Profits",
     main="GLMNET2 Model - Profits Change per Number of Mailings") # see how profits change as more mailings are made
points(n.mail.valid5, max(profit.glmnet2), pch=19, col="red")
c(n.mail.valid5, max(profit.glmnet2)) # report number of mailings and maximum profit
# 1433 11315

### Validating CART2
set.seed(7)
Valid.pred6 <- predict(fit.cart2, Valid.donr[, c(1:7, 9:9, 11:11, 18:19, 21:22)], type="prob")
summary(Valid.pred6)
head(Valid.pred6[,2])

# calculate ordered profit function using average donation = $14.50 and mailing cost = $2
profit.cart2 <- cumsum(14.5*c.valid[order(Valid.pred6[,2], decreasing=T)]-2)
n.mail.valid6 <- which.max(profit.cart2) # number of mailings that maximizes profits
plot(profit.cart2, xlab = "Number of Mailings Sent", col="darkblue",
     ylab = "Profits",
     main="CART2 Model - Profits Change per Number of Mailings") # see how profits change as more mailings are made
points(n.mail.valid6, max(profit.cart2), pch=19, col="red")
c(n.mail.valid6, max(profit.cart2)) # report number of mailings and maximum profit
# 1958.0 10569.5

### Validating KNN2
set.seed(7)
Valid.pred7 <- predict(fit.knn2, Valid.donr[, c(1:7, 9:9, 11:11, 18:19, 21:22)], type="prob")
summary(Valid.pred7)
head(Valid.pred7[,2])

# calculate ordered profit function using average donation = $14.50 and mailing cost = $2
profit.knn2 <- cumsum(14.5*c.valid[order(Valid.pred7[,2], decreasing=T)]-2)
n.mail.valid7 <- which.max(profit.knn2) # number of mailings that maximizes profits
plot(profit.knn2, xlab = "Number of Mailings Sent", col="darkblue",
     ylab = "Profits",
     main="KNN2 Model - Profits Change per Number of Mailings") # see how profits change as more mailings are made
points(n.mail.valid7, max(profit.knn2), pch=19, col="red")
c(n.mail.valid7, max(profit.knn2)) # report number of mailings and maximum profit
# 1294 11477

####################################
### Validating the Best 2 Models ###
####################################

# SVM Model
set.seed(7)
Valid.pred <- predict(fit.svm2, Valid.donr[, c(1:7, 9:9, 11:11, 18:19, 21:22)], type="prob")
summary(Valid.pred)
head(Valid.pred[,2])

# calculate ordered profit function using average donation = $14.50 and mailing cost = $2
profit.svm2 <- cumsum(14.5*c.valid[order(Valid.pred[,2], decreasing=T)]-2)
n.mail.valid <- which.max(profit.svm2) # number of mailings that maximizes profits
plot(profit.svm2, xlab = "Number of Mailings Sent", col="darkblue",
     ylab = "Profits",
     main="SVM2 Model - Profits Change per Number of Mailings") # see how profits change as more mailings are made
points(n.mail.valid, max(profit.svm2), pch=19, col="red")
c(n.mail.valid, max(profit.svm2)) # report number of mailings and maximum profit
# 1396.0 11635.5

cutoff.svm2 <- sort(Valid.pred[,2], decreasing=T)[n.mail.valid+1] # set cutoff based on n.mail.valid
chat.valid.svm2 <- ifelse(Valid.pred[,2]>cutoff.svm2, 1, 0) # mail to everyone above the cutoff
table(chat.valid.svm2, c.valid) # classification table
#               c.valid
#chat.valid.log1   0   1
#              0 618   4
#              1 401 995
# check n.mail.valid = 401+995 = 1,396
# check profit = 14.5*995-2*1396 = 11,635.5

# Results
# n.mail Profit  Model
# 1431   11333.5 GLM2
# 1425   11302   LDA2
# 1403   11186.5 QDA2
# 1433   11315   GLMNET2
# 1958   10569.5 CART2
# 1294   11477   KNN2
# 1396   11635.5 SVM2
# 1260   11719   NNET2 <-- Best Model
# 1329   11624.5 LDA1
# 1291   11642.5 Log1

# MSE
mse(Valid.donr$donr,Valid.pred[[2]]) # 0.09207063 (with incm = 0.08879566 and with incm and rgip = 0.09020772)

# Multiple R-Squared
R2.1 <- 1-(sum((Valid.donr$donr-Valid.pred[[2]])^2)/sum((Valid.donr$donr-mean(Valid.donr$donr))^2))
R2.1 # 0.6316813 (with incm = 0.6447825 and with incm and rgip = 0.6391337)

# NNET Model
set.seed(7)
Valid.pred2 <- predict(fit.nnet2, Valid.donr[, c(1:7, 9:9, 11:11, 18:19, 21:22)], type="prob")
summary(Valid.pred2)
head(Valid.pred2[,2])

# calculate ordered profit function using average donation = $14.50 and mailing cost = $2
profit.nnet2 <- cumsum(14.5*c.valid[order(Valid.pred2[,2], decreasing=T)]-2)
n.mail.valid2 <- which.max(profit.nnet2) # number of mailings that maximizes profits
plot(profit.nnet2, xlab = "Number of Mailings Sent", col="darkblue",
     ylab = "Profits",
     main="NNET2 Model - Profits Change per Number of Mailings") # see how profits change as more mailings are made
points(n.mail.valid2, max(profit.nnet2), pch=19, col="red")
c(n.mail.valid2, max(profit.nnet2)) # report number of mailings and maximum profit
# 1260   11719

cutoff.nnet2 <- sort(Valid.pred2[,2], decreasing=T)[n.mail.valid2+1] # set cutoff based on n.mail.valid
chat.valid.nnet2 <- ifelse(Valid.pred2[,2]>cutoff.nnet2, 1, 0) # mail to everyone above the cutoff
table(chat.valid.nnet2, c.valid) # classification table
#               c.valid
#chat.valid.log1   0   1
#              0 741  17
#              1 278 982
# check n.mail.valid = 278+982 = 1,260
# check profit = 14.5*982-2*1260 = 11,719

# Results
# n.mail Profit  Model
# 1431   11333.5 GLM2
# 1425   11302   LDA2
# 1403   11186.5 QDA2
# 1433   11315   GLMNET2
# 1958   10569.5 CART2
# 1294   11477   KNN2
# 1396   11635.5 SVM2
# 1260   11719   NNET2 <-- Best Model
# 1329   11624.5 LDA1
# 1291   11642.5 Log1

# MSE
mse(Valid.donr$donr,Valid.pred2[[2]]) # 0.08592644 (with incm and rgip = 0.08435041)

# Multiple R-Squared
R2.2 <- 1-(sum((Valid.donr$donr-Valid.pred2[[2]])^2)/sum((Valid.donr$donr-mean(Valid.donr$donr))^2))
R2.2 # 0.6562605 (with incm and rgip = 0.6625652)


##############################################
####### Test Dataset - Classification ########
##############################################

head(data.test.std)

# Decision Tree Variable importance
# chld hinc home reg2 wrat tdon tlag 
#  53   13   11   11    7    4    2 
head(data.test.std[, c(1:7, 9:9, 11:11, 18:19, 21:21)])

set.seed(7)
post.test <- predict(fit.nnet2, data.test.std[, c(1:7, 9:9, 11:11, 18:19, 21:21)],
                     type="prob") # post probs for test data

# Oversampling adjustment for calculating number of mailings for test set
n.mail.valid <- which.max(profit.nnet2)
tr.rate <- .1 # typical response rate is .1
vr.rate <- .5 # whereas validation response rate is .5
adj.test.1 <- (n.mail.valid/n.valid.c)/(vr.rate/tr.rate) # adjustment for mail yes
adj.test.0 <- ((n.valid.c-n.mail.valid)/n.valid.c)/((1-vr.rate)/(1-tr.rate)) # adjustment for mail no
adj.test <- adj.test.1/(adj.test.1+adj.test.0) # scale into a proportion
n.mail.test <- round(n.test*adj.test, 0) # calculate number of mailings for test set

cutoff.test <- sort(post.test[[2]], decreasing=T)[n.mail.test+1] # set cutoff based on n.mail.test
chat.test <- ifelse(post.test[[2]]>cutoff.test, 1, 0) # mail to everyone above the cutoff
table(chat.test)
#    0    1 
# 1694  313
# based on this model we'll mail to the 313 highest posterior probabilities

head(chat.test)
tail(chat.test)

#########################################################################################
### Zero Inflated Poisson and Zero Inflated Negative Binomial Models for Bonus Points ###
#########################################################################################

# Recode Class (donr) back to 0 and 1
head(Train.donr)
ZIP.donr <- Train.donr
head(ZIP.donr)
required.labels <- ZIP.donr$donr
recoded.labels <- car::recode(required.labels, "'No'=0; 'Yes'=1")
ZIP.donr$donr <- as.numeric(as.character(recoded.labels))
head(ZIP.donr$donr)

# Model Zero Inflated Poisson Regression (ZIP) - Training Set
set.seed(7)
summary(train.zip <- zeroinfl(donr~chld+hinc+home+reg1+reg2+reg3+reg4+wrat+tdon+tlag+incm+rgip,
                              data=ZIP.donr))

# Validating train.zip
set.seed(7)
Valid.ZIP <- predict(train.zip, Valid.donr[, c(1:7, 9:9, 11:11, 18:19, 21:22)], type="prob")
summary(Valid.ZIP)
head(Valid.ZIP[,2])

# calculate ordered profit function using average donation = $14.50 and mailing cost = $2
profit.ZIP <- cumsum(14.5*c.valid[order(Valid.ZIP[,2], decreasing=T)]-2)
n.mail.valid.zip <- which.max(profit.ZIP) # number of mailings that maximizes profits
plot(profit.ZIP, xlab = "Number of Mailings Sent", col="darkblue",
     ylab = "Profits",
     main="ZIP Model - Profits Change per Number of Mailings") # see how profits change as more mailings are made
points(n.mail.valid, max(profit.ZIP), pch=19, col="red")
c(n.mail.valid.zip, max(profit.ZIP)) # report number of mailings and maximum profit
# 1340.0 11428.5 with incm and rgip

cutoff.zip <- sort(Valid.ZIP[,2], decreasing=T)[n.mail.valid.zip+1] # set cutoff based on n.mail.valid
chat.valid.zip <- ifelse(Valid.ZIP[,2]>cutoff.zip, 1, 0) # mail to everyone above the cutoff
table(chat.valid.zip, c.valid) # classification table
#               c.valid
#chat.valid.log1   0   1
#              0 652  26
#              1 367 973
# check n.mail.valid = 367+973 = 1,340
# check profit = 14.5*973-2*1340 = 11,428.5

# Results
# n.mail Profit  Model
# 1431   11333.5 GLM2
# 1425   11302   LDA2
# 1403   11186.5 QDA2
# 1433   11315   GLMNET2
# 1958   10569.5 CART2
# 1294   11477   KNN2
# 1396   11635.5 SVM2
# 1260   11719   NNET2 <-- Best Model
# 1340.0 11428.5 ZIP
# 1329   11624.5 LDA1
# 1291   11642.5 Log1

# MSE
mse(Valid.donr$donr,Valid.ZIP[[2]]) # with incm and rgip = 0.3393678

# Multiple R-Squared
R2.ZIP <- 1-(sum((Valid.donr$donr-Valid.ZIP[[2]])^2)/sum((Valid.donr$donr-mean(Valid.donr$donr))^2))
R2.ZIP # with incm and rgip = -0.3576047


### Model Zero Inflated Negative Binomial (ZINB) ###
# Recode Class (donr) back to 0 and 1
head(Train.donr)
ZINB.donr <- Train.donr
head(ZINB.donr)
required.labels <- ZINB.donr$donr
recoded.labels <- car::recode(required.labels, "'No'=0; 'Yes'=1")
ZINB.donr$donr <- as.numeric(as.character(recoded.labels))
head(ZINB.donr$donr)

# Model Zero Inflated Negative Binomial (ZINB) - Training Set
set.seed(7)
summary(train.zinb <- zeroinfl(donr~chld+hinc+home+reg1+reg2+reg3+reg4+wrat+tdon+tlag+incm+rgip,
                              data=ZINB.donr, dist = "negbin", EM = TRUE))

# Validating train.zinb
set.seed(7)
Valid.ZINB <- predict(train.zinb, Valid.donr[, c(1:7, 9:9, 11:11, 18:19, 21:22)], type="prob")
summary(Valid.ZINB)
head(Valid.ZINB[,2])

# calculate ordered profit function using average donation = $14.50 and mailing cost = $2
profit.ZINB <- cumsum(14.5*c.valid[order(Valid.ZINB[,2], decreasing=T)]-2)
n.mail.valid.zinb <- which.max(profit.ZINB) # number of mailings that maximizes profits
plot(profit.ZINB, xlab = "Number of Mailings Sent", col="darkblue",
     ylab = "Profits",
     main="ZINB Model - Profits Change per Number of Mailings") # see how profits change as more mailings are made
points(n.mail.valid, max(profit.ZINB), pch=19, col="red")
c(n.mail.valid.zinb, max(profit.ZINB)) # report number of mailings and maximum profit
# 1340.0 11428.5 with incm and rgip

cutoff.zinb <- sort(Valid.ZINB[,2], decreasing=T)[n.mail.valid.zinb+1] # set cutoff based on n.mail.valid
chat.valid.zinb <- ifelse(Valid.ZINB[,2]>cutoff.zinb, 1, 0) # mail to everyone above the cutoff
table(chat.valid.zinb, c.valid) # classification table
#               c.valid
#chat.valid.log1   0   1
#              0 652  26
#              1 367 973
# check n.mail.valid = 367+973 = 1,340
# check profit = 14.5*973-2*1340 = 11,428.5

# Results
# n.mail Profit  Model
# 1431   11333.5 GLM2
# 1425   11302   LDA2
# 1403   11186.5 QDA2
# 1433   11315   GLMNET2
# 1958   10569.5 CART2
# 1294   11477   KNN2
# 1396   11635.5 SVM2
# 1260   11719   NNET2 <-- Best Model
# 1340.0 11428.5 ZIP
# 1340   11428.5 ZINB
# 1329   11624.5 LDA1
# 1291   11642.5 Log1

# MSE
mse(Valid.donr$donr,Valid.ZINB[[2]]) # with incm and rgip = 0.3393675

# Multiple R-Squared
R2.ZINB <- 1-(sum((Valid.donr$donr-Valid.ZINB[[2]])^2)/sum((Valid.donr$donr-mean(Valid.donr$donr))^2))
R2.ZINB # with incm and rgip = -0.3576035

################################
##### PREDICTION MODELING ######
################################

# Removing the ID, donr and part columns
head(data.train.std.y)
Train.damt <- data.train.std.y # data.train
Train.damt$ID <- NULL
Train.damt$donr <- NULL
Train.damt$part <- NULL
head(Train.damt)

######################################################################
### Models with variables selected by a Decision Tree - Prediction ###
######################################################################
head(data.valid.std.y[, c(1:7, 9:9, 11:11, 18:19, 21:22)])
# Decision Tree Variable importance
# chld hinc home reg2 wrat tdon tlag 
#  53   13   11   11    7    4    2 

# 10-fold cross validation with 3 repeats
trainControl.y <- trainControl(method="repeatedcv", summaryFunction=defaultSummary,
                             number=10, repeats=3, classProbs=FALSE)

set.seed(7)
fit.mlr <- train(damt~chld+hinc+home+reg1+reg2+reg3+reg4+wrat+tdon+tlag+incm+rgip,
                 data=Train.damt, method="lm", trControl=trainControl.y)  # MLR

pred.valid.mlr <- predict(fit.mlr, newdata = data.valid.std.y[, c(1:7, 9:9, 11:11, 18:19, 21:22)]) # validation predictions
mean((y.valid - pred.valid.mlr)^2) # mean prediction error
# 1.867433
# 3.237925
sd((y.valid - pred.valid.mlr)^2)/sqrt(n.valid.y) # std error
# 0.1696498
# 0.2251062

set.seed(7)
fit.las <- train(damt~chld+hinc+home+reg1+reg2+reg3+reg4+wrat+tdon+tlag+incm+rgip,
                  data=Train.damt, method="lasso", trControl=trainControl.y) # LASSO

pred.valid.las <- predict(fit.las, newdata = data.valid.std.y[, c(1:7, 9:9, 11:11, 18:19, 21:22)]) # validation predictions
mean((y.valid - pred.valid.las)^2) # mean prediction error
# 3.240217
sd((y.valid - pred.valid.las)^2)/sqrt(n.valid.y) # std error
# 0.2260751

set.seed(7)
fit.gbm <- train(damt~chld+hinc+home+reg1+reg2+reg3+reg4+wrat+tdon+tlag+incm+rgip,
                  data=Train.damt, method="gbm", trControl=trainControl.y) # GBM

pred.valid.gbm <- predict(fit.gbm, newdata = data.valid.std.y[, c(1:7, 9:9, 11:11, 18:19, 21:22)]) # validation predictions
mean((y.valid - pred.valid.gbm)^2) # mean prediction error
# 3.087681
sd((y.valid - pred.valid.gbm)^2)/sqrt(n.valid.y) # std error
# 0.2202538

set.seed(7)
fit.svm.y <- train(damt~chld+hinc+home+reg1+reg2+reg3+reg4+wrat+tdon+tlag+incm+rgip,
                  data=Train.damt, method="svmRadial", trControl=trainControl.y) # SVM

pred.valid.svm.y <- predict(fit.svm.y, newdata = data.valid.std.y[, c(1:7, 9:9, 11:11, 18:19, 21:22)]) # validation predictions
mean((y.valid - pred.valid.svm.y)^2) # mean prediction error
# 3.150872
sd((y.valid - pred.valid.svm.y)^2)/sqrt(n.valid.y) # std error
# 0.2256835

set.seed(7)
fit.nnet.y <- train(damt~chld+hinc+home+reg1+reg2+reg3+reg4+wrat+tdon+tlag+incm+rgip,
                    data=Train.damt, method="nnet", linout=T, hidden=10,
                    threshold=0.01, trControl=trainControl.y)  # ANN
print(fit.nnet.y)

set.seed(7)
fit.gam <- train(damt~chld+hinc+home+reg1+reg2+reg3+reg4+wrat+tdon+tlag+incm+rgip,
                     data=Train.damt, method="gam", trControl=trainControl.y)  # GAM

# Compare algorithms
results3 <- resamples(list(MLR=fit.mlr, LASSO=fit.las, GBM=fit.gbm,
                           SVM.y=fit.svm.y, ANN.y=fit.nnet.y, GAM=fit.gam))
summary(results3)
dotplot(results3, main=" RMSE and R-Squared Chart - Decision Tree Variables Models")

################################################################################
### Models with All Variables, including the Engineered feature - Prediction ###
################################################################################

# Methods available
names(getModelInfo())

# Data
head(data.valid.std.y)

# 10-fold cross validation with 3 repeats
trainControl.y <- trainControl(method="repeatedcv", summaryFunction=defaultSummary,
                               number=10, repeats=3, classProbs=FALSE)

set.seed(7)
fit.mlr.a <- train(damt~., data=Train.damt, method="lm", trControl=trainControl.y)  # MLR

pred.valid.mlr.a <- predict(fit.mlr.a, newdata = data.valid.std.y) # validation predictions
mean((y.valid - pred.valid.mlr.a)^2) # mean prediction error
# 1.852708541
sd((y.valid - pred.valid.mlr.a)^2)/sqrt(n.valid.y) # std error
# 0.1685098032

set.seed(7)
fit.las.a <- train(damt~., data=Train.damt, method="lasso", trControl=trainControl.y) # LASSO

pred.valid.las.a <- predict(fit.las.a, newdata = data.valid.std.y) # validation predictions
mean((y.valid - pred.valid.las.a)^2) # mean prediction error
# 1.851424841
sd((y.valid - pred.valid.las.a)^2)/sqrt(n.valid.y) # std error
# 0.1684351255

set.seed(7)
fit.gbm.a <- train(damt~., data=Train.damt, method="gbm", trControl=trainControl.y) # GBM

pred.valid.gbm.a <- predict(fit.gbm.a, newdata = data.valid.std.y) # validation predictions
mean((y.valid - pred.valid.gbm.a)^2) # mean prediction error
# 1.375936101 with all variables
sd((y.valid - pred.valid.gbm.a)^2)/sqrt(n.valid.y) # std error
# 0.1587024851 with all variables

set.seed(7)
fit.svm.y.a <- train(damt~., data=Train.damt, method="svmRadial", trControl=trainControl.y) # SVM

pred.valid.svm.y.a <- predict(fit.svm.y.a, newdata = data.valid.std.y) # validation predictions
mean((y.valid - pred.valid.svm.y.a)^2) # mean prediction error
# 1.630582986
sd((y.valid - pred.valid.svm.y.a)^2)/sqrt(n.valid.y) # std error
# 0.1765667003

set.seed(7)
fit.nnet.y.a <- train(damt~., data=Train.damt, method="nnet", linout=T, hidden=10,
                    threshold=0.01, trControl=trainControl.y)  # ANN
print(fit.nnet.y.a)

pred.valid.nnet.y.a <- predict(fit.nnet.y.a, newdata = data.valid.std.y) # validation predictions
mean((y.valid - pred.valid.nnet.y.a)^2) # mean prediction error
# 1.642647375
sd((y.valid - pred.valid.nnet.y.a)^2)/sqrt(n.valid.y) # std error
# 0.1753319806

set.seed(7)
fit.gam.a <- train(damt~., data=Train.damt, method="gam", trControl=trainControl.y)  # GAM

pred.valid.gam.y.a <- predict(fit.gam.a, newdata = data.valid.std.y) # validation predictions
mean((y.valid - pred.valid.gam.y.a)^2) # mean prediction error
# 1.588451527
sd((y.valid - pred.valid.gam.y.a)^2)/sqrt(n.valid.y) # std error
# 0.16443871

# Compare algorithms
results4 <- resamples(list(MLR=fit.mlr.a, LASSO=fit.las.a, GBM=fit.gbm.a,
                           SVM.y=fit.svm.y.a, ANN.y=fit.nnet.y.a, GAM=fit.gam.a))
summary(results4)
dotplot(results4, main=" RMSE and R-Squared Chart - All Variables Models")

##############################
########## Results ###########
##############################
# MPE          Model
# 1.852708541  fit.mlr.a (MLR)
# 1.851424841  fit.las.a (LASSO)
# 1.375936101  fit.gbm.a (GBM) <-- Best Model
# 1.630582986  fit.svm.y.a (SVM)
# 1.642647375  fit.nnet.y.a (ANN)
# 1.588451527  fit.gam.a (GAM)

# select the GBM model since it has minimum mean prediction error in the validation sample
yhat.test <- predict(fit.gbm.a, newdata = data.test.std) # test predictions
str(yhat.test)
head(yhat.test)

#########################################################################
### Using flexmix to model a Latent Class Regression for Bonus Points ###
#########################################################################
# Model 1
set.seed(7)
CW_Reg1 <- flexmix(damt~., data=Train.damt, k=1)

CW_Reg1
summary(CW_Reg1)
#        prior size post>0 ratio
# Comp.1     1 1995   1995     1

# 'log Lik.' -3299.311 (df=23)
# AIC: 6644.623   BIC: 6773.386 

plot(CW_Reg1)

# Parameter Coefficients
parameters(CW_Reg1, component=1)
#                       Comp.1
# coef.(Intercept) 14.180520277
# coef.reg1        -0.045142960
# coef.reg2        -0.080704839
# coef.reg3         0.323030017
# coef.reg4         0.631764009
# coef.home         0.240683624
# coef.chld        -0.607979356
# coef.hinc         0.498425361
# coef.genf        -0.059433575
# coef.wrat         0.003994676
# coef.avhv        -0.052560867
# coef.incm         0.400620735
# coef.inca         0.081387343
# coef.plow         0.394588484
# coef.npro         0.151000302
# coef.tgif         0.050412987
# coef.lgif        -0.062602358
# coef.rgif         0.518606696
# coef.tdon         0.072005520
# coef.tlag         0.021498607
# coef.agif         0.669995240
# coef.rgip         0.027181856
# sigma             1.271713780

result.1 <- predict(CW_Reg1)
head(result.1)

# Coefficients and Metrics - Model 1
Mod1 <- lm(Train.damt$damt~result.1[[1]])
anova(Mod1)
summary(Mod1) # R-Squared = 0.5731163 and RSE = 1.265

pred.valid.CW1.y <- predict(CW_Reg1, newdata = data.valid.std.y) # validation predictions
str(pred.valid.CW1.y)
mean((y.valid - pred.valid.CW1.y[[1]])^2) # mean prediction error
# 1.852709
sd((y.valid - pred.valid.CW1.y[[1]])^2)/sqrt(n.valid.y) # std error
# 0.1685098

####################

# Model 2
set.seed(7)
CW_Reg2 <- flexmix(damt~., data=Train.damt, k=2)

CW_Reg2
summary(CW_Reg2)
#        prior size post>0 ratio
# Comp.1  0.77 1725   1965 0.878
# Comp.2  0.23  270   1995 0.135

# 'log Lik.' -3092.284 (df=47)
# AIC: 6278.568   BIC: 6541.693

plot(CW_Reg2)

# Parameter Coefficients Component 1
parameters(CW_Reg2, component=1)
#                       Comp.1
# coef.(Intercept) 13.91437522
# coef.reg1        -0.08517785
# coef.reg2        -0.11665631
# coef.reg3         0.27565160
# coef.reg4         0.58588213
# coef.home         0.31378906
# coef.chld        -0.57338372
# coef.hinc         0.52034805
# coef.genf        -0.02089789
# coef.wrat         0.07788570
# coef.avhv        -0.01755516
# coef.incm         0.36164294
# coef.inca         0.10204340
# coef.plow         0.42109531
# coef.npro         0.09295502
# coef.tgif         0.11361870
# coef.lgif         0.09680915
# coef.rgif         0.76601000
# coef.tdon         0.07798282
# coef.tlag         0.01829250
# coef.agif         0.60278946
# coef.rgip         0.05078731
# sigma             0.81444333

# Parameter Coefficients Component 2
parameters(CW_Reg2, component=2)
#                        Comp.2
# coef.(Intercept) 15.52087510
# coef.reg1         0.10582794
# coef.reg2        -0.00263079
# coef.reg3         0.44666816
# coef.reg4         0.74277544
# coef.home        -0.37200183
# coef.chld        -0.62209247
# coef.hinc         0.25543241
# coef.genf        -0.17590969
# coef.wrat        -0.15318476
# coef.avhv        -0.02053036
# coef.incm         0.51104231
# coef.inca        -0.07179708
# coef.plow         0.32565508
# coef.npro         0.35568648
# coef.tgif        -0.18152391
# coef.lgif         0.04261066
# coef.rgif         0.14980340
# coef.tdon         0.11764964
# coef.tlag         0.01005010
# coef.agif         0.37248997
# coef.rgip         0.02456853
# sigma             1.61683732

# Price Prediction
pred2 <- predict(CW_Reg2, aggregate = TRUE)[[1]][,1]
head(pred2)
str(pred2)

pred.valid.CW2.y <- predict(CW_Reg2, newdata = data.valid.std.y) # validation predictions
str(pred.valid.CW1.y)
mean((y.valid - pred.valid.CW2.y[[1]])^2) # mean prediction error
# 1.9528
sd((y.valid - pred.valid.CW2.y[[1]])^2)/sqrt(n.valid.y) # std error
# 0.1929624

# looking at clusters and the damt prediction
pred <- predict(CW_Reg2)
clust <- clusters(CW_Reg2,Train.damt)
result <- cbind(Train.damt,data.frame(pred),data.frame(clust),data.frame(pred2))
head(result)

# Coefficients and Metrics - Model 2
Mod2 <- lm(result[[22]]~result[[25]])
anova(Mod2)
summary(Mod2) # R-Squared = 0.2185  and RSE = 1.712

##############################
########## Results ###########
##############################
# MPE          Model
# 1.852708541  fit.mlr.a (MLR)
# 1.851424841  fit.las.a (LASSO)
# 1.375936101  fit.gbm.a (GBM) <-- Best Model
# 1.630582986  fit.svm.y.a (SVM)
# 1.642647375  fit.nnet.y.a (ANN)
# 1.588451527  fit.gam.a (GAM)
# 1.852709000  CW_Reg1 (Latent Class Regression k=1)
# 1.952800000  CW_Reg2 (Latent Class Regression k=2)

############################################################
### FINAL RESULTS - Classification and Regression Models ###
############################################################

# Save final results for both classification and regression
length(chat.test) # check length = 2007
length(yhat.test) # check length = 2007
chat.test[1:10] # check this consists of 0s and 1s
yhat.test[1:10] # check this consists of plausible predictions of damt

ip <- data.frame(chat=chat.test, yhat=yhat.test) # data frame with two variables: chat and yhat
head(ip)
write.csv(ip, file="D:/Charity_Project.csv", row.names=FALSE)



