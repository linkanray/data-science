rm(list=ls())
dev.off()

#load(".RData")

# install package
install.packages("kernlab")
install.packages("caret")
install.packages("e1071")
install.packages("ggplot2")
#install.packages("randomForest")
#install.packages("readr")
#install.packages("h2o")

# load library to environment
library(kernlab)
library(caret)
library(e1071)
library(ggplot2)
#library(randomForest)
#library(gmodels) #Added as needed for CrossTable
#library(readr)
#library(h2o) #For h2o functions in neural networks

## data preparation
train_data = read.csv("train.csv", header = TRUE)                 # read the csv file
test_data = read.csv("test.csv", header = TRUE)

set.seed(0)

train.no.label = subset(train_data, select = -c(label))            # remove the response variable from predictors

## apply PCA: principal components analysis
pca.st.time = proc.time()                                         
train.pca = prcomp(train.no.label)								                # do PCA for dimension reduction
pca.end.time = pca.st.time - proc.time()
pca.end.time

par(mar = rep(2, 4))                                              # plot Principal component and see variance corresponding to each component
plot(train.pca, main="Change in explained variance")


pca.train.var =  train.pca$sdev^2                                 # variance of each principal component
prop.pca.train.var = pca.train.var/sum(pca.train.var)             # proportion of variance
cumsum(prop.pca.train.var)[100]*100                               # 91.5% variance is explained by 100 principal components and 96.7% 
                                                                  # by 200 principal component


plot(cumsum(prop.pca.train.var)*100,                              # % of variance explained by principal components can be plotted
     xlab = "Principal Component",
     ylab = "Percentage",
     type = "l")
title("Percentage of Variance Explained by PC")


train = data.frame(train_data$label, train.pca$x)                  # combine the principal components with label
train.100.pca = train[,1:100]                                      # extract first 65 PCA
test.pca = predict(train.pca, newdata = test_data)                 
test.pca = as.data.frame(test.pca)								                # converting test data set into PCA
test.100.pca = test.pca[,1:100]                                    # extract first 65 PCA for test data



ksvm.rbf.model = ksvm(train_data.label ~ ., data = train.100.pca,
                       kernel = "rbfdot")


ksvm.rbf.predict = predict(ksvm.rbf.model, test.100.pca)          # predict accuracy with non-linear model on test data with 100 variables

for(i in 1:length(ksvm.rbf.predict)){
  if(ksvm.rbf.predict[i] < 0){
    ksvm.rbf.predict[i] <- 1
  }
  else if(ksvm.rbf.predict[i] > 9){
    ksvm.rbf.predict[i] <- 9
  }
}
rm(i)

predictTest <- round(ksvm.rbf.predict, digits = 0)

submit <- data.frame(ImageId = seq(1,nrow(test_data)),
                     Label = predictTest)
write.csv(submit, file = "svm_pca.csv", row.names=F)





