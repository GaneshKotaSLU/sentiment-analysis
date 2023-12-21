library(ggplot2) ## needed for ggplot functions
library(dplyr) ## needed for the mutate() function
library(caret)
library(tidyverse)
library(readxl)
library(plyr)
library(tidyr)
library(janitor)
library(GGally)
library(MASS)
library(randomForest)
library(gbm)
library(doParallel)

ds.obj <- read.csv("Wish_dataset.csv",header = T)

##total number of rows and columns in the dataset
nrow(ds.obj)
ncol(ds.obj)

##Changing the Col names of first two columns
ds.obj <- rename(ds.obj,c("title" = "French_Title",
                          "title_orig" = "English_Title"))
colnames(ds.obj)
summary(ds.obj)

####### Beginning of Data Cleaning process ##########

##Step 1 -  Cleaning Column names
ds.obj <- clean_names(ds.obj)

##Step 2 - remove rows and columns if completely empty
ds.obj <- remove_empty(ds.obj,which = c("rows","cols"),
                       quiet = FALSE)

##There are no completely empty rows or completely empty columns to remove

## Step 3 - Removing duplicate rows of data in R
ds.obj <- distinct(ds.obj)
nrow(ds.obj)

##Step 4 - Deleting all the rows with missing values
ds.obj <- ds.obj[complete.cases(ds.obj), ]
nrow(ds.obj)

##Step 5 - Finding if there are any 'NA' values and deleting them

##Below command gives the total number of 'NA' values true and false
table(is.na(ds.obj))  

##below command gives number of 'NA' row wise
rowSums(is.na(ds.obj))

##below command gives the number of 'NA' values column wise
sapply(ds.obj,function(x) sum(is.na(x)))

## Finding how many missing values are there and delete them
ds.obj <- na.omit(ds.obj)
nrow(ds.obj)

##Running Numerical Summaries
summary(ds.obj)

##Creating a new variable to capture revenue of each product after discount
ds.obj$revenue <- ds.obj$price * ds.obj$units_sold

ggpairs(ds.obj, columns = c(3,4,(6:18),22,(24:28),34,35,37,44))

################ The analysis starts here  #######################

cl <- makePSOCKcluster(5)
registerDoParallel(cl)

## identifying any predictors that have zero or near-zero variance
nzv <- nearZeroVar(ds.obj, saveMetrics= TRUE)
nzv 

## product_factor <- (factor(ds.obj$english_title))
## prod_ids <- as.integer(product_factor)
## ds.obj$product_id <- prod_ids

ds.obj$product_significance <- ifelse(ds.obj$rating >=3 & ds.obj$revenue >=20000,
                                      "POSITIVE", "NEGATIVE")
#ds.obj$product_significance <- ifelse(ds.obj$product_significance == 0, "FALSE", "TRUE")
ds.obj$product_significance <- as.factor(ds.obj$product_significance)

#ds.obj$product_significance <- as.factor(ifelse(ds.obj$product_significance == 0, 0, 1))
ncol(ds.obj)

ds.obj <- dplyr::select(ds.obj,3,4,6,7,18,20,21,22,24,25,26,27,34,35,45)
str(ds.obj)

ds.obj <- ds.obj[-c(1,3,6,7)]  ##removing the redundancy variables
str(ds.obj)

set.seed(1984)
trainIndices <- createDataPartition(ds.obj$product_significance, ## indicate which var. is outcome
                                    p = 0.8, # indicate proportion to use in training-testing
                                    list = FALSE, 
                                    times = 1)

training <- ds.obj[trainIndices,]
holdout <- ds.obj[-trainIndices,]

## centering and scaling as part of the pre-processing step
preProcValues <- preProcess(training, method = c("center", "scale"))

## Next, create the scaled+centered of the training+testing subset of the dataset
trainTransformed <- predict(preProcValues, training) 
## apply the same scaling and centering on the holdout set, too
holdoutTransformed <- predict(preProcValues, holdout)

#5-fold CV
fitControl <- trainControl(
  method = "repeatedcv", ## perform repeated k-fold CV
  number = 5,
  repeats = 3,
  classProbs = TRUE)


                ################## SVM with Linear Kernel Version ####################
grid <- expand.grid(C = c(0.01, 0.1, 10, 100, 1000))#,
#sigma = c(0.5, 1, 2, 3, 4))

svmlinearfit <- train(product_significance ~ .,
                      data = trainTransformed,
                      method = "svmLinear",
                      trControl = fitControl,
                      verbose = FALSE, ## setting this to TRUE or excluding this leads to a lot of output
                      tuneGrid = grid)

names(svmlinearfit)

## some plots
trellis.par.set(caretTheme())
plot(svmlinearfit)

## make predictions on the hold-out set
predvals <- predict(svmlinearfit, holdoutTransformed)

## compute the performance metrics
confusionMatrix(factor(predvals),holdoutTransformed$product_significance, positive = "POSITIVE")

## Rank the variables in terms of their importance
varImp(svmlinearfit)



                  ################## SVM with Radial Kernel Version ####################
grid <- expand.grid(C = c(0.01, 0.1, 10, 100, 1000),
                    sigma = c(0.5, 1, 2, 3, 4))

svmradialfit <- train(product_significance ~ .,
                      data = trainTransformed,
                      method = "svmRadial",
                      trControl = fitControl,
                      verbose = FALSE, ## setting this to TRUE or excluding this leads to a lot of output
                      tuneGrid = grid)

names(svmlinearfit)

## some plots
trellis.par.set(caretTheme())
plot(svmlinearfit)

## make predictions on the hold-out set
predvals <- predict(svmradialfit, holdoutTransformed)

## compute the performance metrics
confusionMatrix(factor(predvals),holdoutTransformed$product_significance, positive = "POSITIVE")

## Rank the variables in terms of their importance
varImp(svmradialfit)


                          ############### Logistic Regression ################
ctrl <- trainControl(method = "cv",
                     number = 5, 
                     verboseIter = TRUE)

# Train a logistic regression model using 5-fold cross-validation
model <- train(product_significance ~ ., 
               data = trainTransformed, 
               method = "glm", 
               family = "binomial", 
               trControl = ctrl)

# Make predictions on the test set
predictions <- predict(model, holdoutTransformed)

# Evaluate the performance using confusion matrix
confusionMatrix(predictions, holdoutTransformed$product_significance,positive = "POSITIVE")
varImp(model)
stopCluster(cl)
