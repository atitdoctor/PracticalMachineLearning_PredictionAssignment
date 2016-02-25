Practical Machine Learning - Recognizing Qualitative Activity
==========================================
#### Atit Doctor
##### February 25, 2016



### Introduction

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, we will use data recorded from accelerometers on the belt, forearm, arm, and dumbbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways.

The goal of this project is to predict the manner in which the participants did the exercise. This is the classe variable of the training set, which classifies the correct and incorrect outcomes into A, B, C, D, and E categories. This report describes how the model for the project was built, its cross validation, expected out of sample error calculation, and the choices made. It was used successfully to accurately predict all 20 different test cases on the Coursera website.



###  Data Loading and Cleaning


```r
# Set working directory
setwd("/Users/adoctor/Documents/PracticalMachineLearning_PredictionAssignment/")

# Read data
pmlTrain<-read.csv("pml-training.csv", header=T, na.strings=c("NA", "#DIV/0!"))
pmlTest<-read.csv("pml-testing.csv", header=T, na.string=c("NA", "#DIV/0!"))
```

Training data was partitioned and preprocessed using the code described below. In brief, all variables with at least one "NA" were excluded from the analysis. Variables related to time and user information were excluded for a total of 51 variables and 19622 class measurements. Same variables were mainteined in the test data set (Validation dataset) to be used for predicting the 20 test cases provided.


```r
## NA exclusion for all available variables
noNApmlTrain<-pmlTrain[, apply(pmlTrain, 2, function(x) !any(is.na(x)))] 
dim(noNApmlTrain)
```

```
## [1] 19622    60
```

```r
## variables with user information, time and undefined
cleanpmlTrain<-noNApmlTrain[,-c(1:8)]
dim(cleanpmlTrain)
```

```
## [1] 19622    52
```

```r
## 20 test cases provided clean info - Validation data set
cleanpmltest<-pmlTest[,names(cleanpmlTrain[,-52])]
dim(cleanpmltest)
```

```
## [1] 20 51
```



### Data Partitioning and Prediction Process

The cleaned downloaded data set was subset in order to generate a test set independent from the 20 cases provided set. Partitioning was performed to obtain a 75% training set and a 25% test set.


```r
#data cleaning
library(caret)
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```r
inTrain<-createDataPartition(y=cleanpmlTrain$classe, p=0.75,list=F)
training<-cleanpmlTrain[inTrain,] 
test<-cleanpmlTrain[-inTrain,] 
#Training and test set dimensions
dim(training)
```

```
## [1] 14718    52
```

```r
dim(test)
```

```
## [1] 4904   52
```



### Results

Random forest trees were generated for the training dataset using cross-validation. Then the generated algorithm was examnined under the partitioned training set to examine the accuracy and estimated error of prediction. By using 51 predictors for five classes using cross-validation at a 5-fold an accuracy of 99.2% with a 95% CI [0.989-0.994] was achieved accompanied by a Kappa value of 0.99.


```r
library(caret)
set.seed(13333)
fitControl2<-trainControl(method="cv", number=5, allowParallel=T, verbose=T)
rffit<-train(classe~.,data=training, method="rf", trControl=fitControl2, verbose=F)
```

```
## Loading required package: randomForest
```

```
## randomForest 4.6-12
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

```
## + Fold1: mtry= 2 
## - Fold1: mtry= 2 
## + Fold1: mtry=26 
## - Fold1: mtry=26 
## + Fold1: mtry=51 
## - Fold1: mtry=51 
## + Fold2: mtry= 2 
## - Fold2: mtry= 2 
## + Fold2: mtry=26 
## - Fold2: mtry=26 
## + Fold2: mtry=51 
## - Fold2: mtry=51 
## + Fold3: mtry= 2 
## - Fold3: mtry= 2 
## + Fold3: mtry=26 
## - Fold3: mtry=26 
## + Fold3: mtry=51 
## - Fold3: mtry=51 
## + Fold4: mtry= 2 
## - Fold4: mtry= 2 
## + Fold4: mtry=26 
## - Fold4: mtry=26 
## + Fold4: mtry=51 
## - Fold4: mtry=51 
## + Fold5: mtry= 2 
## - Fold5: mtry= 2 
## + Fold5: mtry=26 
## - Fold5: mtry=26 
## + Fold5: mtry=51 
## - Fold5: mtry=51 
## Aggregating results
## Selecting tuning parameters
## Fitting mtry = 26 on full training set
```

```r
predrf<-predict(rffit, newdata=test)
confusionMatrix(predrf, test$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1393    5    0    0    0
##          B    0  942    9    0    0
##          C    1    2  845   14    0
##          D    0    0    1  790    2
##          E    1    0    0    0  899
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9929         
##                  95% CI : (0.9901, 0.995)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.991          
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9986   0.9926   0.9883   0.9826   0.9978
## Specificity            0.9986   0.9977   0.9958   0.9993   0.9998
## Pos Pred Value         0.9964   0.9905   0.9803   0.9962   0.9989
## Neg Pred Value         0.9994   0.9982   0.9975   0.9966   0.9995
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2841   0.1921   0.1723   0.1611   0.1833
## Detection Prevalence   0.2851   0.1939   0.1758   0.1617   0.1835
## Balanced Accuracy      0.9986   0.9952   0.9921   0.9909   0.9988
```

```r
pred20<-predict(rffit, newdata=cleanpmltest)
# Output for the prediction of the 20 cases provided
pred20
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

A boosting algorithm was also run to confirm and be able to compare predictions. Data is not shown but the boosting approach presented less accuracy (96%) (Data not shown). However, when the predictions for the 20 test cases were compared match was same for both ran algorimths.


```r
fitControl2<-trainControl(method="cv", number=5, allowParallel=T, verbose=T)
gmbfit<-train(classe~.,data=training, method="gbm", trControl=fitControl2, verbose=F)
```

```
## Loading required package: gbm
```

```
## Loading required package: survival
```

```
## 
## Attaching package: 'survival'
```

```
## The following object is masked from 'package:caret':
## 
##     cluster
```

```
## Loading required package: splines
```

```
## Loading required package: parallel
```

```
## Loaded gbm 2.1.1
```

```
## Loading required package: plyr
```

```
## + Fold1: shrinkage=0.1, interaction.depth=1, n.minobsinnode=10, n.trees=150 
## - Fold1: shrinkage=0.1, interaction.depth=1, n.minobsinnode=10, n.trees=150 
## + Fold1: shrinkage=0.1, interaction.depth=2, n.minobsinnode=10, n.trees=150 
## - Fold1: shrinkage=0.1, interaction.depth=2, n.minobsinnode=10, n.trees=150 
## + Fold1: shrinkage=0.1, interaction.depth=3, n.minobsinnode=10, n.trees=150 
## - Fold1: shrinkage=0.1, interaction.depth=3, n.minobsinnode=10, n.trees=150 
## + Fold2: shrinkage=0.1, interaction.depth=1, n.minobsinnode=10, n.trees=150 
## - Fold2: shrinkage=0.1, interaction.depth=1, n.minobsinnode=10, n.trees=150 
## + Fold2: shrinkage=0.1, interaction.depth=2, n.minobsinnode=10, n.trees=150 
## - Fold2: shrinkage=0.1, interaction.depth=2, n.minobsinnode=10, n.trees=150 
## + Fold2: shrinkage=0.1, interaction.depth=3, n.minobsinnode=10, n.trees=150 
## - Fold2: shrinkage=0.1, interaction.depth=3, n.minobsinnode=10, n.trees=150 
## + Fold3: shrinkage=0.1, interaction.depth=1, n.minobsinnode=10, n.trees=150 
## - Fold3: shrinkage=0.1, interaction.depth=1, n.minobsinnode=10, n.trees=150 
## + Fold3: shrinkage=0.1, interaction.depth=2, n.minobsinnode=10, n.trees=150 
## - Fold3: shrinkage=0.1, interaction.depth=2, n.minobsinnode=10, n.trees=150 
## + Fold3: shrinkage=0.1, interaction.depth=3, n.minobsinnode=10, n.trees=150 
## - Fold3: shrinkage=0.1, interaction.depth=3, n.minobsinnode=10, n.trees=150 
## + Fold4: shrinkage=0.1, interaction.depth=1, n.minobsinnode=10, n.trees=150 
## - Fold4: shrinkage=0.1, interaction.depth=1, n.minobsinnode=10, n.trees=150 
## + Fold4: shrinkage=0.1, interaction.depth=2, n.minobsinnode=10, n.trees=150 
## - Fold4: shrinkage=0.1, interaction.depth=2, n.minobsinnode=10, n.trees=150 
## + Fold4: shrinkage=0.1, interaction.depth=3, n.minobsinnode=10, n.trees=150 
## - Fold4: shrinkage=0.1, interaction.depth=3, n.minobsinnode=10, n.trees=150 
## + Fold5: shrinkage=0.1, interaction.depth=1, n.minobsinnode=10, n.trees=150 
## - Fold5: shrinkage=0.1, interaction.depth=1, n.minobsinnode=10, n.trees=150 
## + Fold5: shrinkage=0.1, interaction.depth=2, n.minobsinnode=10, n.trees=150 
## - Fold5: shrinkage=0.1, interaction.depth=2, n.minobsinnode=10, n.trees=150 
## + Fold5: shrinkage=0.1, interaction.depth=3, n.minobsinnode=10, n.trees=150 
## - Fold5: shrinkage=0.1, interaction.depth=3, n.minobsinnode=10, n.trees=150 
## Aggregating results
## Selecting tuning parameters
## Fitting n.trees = 150, interaction.depth = 3, shrinkage = 0.1, n.minobsinnode = 10 on full training set
```

```r
gmbfit$finalModel
```

```
## A gradient boosted model with multinomial loss function.
## 150 iterations were performed.
## There were 51 predictors of which 45 had non-zero influence.
```

```r
class(gmbfit)
```

```
## [1] "train"         "train.formula"
```

```r
predgmb<-predict(gmbfit, newdata=test)
confusionMatrix(predgmb, test$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1380   34    0    0    1
##          B    7  889   32    3    6
##          C    8   20  811   29    4
##          D    0    1    9  761    9
##          E    0    5    3   11  881
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9629         
##                  95% CI : (0.9572, 0.968)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.953          
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9892   0.9368   0.9485   0.9465   0.9778
## Specificity            0.9900   0.9879   0.9849   0.9954   0.9953
## Pos Pred Value         0.9753   0.9488   0.9300   0.9756   0.9789
## Neg Pred Value         0.9957   0.9849   0.9891   0.9896   0.9950
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2814   0.1813   0.1654   0.1552   0.1796
## Detection Prevalence   0.2885   0.1911   0.1778   0.1591   0.1835
## Balanced Accuracy      0.9896   0.9623   0.9667   0.9709   0.9865
```

```r
predtrain<-predict(gmbfit, newdata=training)
confusionMatrix(predtrain, training$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 4138   84    0    0    4
##          B   30 2699   52    5   15
##          C   13   57 2479   64   18
##          D    4    2   32 2330   25
##          E    0    6    4   13 2644
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9709          
##                  95% CI : (0.9681, 0.9736)
##     No Information Rate : 0.2843          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9632          
##  Mcnemar's Test P-Value : 4.129e-12       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9888   0.9477   0.9657   0.9660   0.9771
## Specificity            0.9916   0.9914   0.9875   0.9949   0.9981
## Pos Pred Value         0.9792   0.9636   0.9422   0.9737   0.9914
## Neg Pred Value         0.9955   0.9875   0.9927   0.9933   0.9949
## Prevalence             0.2843   0.1935   0.1744   0.1639   0.1839
## Detection Rate         0.2812   0.1834   0.1684   0.1583   0.1796
## Detection Prevalence   0.2871   0.1903   0.1788   0.1626   0.1812
## Balanced Accuracy      0.9902   0.9695   0.9766   0.9804   0.9876
```

```r
predtrain<-predict(gmbfit, newdata=training)
confusionMatrix(predtrain, training$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 4138   84    0    0    4
##          B   30 2699   52    5   15
##          C   13   57 2479   64   18
##          D    4    2   32 2330   25
##          E    0    6    4   13 2644
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9709          
##                  95% CI : (0.9681, 0.9736)
##     No Information Rate : 0.2843          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9632          
##  Mcnemar's Test P-Value : 4.129e-12       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9888   0.9477   0.9657   0.9660   0.9771
## Specificity            0.9916   0.9914   0.9875   0.9949   0.9981
## Pos Pred Value         0.9792   0.9636   0.9422   0.9737   0.9914
## Neg Pred Value         0.9955   0.9875   0.9927   0.9933   0.9949
## Prevalence             0.2843   0.1935   0.1744   0.1639   0.1839
## Detection Rate         0.2812   0.1834   0.1684   0.1583   0.1796
## Detection Prevalence   0.2871   0.1903   0.1788   0.1626   0.1812
## Balanced Accuracy      0.9902   0.9695   0.9766   0.9804   0.9876
```



### Conclusion

Once, the predictions were obtained for the 20 test cases provided, the below shown script was used to obtain single text files to be uploaded to the courses web site to comply with the submission assigment. 20 out of 20 hits also confirmed the accuracy of the obtained models.
