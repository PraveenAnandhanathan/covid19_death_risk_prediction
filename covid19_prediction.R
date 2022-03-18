library(datasets)
library(caret)
library(randomForest)
library(e1071)
library(party)
library(rpart)
library(rpart.plot)
library(xgboost)
library(magrittr)
library(dplyr)
library(Matrix)
library(RWeka)
library(partykit)
library(nnet)
library(glmnet)
library(ggplot2)
library(corrplot)
library(mlbench)
library(mclust)
library(pvclust)
library(ROCR)
library(neuralnet)
library(arules)
library(devtools)
library(deepnet)
library(Boruta)
library(Metrics) 

data <- read.csv("data.csv",header = TRUE, sep = ",")
str(data)
head(data)
View(data)
summary(data)

# preprocessing
data <- data[ , c(4,5,10:17)]
data$death = as.factor(data$death) #not for linear regression,xgboost,neuralnet
data$gender = as.numeric(as.factor(data$gender))#only for j48,neuralnet
data$symptom1 = as.numeric(as.factor(data$symptom1))
data$symptom2 = as.numeric(as.factor(data$symptom2))
data$symptom3 = as.numeric(as.factor(data$symptom3))
data$symptom4 = as.numeric(as.factor(data$symptom4))
data$symptom5 = as.numeric(as.factor(data$symptom5))
data$symptom6 = as.numeric(as.factor(data$symptom6))
data$age[which(is.na(data$age))] <- 0
data$gender[which(is.na(data$gender))] <- 0
data$symptom1[which(is.na(data$symptom1))] <- 0
data$symptom2[which(is.na(data$symptom2))] <- 0
data$symptom3[which(is.na(data$symptom3))] <- 0
data$symptom4[which(is.na(data$symptom4))] <- 0
data$symptom5[which(is.na(data$symptom5))] <- 0
data$symptom6[which(is.na(data$symptom6))] <- 0

# data$location = as.numeric(as.factor(data$location))
# data$country = as.numeric(as.factor(data$country))
# data$sym_on = as.numeric(as.factor(data$sym_on))
# data$hosp_vis = as.numeric(as.factor(data$hosp_vis))
# data$location[which(is.na(data$location))] <- 0
# data$country[which(is.na(data$country))] <- 0
# data$sym_on[which(is.na(data$sym_on))] <- 0
# data$hosp_vis[which(is.na(data$hosp_vis))] <- 0
# data$from_wuhan[which(is.na(data$from_wuhan))] <- 0
# data$vis_wuhan[which(is.na(data$vis_wuhan))] <- 0

#training and testing
classdata <- data[ which(data$death == 1 | data$recov == 1),]
unclassdata <- data[ which(data$death == 0 & data$recov == 0),]
write.table(classdata, file = "classdata.csv", sep=",", row.names=FALSE)
set.seed(222)

ind <-sample(2, nrow(classdata), replace = TRUE, prob = c(0.8,0.2))
tdata <- classdata[ind==1,]
vdata <- classdata[ind==2,]

#execution in rattle
library(RGtk2)
library(rattle)
rattle()


##########FEATURE SELECTION##########
##boruta 
boruta <- Boruta(death~age+gender+symptom1+symptom2+symptom3+symptom4+symptom5+symptom6,
                 data=classdata, doTrace = 2)
print(boruta)
plot(boruta, las = 2, cex.axis = 0.7)
plotImpHistory(boruta)
getConfirmedFormula(boruta)
bor <- TentativeRoughFix(boruta)
print(bor)
attStats(boruta)

##xgboost 
trainm <- sparse.model.matrix(death~age+gender+symptom1+symptom2+symptom3+symptom4+symptom5+symptom6,
                              data=classdata)
train_label <- classdata[,"death"]
train_matrix <- xgb.DMatrix( data = as.matrix(trainm), label = train_label)

nc <- length(unique(train_label))
xgb_params <- list("objective" = "multi:softprob",
                   "eval_metric" = "mlogloss",
                   "num_class" = nc)
model <- xgb.train(params = xgb_params,
                   data = train_matrix,
                   nrounds = 100,
                   eta=0.3)
xgb.importance(model = model)
importance_matrix <- xgb.importance(colnames(data$death), model = model)
xgb.plot.importance(importance_matrix, rel_to_first = TRUE, xlab = "Relative importance")


##########HOLDOUT##########
##random forest
rf <- randomForest(death~age+gender+symptom1,
                   data=tdata)
varImpPlot(rf,type=2)
rf_pred <- predict(rf, vdata)
rf_pred
rf_confmax <- table(rf_pred, vdata$death)
rf_confmax
rf_acc <- sum(diag(rf_confmax))/sum(rf_confmax)
rf_pre <- diag(rf_confmax)/colSums(rf_confmax)
rf_rec <- diag(rf_confmax)/rowSums(rf_confmax)
rf_f1 <- 2 * rf_pre * rf_rec / (rf_pre + rf_rec) 
rf_actual = c(vdata$death)
rf_predicted = c(rf_pred)
rf_rmse = rmse(rf_actual, rf_predicted) 
rf_mse = mse(rf_actual, rf_predicted) 
rf_mae = mae(rf_actual, rf_predicted) 

##support vector machine
svm <- svm(death~age+gender+symptom1,
           data=tdata)
svm_pred <- predict(svm, vdata)
svm_pred
svm_confmax <- table(svm_pred, vdata$death)
svm_confmax
svm_acc <- sum(diag(svm_confmax))/sum(svm_confmax)
svm_pre <- diag(svm_confmax)/colSums(svm_confmax)
svm_rec <- diag(svm_confmax)/rowSums(svm_confmax)
svm_f1 <- 2 * svm_pre * svm_rec / (svm_pre + svm_rec) 
svm_actual = c(vdata$death)
svm_predicted = c(svm_pred)
svm_rmse = rmse(svm_actual, svm_predicted) 
svm_mse = mse(svm_actual, svm_predicted) 
svm_mae = mae(svm_actual, svm_predicted) 

##naive bayes
nb <- naiveBayes(death~age+gender+symptom1,
                 data=tdata)
nb_pred <- predict(nb, vdata)
nb_pred
nb_confmax <- table(nb_pred, vdata$death)
nb_confmax
nb_acc <- sum(diag(nb_confmax))/sum(nb_confmax)
nb_pre <- diag(nb_confmax)/colSums(nb_confmax)
nb_rec <- diag(nb_confmax)/rowSums(nb_confmax)
nb_f1 <- 2 * nb_pre * nb_rec / (nb_pre + nb_rec) 
nb_actual = c(vdata$death)
nb_predicted = c(nb_pred)
nb_rmse = rmse(nb_actual, nb_predicted) 
nb_mse = mse(nb_actual, nb_predicted) 
nb_mae = mae(nb_actual, nb_predicted) 

##decision tree
tree <- rpart(death~age+gender+symptom1,
              data=tdata,  method = "class", minsplit = 1, minbucket = 1,  maxdepth = 30)
rpart.plot(tree)
rpart.rules(tree)#rule generation
dt_pred <- predict(tree,vdata, type = 'class')
dt_pred
dt_confmax <- table(dt_pred, vdata$death)
dt_confmax
dt_acc <- sum(diag(dt_confmax))/sum(dt_confmax)
dt_pre <- diag(dt_confmax)/colSums(dt_confmax)
dt_rec <- diag(dt_confmax)/rowSums(dt_confmax)
dt_f1 <- 2 * dt_pre * dt_rec / (dt_pre + dt_rec) 
dt_actual = c(vdata$death)
dt_predicted = c(dt_pred)
dt_rmse = rmse(dt_actual, dt_predicted) 
dt_mse = mse(dt_actual, dt_predicted) 
dt_mae = mae(dt_actual, dt_predicted) 

##linear regression
lm <- lm(death~age+gender+symptom1,
         tdata)
predx <-predict(lm, vdata)
lm_pred <- round(predx, 0)
lm_pred
lm_confmax <- table(lm_pred, vdata$death)
lm_confmax
lm_acc <- sum(diag(lm_confmax))/sum(lm_confmax)
lm_pre <- diag(lm_confmax)/colSums(lm_confmax)
lm_rec <- diag(lm_confmax)/rowSums(lm_confmax)
lm_f1 <- 2 * lm_pre * lm_rec / (lm_pre + lm_rec) 
lm_actual = c(vdata$death)
lm_predicted = c(lm_pred)
lm_rmse = rmse(lm_actual, lm_predicted) 
lm_mse = mse(lm_actual, lm_predicted) 
lm_mae = mae(lm_actual, lm_predicted) 

##logistic regression
glm <- glm(death~age+gender+symptom1,
           data=tdata, family = 'binomial')
predx <- predict(glm,vdata,type = "response")
glm_pred <- round(predx, 0)
glm_pred
glm_confmax <- table(glm_pred, vdata$death)
glm_confmax
glm_acc <- sum(diag(glm_confmax))/sum(glm_confmax)
glm_pre <- diag(glm_confmax)/colSums(glm_confmax)
glm_rec <- diag(glm_confmax)/rowSums(glm_confmax)
glm_f1 <- 2 * glm_pre * glm_rec / (glm_pre + glm_rec) 
glm_actual = c(vdata$death)
glm_predicted = c(glm_pred)
glm_rmse = rmse(glm_actual, glm_predicted) 
glm_mse = mse(glm_actual, glm_predicted) 
glm_mae = mae(glm_actual, glm_predicted) 

##k nearest neighbour
knn <- train(death~age+gender+symptom1,
             data=tdata,
             method = 'knn',
             tuneLength = 20)
knn_pred <- predict(knn,vdata)
knn_pred
knn_confmax <- table(knn_pred, vdata$death)
knn_confmax
knn_acc <- sum(diag(knn_confmax))/sum(knn_confmax)
knn_pre <- diag(knn_confmax)/colSums(knn_confmax)
knn_rec <- diag(knn_confmax)/rowSums(knn_confmax)
knn_f1 <- 2 * knn_pre * knn_rec / (knn_pre + knn_rec) 
knn_actual = c(vdata$death)
knn_predicted = c(knn_pred)
knn_rmse = rmse(knn_actual, knn_predicted) 
knn_mse = mse(knn_actual, knn_predicted) 
knn_mae = mae(knn_actual, knn_predicted) 

##feedforward
nn <- neuralnet(death~age+gender+symptom1,
                data=tdata, hidden = c(2,1), threshold = 0.01, linear.output = FALSE)
nn$result.matrix
plot(nn)

temp_test <- subset(vdata, select = c("age","gender","symptom1"))
nn.results <- compute(nn, temp_test)

results <- data.frame(actual = vdata$death, prediction = nn.results$net.result)
roundedresults <- sapply(results,round,digits=0)
roundedresultsdf = data.frame(roundedresults)
attach(roundedresultsdf)
nn_confmax <- table(actual,prediction)
nn_confmax
nn_acc <- sum(diag(nn_confmax))/sum(nn_confmax)
nn_pre <- diag(nn_confmax)/colSums(nn_confmax)
nn_rec <- diag(nn_confmax)/rowSums(nn_confmax)
nn_f1 <- 2 * nn_pre * nn_rec / (nn_pre + nn_rec) 
nn_actual = c(vdata$death)
nn_predicted = c(nn.results$net.result)
nn_rmse = rmse(nn_actual, nn_predicted) 
nn_mse = mse(nn_actual, nn_predicted) 
nn_mae = mae(nn_actual, nn_predicted) 

##backpropogation
nnb <- neuralnet(death~age+gender+symptom1,
                 data=tdata, hidden = c(2,1), algorithm = "backprop", learningrate = 0.1,threshold = 0.01, linear.output = FALSE)
nnb$result.matrix
plot(nnb)

temp_testb <- subset(vdata, select = c("age","gender","symptom1"))
nn.resultsb <- compute(nnb, temp_testb)

resultsb <- data.frame(actualb = vdata$death, predictionb = nn.results$net.result)
roundedresultsb <- sapply(resultsb,round,digits=0)
roundedresultsdfb = data.frame(roundedresultsb)
attach(roundedresultsdfb)
nnb_confmax <- table(actualb,predictionb)
nnb_confmax
nnb_acc <- sum(diag(nnb_confmax))/sum(nnb_confmax)
nnb_pre <- diag(nnb_confmax)/colSums(nnb_confmax)
nnb_rec <- diag(nnb_confmax)/rowSums(nnb_confmax)
nnb_f1 <- 2 * nnb_pre * nnb_rec / (nnb_pre + nnb_rec) 
nnb_actual = c(vdata$death)
nnb_predicted = c(nn.results$net.result)
nnb_rmse = rmse(nnb_actual, nnb_predicted) 
nnb_mse = mse(nnb_actual, nnb_predicted) 
nnb_mae = mae(nnb_actual, nnb_predicted) 

##j48
j48 = J48(death~age+gender+symptom1,
          tdata)
plot(j48)
j48_pred <- predict(j48,vdata)
j48_pred
j48_confmax <- table(j48_pred, vdata$death)
j48_confmax
j48_acc <- sum(diag(j48_confmax))/sum(j48_confmax)
j48_pre <- diag(j48_confmax)/colSums(j48_confmax)
j48_rec <- diag(j48_confmax)/rowSums(j48_confmax)
j48_f1 <- 2 * j48_pre * j48_rec / (j48_pre + j48_rec) 
j48_actual = c(vdata$death)
j48_predicted = c(j48_pred)
j48_rmse = rmse(j48_actual, j48_predicted) 
j48_mse = mse(j48_actual, j48_predicted) 
j48_mae = mae(j48_actual, j48_predicted) 

##lasso regression
lasso <- train(death~age+gender+symptom1,
               data=tdata,
               method = 'glmnet',
               tuneGrid = expand.grid(alpha = 1,
                                      lambda = seq(0.0001, 0.2, length=5)))
lasso_pred <- predict(lasso,vdata)
lasso_pred
lasso_confmax <- table(lasso_pred, vdata$death)
lasso_confmax
lasso_acc <- sum(diag(lasso_confmax))/sum(lasso_confmax)
lasso_pre <- diag(lasso_confmax)/colSums(lasso_confmax)
lasso_rec <- diag(lasso_confmax)/rowSums(lasso_confmax)
lasso_f1 <- 2 * lasso_pre * lasso_rec / (lasso_pre + lasso_rec) 
lasso_actual = c(vdata$death)
lasso_predicted = c(lasso_pred)
lasso_rmse = rmse(lasso_actual, lasso_predicted) 
lasso_mse = mse(lasso_actual, lasso_predicted) 
lasso_mae = mae(lasso_actual, lasso_predicted)

##ridge regression
ridge <- train(death~age+gender+symptom1,
               data=tdata,
               method = 'glmnet',
               tuneGrid = expand.grid(alpha = 0,
                                      lambda = seq(0.0001, 1, length=5)))
ridge_pred <- predict(ridge,vdata)
ridge_pred
ridge_confmax <- table(ridge_pred, vdata$death)
ridge_confmax
ridge_acc <- sum(diag(ridge_confmax))/sum(ridge_confmax)
ridge_pre <- diag(ridge_confmax)/colSums(ridge_confmax)
ridge_rec <- diag(ridge_confmax)/rowSums(ridge_confmax)
ridge_f1 <- 2 * ridge_pre * ridge_rec / (ridge_pre + ridge_rec) 
ridge_actual = c(vdata$death)
ridge_predicted = c(ridge_pred)
ridge_rmse = rmse(ridge_actual, ridge_predicted) 
ridge_mse = mse(ridge_actual, ridge_predicted) 
ridge_mae = mae(ridge_actual, ridge_predicted)

##elastic net regression
en <- train(death~age+gender+symptom1,
            data=tdata,
            method = 'glmnet',
            tuneGrid = expand.grid(alpha = seq(0,1,length=10),
                                   lambda = seq(0.0001, 0.2, length=5)))
en_pred <- predict(en,vdata)
en_pred
en_confmax <- table(en_pred, vdata$death)
en_confmax
en_acc <- sum(diag(en_confmax))/sum(en_confmax)
en_pre <- diag(en_confmax)/colSums(en_confmax)
en_rec <- diag(en_confmax)/rowSums(en_confmax)
en_f1 <- 2 * en_pre * en_rec / (en_pre + en_rec) 
en_actual = c(vdata$death)
en_predicted = c(en_pred)
en_rmse = rmse(en_actual, en_predicted) 
en_mse = mse(en_actual, en_predicted) 
en_mae = mae(en_actual, en_predicted)

# ##########CROSS_VALIDATION##########
# 
# control = trainControl(method = 'cv', number = 8)
# grid = 10^seq(10, -2, length = 100)
# 
# ##randomforest
# krf <- train(death~age+gender+symptom1+symptom2+symptom3,
#              data=tdata,trControl = control,method = 'rf')
# print(krf)
# krf_pred <- predict(krf,vdata)
# krf_confmax <- confusionMatrix(krf_pred, vdata$death)
# 
# ##svm
# ksvm <- train(death~age+gender+symptom1+symptom2+symptom3,
#               data=tdata,trControl = control,method = 'svmLinear')
# print(ksvm)
# ksvm_pred <- predict(ksvm,vdata)
# ksvm_confmax <- confusionMatrix(ksvm_pred, vdata$death)
# 
# ##naivebayes
# knb <- train(death~age+gender+symptom1+symptom2+symptom3,
#              data=tdata,trControl = control,method = 'nb')
# print(knb)
# knb_pred <- predict(knb,vdata)
# knb_confmax <- confusionMatrix(knb_pred, vdata$death)
# 
# ##decisiontree
# kdt <- train(death~age+gender+symptom1+symptom2+symptom3,
#              data=tdata,trControl = control,method = 'rpart')
# print(kdt)
# kdt_pred <- predict(kdt,vdata)
# kdt_confmax <- confusionMatrix(kdt_pred, vdata$death)
# 
# ##logisticregression
# kglm <- train(death~age+gender+symptom1+symptom2+symptom3,
#               data=tdata,trControl = control,method = 'glm')
# print(kglm)
# kglm_pred <- predict(kglm,vdata)
# kglm_confmax <- confusionMatrix(kglm_pred, vdata$death)
# 
# ##knn
# kknn <- train(death~age+gender+symptom1+symptom2+symptom3,
#               data=tdata,trControl = control,method = 'knn')
# print(kknn)
# kknn_pred <- predict(kknn,vdata)
# kknn_confmax <- confusionMatrix(kknn_pred, vdata$death)
# 
# ##feedforward
# knnet <- train(death~age+gender+symptom1+symptom2+symptom3,
#                data=tdata,trControl = control,method = 'nnet')
# print(knnet)
# knnet_pred <- predict(knnet,vdata)
# knnet_confmax <- confusionMatrix(knnet_pred, vdata$death)
# 
# ##backpropogation
# knnetb <- train(death~age+gender+symptom1+symptom2+symptom3,
#                 data=tdata,trControl = control,method = 'nnet',algorithm= "backprop")
# print(knnetb)
# knnetb_pred <- predict(knnet,vdata)
# knnetb_confmax <- confusionMatrix(knnet_pred, vdata$death)
# 
# ##j48
# kj48 <- train(death~age+gender+symptom1+symptom2+symptom3,
#               data=tdata,trControl = control,method = 'J48')
# print(kj48)
# kj48_pred <- predict(kj48,vdata)
# kj48_confmax <- confusionMatrix(kj48_pred, vdata$death)
# 
# ##lasso
# klasso <- train(death~age+gender+symptom1+symptom2+symptom3,
#                 data=tdata,
#                 method = 'glmnet',
#                 tuneGrid = expand.grid(alpha = 1,
#                                        lambda = seq(0.0001, 0.2, length=5)),
#                 trControl = control)
# print(klasso)
# klasso_pred <- predict(klasso,vdata)
# klasso_confmax <- confusionMatrix(klasso_pred, vdata$death)
# 
# ##ridge
# kridge <- train(death~age+gender+symptom1+symptom2+symptom3,
#                 data=tdata,
#                 method = 'glmnet',
#                 tuneGrid = expand.grid(alpha = 0,
#                                        lambda = seq(0.0001, 1, length=5)),
#                 trControl = control)
# print(kridge)
# kridge_pred <- predict(kridge,vdata)
# kridge_confmax <- confusionMatrix(kridge_pred, vdata$death)
# 
# ##elasticnet
# ken <- train(death~age+gender+symptom1+symptom2+symptom3,
#              data=tdata,
#              method = 'glmnet',
#              tuneGrid = expand.grid(alpha = seq(0,1,length=10),
#                                     lambda = seq(0.0001, 0.2, length=5)),
#              trControl = control)
# print(ken)
# ken_pred <- predict(ken,vdata)
# ken_confmax <- confusionMatrix(ken_pred, vdata$death)

##########PLOTTING##########

# #individual plot
# j48_plot <- c(j48_acc,j48_pre[1],j48_rec[1],j48_f1[1])
# J48alg <- barplot(j48_plot,
#                   main = "J48 Classifier",
#                   col = c("red","blue","purple","green"),
#                   names.arg = c("Accuracy", "Precision", "Recall", "F1 Score"))
# text(J48alg, 0, round(j48_plot, 3),cex=1,pos=3) 
# 
# lasso_plot <- c(lasso_acc,lasso_pre[1],lasso_rec[1],lasso_f1[1])
# Lassoreg <- barplot(lasso_plot,
#                     main = "Lasso Regression",
#                     col = c("red","blue","purple","green"),
#                     names.arg = c("Accuracy", "Precision", "Recall", "F1 Score"))
# text(Lassoreg, 0, round(lasso_plot, 3),cex=1,pos=3) 
# 
# ridge_plot <- c(ridge_acc,ridge_pre[1],ridge_rec[1],ridge_f1[1])
# Ridgereg <- barplot(ridge_plot,
#                     main = "Ridge Regression",
#                     col = c("red","blue","purple","green"),
#                     names.arg = c("Accuracy", "Precision", "Recall", "F1 Score"))
# text(Ridgereg, 0, round(ridge_plot, 3),cex=1,pos=3) 
# 
# en_plot <- c(en_acc,en_pre[1],en_rec[1],en_f1[1])
# ElasticnetReg <- barplot(en_plot,
#                          main = "Elastic Net Regression",
#                          col = c("red","blue","purple","green"),
#                          names.arg = c("Accuracy", "Precision", "Recall", "F1 Score"))
# text(ElasticnetReg, 0, round(en_plot, 3),cex=1,pos=3) 

#accuracy plot
x_acc <- c(rf_acc,svm_acc,nb_acc,dt_acc,lm_acc,glm_acc,knn_acc,nn_acc,nnb_acc,j48_acc,lasso_acc,ridge_acc,en_acc)
accuracy <- barplot(x_acc,
                    main = "Comparison of Accuracy",
                    xlab = "Models",
                    col = c("red"),
                    names.arg = c("RF", "SVM", "NB", "DT", "LM", "GLM", "KNN" , "NN" , "NNB", "J48" , "LASO" , "RID" , "EN"))
text(accuracy, 0, round(x_acc, 3),cex=1,pos=3) 

#precision plot
x_pre <- c(rf_pre[1],svm_pre[1],nb_pre[1],dt_pre[1],lm_pre[1],glm_pre[1],knn_pre[1],nn_pre[1],nnb_pre[1],j48_pre[1],lasso_pre[1],ridge_pre[1],en_pre[1])
precision <- barplot(x_pre,
                     main = "Comparison of Precision",
                     xlab = "Models",
                     col = c("red"),
                     names.arg = c("RF", "SVM", "NB", "DT", "LM", "GLM", "KNN" , "NN" , "NNB", "J48" , "LASO" , "RID" , "EN"))
text(precision, 0, round(x_pre, 3),cex=1,pos=3) 

#recall plot
x_rec <- c(rf_rec[1],svm_rec[1],nb_rec[1],dt_rec[1],lm_rec[1],glm_rec[1],knn_rec[1],nn_rec[1],nnb_rec[1],j48_rec[1],lasso_rec[1],ridge_rec[1],en_rec[1])
recall <- barplot(x_rec,
                  main = "Comparison of Recall",
                  xlab = "Models",
                  col = c("red"),
                  names.arg = c("RF", "SVM", "NB", "DT", "LM", "GLM", "KNN" , "NN" , "NNB", "J48" , "LASO" , "RID" , "EN"))
text(recall, 0, round(x_rec, 3),cex=1,pos=3) 

#f1 plot
x_f1 <- c(rf_f1[1],svm_f1[1],nb_f1[1],dt_f1[1],lm_f1[1],glm_f1[1],knn_f1[1],nn_f1[1],nnb_f1[1],j48_f1[1],lasso_f1[1],ridge_f1[1],en_f1[1])
f1 <- barplot(x_f1,
              main = "Comparison of F1 Score",
              xlab = "Models",
              col = c("red"),
              names.arg = c("RF", "SVM", "NB", "DT", "LM", "GLM", "KNN" , "NN" , "NNB" , "J48" , "LASO" , "RID" , "EN"))
text(f1, 0, round(x_f1, 3),cex=1,pos=3) 

#rmse plot
x_rmse <- c(rf_rmse,svm_rmse,nb_rmse,dt_rmse,lm_rmse,glm_rmse,knn_rmse,nn_rmse,nnb_rmse,j48_rmse,lasso_rmse,ridge_rmse,en_rmse)
rmse <- barplot(x_rmse,
                    main = "Comparison of RMSE",
                    xlab = "Models",
                    col = c("red"),
                    names.arg = c("RF", "SVM", "NB", "DT", "LM", "GLM", "KNN" , "NN" , "NNB", "J48" , "LASO" , "RID" , "EN"))
text(rmse, 0, round(x_rmse, 3),cex=1,pos=3) 

#mse plot
x_mse <- c(rf_mse,svm_mse,nb_mse,dt_mse,lm_mse,glm_mse,knn_mse,nn_mse,nnb_mse,j48_mse,lasso_mse,ridge_mse,en_mse)
mse <- barplot(x_mse,
                main = "Comparison of MSE",
                xlab = "Models",
                col = c("red"),
                names.arg = c("RF", "SVM", "NB", "DT", "LM", "GLM", "KNN" , "NN" , "NNB", "J48" , "LASO" , "RID" , "EN"))
text(mse, 0, round(x_mse, 3),cex=1,pos=3) 

#mae plot
x_mae <- c(rf_mae,svm_mae,nb_mae,dt_mae,lm_mae,glm_mae,knn_mae,nn_mae,nnb_mae,j48_mae,lasso_mae,ridge_mae,en_mae)
mae <- barplot(x_mae,
               main = "Comparison of MAE",
               xlab = "Models",
               col = c("red"),
               names.arg = c("RF", "SVM", "NB", "DT", "LM", "GLM", "KNN" , "NN" , "NNB", "J48" , "LASO" , "RID" , "EN"))
text(mae, 0, round(x_mae, 3),cex=1,pos=3) 

#general plot
corrplot(cor(data))#co relation plot

plot(density(data$age))#density plot
plot(density(data$gender))#density plot
plot(density(data$death))#density plot
plot(density(data$recov))#density plot
plot(density(data$symptom1))#density plot
plot(density(data$symptom2))#density plot
plot(density(data$symptom3))#density plot
plot(density(data$symptom4))#density plot
plot(density(data$symptom5))#density plot
plot(density(data$symptom6))#density plot

plot(Mclust(data))#cluster classification

plot(pvclust(data, method.hclust="ward",method.dist="euclidean"))#cluster dendrogram


#TODO: ROC,genetic algorithm,hybrid

remove(xxx)


