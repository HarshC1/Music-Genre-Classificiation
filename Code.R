rm(list=ls())
cat("\014")

raw_data <- read.csv("your path", na.strings = '?', stringsAsFactors = TRUE)
colnames(raw_data) <- c("genre",	"track_id",	"artist_name",	"title",	"loudness",	"tempo",	"time_signature",	"key",	"mode",	"duration",	"avg_timbre1",	"avg_timbre2",	"avg_timbre3",	"avg_timbre4",	"avg_timbre5",	"avg_timbre6",	"avg_timbre7",	"avg_timbre8",	"avg_timbre9",	"avg_timbre10",	"avg_timbre11",	"avg_timbre12",	"var_timbre1",	"var_timbre2",	"var_timbre3",	"var_timbre4",	"var_timbre5",	"var_timbre6",	"var_timbre7",	"var_timbre8",	"var_timbre9",	"var_timbre10",	"var_timbre11",	"var_timbre12")
genre_keep <- c('classical','metal','punk','soul and reggae','jazz and blues','dance and electronica')
genre_remove <- c('classic pop and rock','folk','hip-hop','pop')
head(raw_data)
View(raw_data)

library(dplyr)
library(randomForest)

#Basic data completeness and cleanliness checks

proc_data = raw_data[raw_data$genre %in% genre_keep,-c(2,3,4)]
proc_data$genre = droplevels(proc_data$genre, exclude = genre_remove)

proc_data <- proc_data %>%
  mutate(genre = as.factor(genre),
         time_signature = as.factor(time_signature),
         key = as.factor(key),
         mode = as.factor(mode)
  )

head(proc_data)
summary(proc_data)
colSums(is.na(proc_data))
col_dups <- apply(raw_data, 2, duplicated)
summary(col_dups)

#I suspect that a lot of variables in this case would be correlated. Will check correlations among them

library(corrplot)
corrplot(cor(proc_data[,-c(1,4,5,6)]), method = 'number')
correlation <- as.data.frame(cor(proc_data[,-c(1,3,4,5,6)]))
write.csv(correlation,"Project_PreProc_Correlations.csv",sep="|")
removing  <- c("avg_timbre6", "var_timbre3", "var_timbre4", "var_timbre6", "var_timbre7", "var_timbre8", "var_timbre9", "var_timbre10") 
removing_index <- sapply(removing, function(x) which(names(proc_data) == x))


#########   MAIN DATA FRAME CREATION ###############

music.df <- proc_data[proc_data$tempo != 0,-removing_index]

head(music.df)
summary(music.df)


#############################################################################################################################
#                                             Sampling and Training/Testing data creation
#############################################################################################################################
rm(proc_data,raw_data,correlation)
library(caret)


keep__mdl <- createDataPartition(music.df$genre, p = 1, list = FALSE) #Keeping the same proportions, let's sample the data and select 50% of the total data
summary(music.df[keep__mdl,"genre"])

input.df <- music.df[keep__mdl,]
#Scaling numerical columns
input.df[, sapply(input.df, is.numeric)] <- scale(input.df[, sapply(input.df, is.numeric)])


set.seed(123)
train_index <- createDataPartition(input.df$genre, p = 0.8, list = FALSE)
train_data <- input.df[train_index, ]
test_data <- input.df[-train_index, ]

#############################################################################################################################
#                                                         Logistic Regression
#############################################################################################################################

log_model =glm(genre~.,data=train_data,family=binomial)
summary(log_model)

coef(log_model)
summary(log_model)$coef

probs<-predict(log_model,type="response")
probs[c(1:10)]
contrasts(train_data$genre)


#############################################################################################################################
#                                                         Random Forest
#############################################################################################################################
rm(col_dups,genre_keep,genre_remove,removing_index)


rf_model <- randomForest(genre~., data = train_data, mtry = 21, ntree = 1000, importance = TRUE, replace=TRUE)
pred <- predict(rf_model, test_data)

# Evaluate the model
confusionMatrix(pred, test_data$genre)
mean(pred==test_data$genre)

varImpPlot(rf_model)
importance(rf_model)


#############################################################################################################################
#                                                             KNN
#############################################################################################################################

library(class)


k <- 100  # set the number of neighbors
train.y <- train_data$genre  # set the training labels
train.x <- train_data[, !names(train_data) %in% "genre"]  # remove the class variable from the training data
test.x <- test_data[, !names(test_data) %in% "genre"]  # remove the class variable from the test data
test.y <- test_data$genre
knn_model<-knn(train.x,test.x,train.y,k)

knn.predresults <- cbind.data.frame(knn_model,test_data$genre)

mean(knn_model==test_data$genre)

summary(knn.predresults)
#rite.csv(knn.predresults,"knn.predresults_3.csv",sep="|")


acc<-rep(0,100)
for (knum in 1:100) {
  knn.fit<-knn(train.x,test.x,train.y,k=knum)
  acc[knum]<-mean(knn.fit==test.y)
}

print(paste("Highest accuracy of",as.character(acc[which.max(acc)]),"at k =",as.character(which.max(acc) )))
plot(1:100,acc,type="p",xlab="k",ylab="Test Acc")



#############################################################################################################################
#                                                             SVM
#############################################################################################################################

library(e1071)

# Tune a support vector classifier 

set.seed(123)
tune.out=tune(svm,genre~.,data=train_data,kernel="linear",ranges=list(cost=c(0.001, 0.01, 0.1, 1,5,10,100)))
summary(tune.out)
bestmod=tune.out$best.model
summary(bestmod)

# Predict using your best model on the test data and check accuracy
predict.y=predict(bestmod,test_data)
table(predict.y,test_data$genre)
mean(predict.y==test_data$genre)



tune.out=tune(svm,genre~.,data=train_data,kernel="radial",ranges=list(cost=c(1,10,100,1000),gamma=c(0.1,0.5,1)))
summary(tune.out)
bestmod=tune.out$best.model
summary(bestmod)

# Predict using your best model on the test data and check accuracy
predict.y=predict(bestmod,test_data)
table(predict.y,test_data$genre)
mean(predict.y==test_data$genre) 



tune.out=tune(svm,genre~.,data=train_data,kernel="polynomial",ranges=list(cost=c(100,1000),degree=c(2,3,4)))
summary(tune.out)
bestmod=tune.out$best.model
summary(bestmod)

# Predict using your best model on the test data and check accuracy
predict.y=predict(bestmod,test_data)
table(predict.y,test_data$genre)
mean(predict.y==test_data$genre) 


#############################################################################################################################
#                                                             Neural Network
#############################################################################################################################


# Load required packages
library(nnet)

nn_grid <- expand.grid(size=c(5,10,15,20),decay = c(0,0.01,0.1,1))
nn_model <- nnet(genre~., data = train_data, size = 20, decay = 10, maxit = 10000)


library(corrplot)
# Alternate method

ctrl <- trainControl(method = 'cv', number = 5)
nn_tune <- train(genre~., data = train_data, method = "nnet", trControl = ctrl, tuneGrid = nn_grid)

nn_tune$bestTune

# Make predictions on testing data
predictions <- predict(nn_model, newdata = test.x, type = "class")

# Evaluate model performance 
table(predictions,test_data$genre)
mean(predictions==test_data$genre)


summary(predictions, test.y)
