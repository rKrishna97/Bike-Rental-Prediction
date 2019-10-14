rm(list = ls())
# Setting working directory
setwd("~/OneDrive/Data Science/Project/Edwisor/Project 2")

# Importing libraries
x = c("ggplot2", "corrgram", "DataCombine", "caret",
      "stats", "mlr", "rpart", "randomForest", "class", "gbm")
lapply(x, require, character.only=TRUE)

# Importing data
data_original = read.csv("day.csv")
data = data_original

# Dimension/Shape of the data
dim(data)

# Structure of the dataset
str(data)

# Columns names of the data
names(data)

# Summary of the data
summary(data)


# We have a dteday column with dates. We also have month and year columns, so i will extract only date from dteday 
# column with the help of as.Date function and then i will remove dteday columns

data$date = format(as.Date(data$dteday, format='%Y-%m-%d'), format = "%d")
# Removing dteday as we dont need it now
data$dteday = NULL
#View(data)

data = as.data.frame(data[,c("season", "yr", "mnth", "date", "holiday", "weekday", "workingday", 
               "weathersit",  "temp", "atemp", "hum", "windspeed", "casual", "registered", "cnt")])


# Missing Value Analysis -----------------------------------------------------------

# Checking for missing values
anyNA(data) # There are no missing values



# Categorical columns names
categorical_col = c("season", "yr", "mnth", "date", "holiday", "weekday", 
  "workingday", "weathersit")

# Continuous_col columns names
continuous_col = c("temp", "atemp", "windspeed", "casual",
                   "registered", "cnt")

continuous_col_wo_target = c("temp", "atemp", "windspeed", "casual",
                             "registered")
  
# Converting from numeric to factor 
for(i in categorical_col){
  data[,i] = as.factor(as.character(data[,i]))
}

str(data)


# Outlier Analysis --------------------------------------------------------
boxplot(data$temp)
boxplot(data$atemp)
boxplot(data$hum)
boxplot(data$windspeed)
boxplot(data$casual)
boxplot(data$registered)
boxplot(data$cnt)


# Feature Selection -------------------------------------------------------

# Correlation plot
corrgram(data[,continuous_col], order = F, upper.panel = panel.pie, text.panel = panel.txt,
                   main = "Correlation Plot")


for(i in continuous_col[-6]){
  print(i)
  print(cor(data$cnt, data[,i], method = "pearson"))
  cat("\n")
}


data_fsel = subset(data, select = -c(atemp, casual, hum, registered))
data_fsel = as.data.frame(data_fsel)

continuous_col = setdiff(continuous_col, c('atemp', 'casual','hum', 'registered'))


# Feature Scaling ---------------------------------------------------------

hist(data_fsel$temp)
hist(data_fsel$windspeed)
hist(data_fsel$cnt)



# Model Building  ---------------------------------------------------------
rmExcept(c("data_fsel", "data"))

# # Divide the data into train and test using the stratified sampling method
# set.seed(1234)
# train.index = createDataPartition(data_fsel$cnt, p=0.8, list = FALSE)
# train = data_fsel[train.index,]
# test = data_fsel[-train.index,]

#write.csv(train, file = "train.csv", row.names = FALSE)
#write.csv(test, file = "test.csv", row.names = FALSE)

# Importing Train and Test data that is alreadly saved
train = read.csv("train.csv")
test = read.csv("test.csv")


# Creating R-Squared fuction
rsq <- function (x, y) cor(x, y) ^ 2


# Linear Regression
lin_reg_model = lm(cnt~. , data = train)
predict_lin_reg = predict(lin_reg_model, test[,-11])
mse_linr = measureMSE(as.numeric(test$cnt), as.numeric(predict_lin_reg))
rmse_linr = measureRMSE(as.numeric(test$cnt), as.numeric(predict_lin_reg))
rsq_linr = rsq(predict_lin_reg, test$cnt)
cat("MSE:       ", mse_linr,
    "\nRMSE:      ", rmse_linr,
    "\nR Squared: ", rsq_linr)

# Decision Tree
dt_model = rpart(cnt~., data=train)
#summary(dt_model)
predict_dt = predict(dt_model, test[,-11])
mse_dt = measureMSE(as.numeric(test$cnt), as.numeric(predict_dt))
rmse_dt = measureRMSE(as.numeric(test$cnt), as.numeric(predict_dt))
rsq_dt = rsq(predict_dt, test$cnt)
cat("MSE:       ", mse_dt,
    "\nRMSE:      ", rmse_dt,
    "\nR Squared: ", rsq_dt)

# Random Forest
rf_model = randomForest(x=train[,-11], y=train$cnt,
                        importance = TRUE, ntree = 500)
#summary(rf_model)
predict_rf = predict(rf_model, test[,-11])
mse_rf = measureMSE(as.numeric(test$cnt), as.numeric(predict_rf))
rmse_rf = measureRMSE(as.numeric(test$cnt), as.numeric(predict_rf))
rsq_rf = rsq(predict_rf, test$cnt)
cat("MSE:       ", mse_rf,
    "\nRMSE:      ", rmse_rf,
    "\nR Squared: ", rsq_rf)



# KNN Regression
knn_model = knnreg(train[,-11], train$cnt, k=3)
predict_knn = predict(knn_model, test[,-11])
mse_knn = measureMSE(as.numeric(test$cnt), as.numeric(predict_knn))
rmse_knn = measureRMSE(as.numeric(test$cnt), as.numeric(predict_knn))
rsq_knn = rsq(predict_knn, test$cnt)
cat("MSE:       ", mse_knn,
    "\nRMSE:      ", rmse_knn,
    "\nR Squared: ", rsq_knn)


#Gradient Boosting
xgb_model = gbm(cnt~., data = train, n.trees = 500, interaction.depth = 2)
predict_xgb = predict(xgb_model, test[,-11], n.trees = 500)
mse_xgb = measureMSE(as.numeric(test$cnt), as.numeric(predict_xgb))
mae_xgb = measureMAE(as.numeric(test$cnt), as.numeric(predict_xgb))
rmse_xgb = measureRMSE(as.numeric(test$cnt), as.numeric(predict_xgb))
rsq_xgb = rsq(predict_xgb, test$cnt)
cat("MSE:       ", mse_xgb,
    "\nRMSE:      ", rmse_xgb,
    "\nR Squared: ", rsq_xgb)



# Creating new dataframe with Predicted Values
data_predicted = test[,-11]
data_predicted["Actual_Count"] = test$cnt
data_predicted["Predicted_Count"] = round(predict_xgb)

View(data_predicted)

