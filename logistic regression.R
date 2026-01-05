# 📦 Load libraries
library(dplyr)
library(caret)
library(car)
library(MASS)
library(pROC)
library(ggplot2)
library(stats)

# 📂 Load and clean data
data <- read.csv("C:/Users/ACCENTURE/OneDrive/Desktop/Copy of PROJECT DATA 3.csv")

# 🧼 Impute missing numeric values with mean
data_imputed <- data %>%
  mutate(across(where(is.numeric), ~ ifelse(is.na(.), mean(., na.rm = TRUE), .)))

# 🔄 Convert categorical variables to factors
data_imputed$STATUS.SEVERITY <- as.factor(data_imputed$STATUS.SEVERITY)
data_imputed$REFERRAL.STATUS <- as.factor(data_imputed$REFERRAL.STATUS)
data_imputed$HIV.STATUS <- as.factor(data_imputed$HIV.STATUS)
data_imputed$DELIVERY.METHOD <- as.factor(data_imputed$DELIVERY.METHOD)

# 🎯 Convert target variable to binary
data_imputed$ADVERSE.NEONATAL.OUTCOME <- ifelse(data_imputed$ADVERSE.NEONATAL.OUTCOME == "Yes", 1, 0)

# ✂️ Split data into training and testing sets
set.seed(123)
train_index <- createDataPartition(data_imputed$ADVERSE.NEONATAL.OUTCOME, p = 0.6, list = FALSE)
train_data <- data_imputed[train_index, ]
test_data <- data_imputed[-train_index, ]

# 🧹 Drop unnecessary columns
train_data <- dplyr::select(train_data, -ID.No, -X)

# 📊 Fit logistic regression model
logit_model <- glm(ADVERSE.NEONATAL.OUTCOME ~ ., data = train_data, family = binomial)
summary(logit_model)
# 🔍 Stepwise selection
step_model <- stepAIC(logit_model, direction = "both")

# 📉 Check multicollinearity
vif_values <- vif(step_model)
print(vif_values)

# 🔮 Predict on test set
pred_probs <- predict(step_model, newdata = test_data, type = "response")
pred_class <- ifelse(pred_probs > 0.5, 1, 0)

# 🧮 Confusion matrix
cm <- confusionMatrix(as.factor(pred_class), as.factor(test_data$ADVERSE.NEONATAL.OUTCOME))
print(cm)

# 🔥 Heatmap for confusion matrix
conf_matrix <- as.table(cm$table)
conf_df <- as.data.frame(conf_matrix)
colnames(conf_df) <- c("Actual", "Predicted", "Freq")
win.graph()
ggplot(conf_df, aes(x = Predicted, y = Actual, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), size = 5) +
  scale_fill_gradient(low = "lightblue", high = "darkblue") +
  labs(title = "Confusion Matrix for logistic regression", x = "Predicted", y = "Actual") +
  theme_minimal()

# 🧪 ROC curve and AUC
roc_obj <- roc(test_data$ADVERSE.NEONATAL.OUTCOME, pred_probs)
plot(roc_obj, col = "blue", main = "ROC Curve for Logistic Regression")
auc_value <- auc(roc_obj)
text(0.6, 0.4, paste("AUC =", round(auc_value, 3)), col = "red", cex = 1.2)
cat("AUC:", auc_value, "\n")

# 🔢 Compute distance matrix from predicted probabilities
dist_matrix <- dist(pred_probs)

# 🔄 Apply classical MDS
mds_result <- cmdscale(dist_matrix, k = 2)

win.graph
# 🖼️ Plot MDS
plot(mds_result, 
     main = "MDS Plot of Predicted Probabilities", 
     xlab = "Dimension 1", 
     ylab = "Dimension 2", 
     col = ifelse(test_data$ADVERSE.NEONATAL.OUTCOME == 1, "red", "blue"), 
     pch = 19)
legend("topright", legend = c("Adverse Outcome", "No Adverse Outcome"), 
       col = c("red", "blue"), pch = 19)

