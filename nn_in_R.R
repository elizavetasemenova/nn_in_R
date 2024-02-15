# ----------------- simulate data ----------------- 

# Generate some sample data
set.seed(123)
n <- 1000
x1 <- runif(n, min = -1, max = 1)
x2 <- runif(n, min = -1, max = 1)
y <- ifelse((x1 * x2) > 0, 1, 0)  # binary classification task

# Visualize dependence between X and y
plot(x1, x2, col = ifelse(y == 1, "blue", "red"), pch = 20, 
     #main = "Dependence between X and y",
     main = "",
     xlab = "x1", ylab = "x2")
legend("bottomright", legend = c("y = 0", "y = 1"), col = c("red", "blue"), pch = 20)

# Combine into a data frame
data <- data.frame(x1, x2, y)

# ----------------- define neural network ----------------- 

# Install and load required packages
library(neuralnet)


# Split data into training and testing sets
train_indices <- sample(1:n, 0.7 * n)
train_data <- data[train_indices, ]
test_data <- data[-train_indices, ]

# Define the neural network architecture
# Here, we have 2 input nodes (x1 and x2), 2 nodes in the hidden layer, and 1 output node
# We use 'sigmoid' as the activation function for both the hidden and output layers
nn <- neuralnet(y ~ x1 + x2, data = train_data, hidden = 5, act.fct = "logistic")

# ----------------- evaluate neural network ----------------- 

# Predict on test data
predicted <- predict(nn, test_data)

# Convert predicted probabilities to binary predictions
predicted_class <- ifelse(predicted > 0.5, 1, 0)

# Calculate accuracy
accuracy <- mean(predicted_class == test_data$y)
cat("Accuracy:", accuracy, "\n")

predicted_full <- predict(nn, data)
y_pred_full <- ifelse(predicted_full > 0.5, 1, 0)

# Visualize dependence between X and y
plot(x1, x2, col = ifelse(y == 1, "blue", "red"), pch = 19, 
     main = "",
     xlab = "x1", ylab = "x2")
points(x1, x2, col = ifelse(y_pred_full == 1, "green", "orange"), pch = 4)

