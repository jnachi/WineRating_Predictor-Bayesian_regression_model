# Loading necessary libraries
library(readr)        # For reading CSV files
library(tm)           # Text mining capabilities
library(SnowballC)    # Provides word stemming
library(wordcloud)    # Visualization of text data
library(RColorBrewer) # Enhances color selection for plots
library(dplyr)        # Data manipulation
library(data.table)   # Efficient data handling
library(textstem)     # Provides lemmatization
library(caret)        # Machine learning methods
library(tidyverse)    # Data manipulation and visualization
library(brms)         # Bayesian regression modeling

# Load the dataset
wine_reviews <- read_csv("final_df.csv")
names(wine_reviews)
wine_reviews$variety <- as.factor(wine_reviews$variety)

correlation <- cor(wine_reviews[sapply(wine_reviews, is.numeric)], use = "pairwise.complete.obs")
correlation <- abs(correlation['points', ])

# Remove 'points' and 'superior_rating' from the correlation vector
correlation <- correlation[!names(correlation) %in% c('points', 'superior_rating')]
# Generate the word cloud
wordcloud(names(correlation), correlation, scale=c(3,0.5), max.words=100, random.order=FALSE, rot.per=0.35, colors=brewer.pal(8, "Dark2"))


library(caret)
set.seed(123)

# Partition the data into training and testing sets (80% training, 20% testing)
split <- createDataPartition(wine_reviews$points, p = 0.80, list = FALSE)
train_data <- wine_reviews[split, ]
test_data <- wine_reviews[-split, ]

# Store the 'superior_rating' from test data for later use in validation
superior_rating <- test_data$superior_rating

# Remove the 'superior_rating' column from both training and testing datasets
# This ensures it does not influence the regression model
train_data$superior_rating <- NULL
test_data$superior_rating <- NULL

brms_model <- brm(
  formula = points ~ price + Rich + Soft + age + rich + year + soft + light + fruity + attractive + great+ richness+ dense + powerful + long  +intense + (1 + price | variety),
  data = train_data,
  family = gaussian(),
  prior = c(
    set_prior("normal(88, 5)", class = "Intercept"),
    set_prior("normal(0, 1)", class = "b"),
    set_prior("cauchy(0, 2)", class = "sd"),
    set_prior("cauchy(0, 2)", class = "sigma")
  ),
  chains = 4,
  iter = 2000,
    # warmup = 1000,
  control = list(adapt_delta = 0.95)
)


summary(brms_model)

library(bayesplot)
color_scheme_set("brightblue")
mcmc_trace(brms_model, pars = c("b_Intercept", "b_price"))
mcmc_pairs(brms_model, pars = c("b_price", "b_age"))

pp_check(brms_model)

predicted_points_test <- fitted(brms_model, newdata = test_data, summary = FALSE)
predicted_points_test <- colMeans(predicted_points_test)  # Assuming one observation per row
rmse_value_test <- sqrt(mean((predicted_points_test - test_data$points)^2))
ss_res_test <- sum((predicted_points_test - test_data$points)^2)
ss_tot_test <- sum((test_data$points - mean(test_data$points))^2)
r_squared_test <- 1 - ss_res_test / ss_tot_test
cat("Testing RMSE:", rmse_value_test, "\n")
cat("Testing R^2:", r_squared_test, "\n")

fixef(brms_model)
ranef(brms_model)  # To view random effects deviations by variety

# conditional_effects(brms_model)
plot(brms_model, variable =c("b_Intercept", "b_price", "b_Rich", "b_Soft","b_powerful"))
plot(brms_model, variable =c("b_year", "b_age", "b_rich", "b_soft","b_dense"))
plot(brms_model, variable =c("b_light", "b_fruity", "b_attractive", "b_great","b_richness"))

predicted_classifications <- ifelse(predicted_points_test >= 90, 1, 0)
conf_matrix <- confusionMatrix(as.factor(predicted_classifications), as.factor(superior_rating))
cat("Accuracy using caret:", conf_matrix$overall['Accuracy'], "\n")
cat("Sensitivity (Recall):", conf_matrix$byClass['Sensitivity'], "\n")
cat("Specificity:", conf_matrix$byClass['Specificity'], "\n")
cat("Precision:", conf_matrix$byClass['Precision'], "\n")
cat("F1 Score:", conf_matrix$byClass['F1'], "\n")

random_effects <- ranef(brms_model)
random_effects <- random_effects$variety 

# If the above doesn't work, use the following to try and see the content
print(str(random_effects))
intercept_df <- data.frame(
  Variety = dimnames(random_effects)[[1]],  # Extract varieties names
  Intercept_Estimate = random_effects[, "Estimate", "Intercept"],
  Intercept_EstError = random_effects[, "Est.Error", "Intercept"],
  Intercept_Q2.5 = random_effects[, "Q2.5", "Intercept"],
  Intercept_Q97.5 = random_effects[, "Q97.5", "Intercept"]
)

# Create a dataframe for price estimates
price_df <- data.frame(
  Variety = dimnames(random_effects)[[1]],  # Extract varieties names
  Price_Estimate = random_effects[, "Estimate", "price"],
  Price_EstError = random_effects[, "Est.Error", "price"],
  Price_Q2.5 = random_effects[, "Q2.5", "price"],
  Price_Q97.5 = random_effects[, "Q97.5", "price"]
)

# Merge dataframes by 'Variety'
combined_df <- merge(intercept_df, price_df, by = "Variety")

combined_plot <- ggplot(combined_df, aes(x = Intercept_Estimate, y = Price_Estimate, label = Variety)) +
  geom_point() +
  geom_text(aes(label=Variety), hjust=1.1, vjust=1.1, angle=45, size=3) +
  labs(title = "Intercept and Price Effects Across Wine Varieties",
       x = "Intercept Estimate",
       y = "Price Estimate") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 8),
        axis.text.y = element_text(size = 8)) +
  expand_limits(x = c(min(combined_df$Intercept_Estimate) - 2, max(combined_df$Intercept_Estimate) + 2), 
                y = c(min(combined_df$Price_Estimate) - 2, max(combined_df$Price_Estimate) + 2))

print(combined_plot)

# Ensure you have the necessary package
if (!requireNamespace("broom.mixed", quietly = TRUE)) {
  install.packages("broom.mixed")
}
library(broom.mixed)

# Extracting coefficients and confidence intervals
model_coef <- tidy(brms_model, effects = "fixed", conf.int = TRUE)

# Check the structure
print(model_coef)

coef_plot <- ggplot(model_coef, aes(x = term, y = estimate, ymin = conf.low, ymax = conf.high)) +
  geom_pointrange() +
  coord_flip() +  # Flip coordinates for easier reading
  labs(title = "Effect Sizes of Predictors on Wine Ratings",
       x = "Predictors",
       y = "Effect Size (Estimate with Confidence Interval)") +
  theme_minimal() +
  theme(axis.text.x = element_text(size = 10), 
        axis.title.x = element_text(size = 12),
        axis.text.y = element_text(size = 10), 
        axis.title.y = element_text(size = 12))

# Print the plot
print(coef_plot)



predictor_coef <- model_coef %>%
  filter(term != "(Intercept)")  # Adjust this line if the intercept has a different label


coef_plot <- ggplot(predictor_coef, aes(x = term, y = estimate, ymin = conf.low, ymax = conf.high)) +
  geom_pointrange() +
  coord_flip() +  # Flip coordinates for easier reading
  labs(title = "Effect Sizes of Predictors on Wine Ratings (Excluding Intercept)",
       x = "Predictors",
       y = "Effect Size (Estimate with Confidence Interval)") +
  theme_minimal() +
  theme(axis.text.x = element_text(size = 10), 
        axis.title.x = element_text(size = 12),
        axis.text.y = element_text(size = 10), 
        axis.title.y = element_text(size = 12))

# Print the plot
print(coef_plot)


points_data <- data.frame(
  Score = c(test_data$points, predicted_points_test),
  Type = factor(rep(c("Observed", "Predicted"), each = length(test_data$points)))
)

# Using ggplot2 to create a box plot

# Create the box plot comparing observed and predicted points
box_plot <- ggplot(points_data, aes(x = Type, y = Score, fill = Type)) +
  geom_boxplot() +
  labs(title = "Comparison of Observed and Predicted Points",
       x = "Type of Data",
       y = "Points") +
  scale_fill_brewer(palette = "Pastel1") +
  theme_minimal()

# Print the box plot
print(box_plot)

test_data <- test_data %>% 
  mutate(predicted_rating = predicted_points_test)

# Now filter for specific varieties
specific_varieties <- test_data %>% 
  filter(variety %in% c("Pinot Gris", "Bordeaux-style Red Blend"))

ggplot(specific_varieties, aes(x = predicted_rating, fill = variety)) +
  geom_density(alpha = 0.5) +
  labs(title = "Probability Distribution of Predicted Ratings for 
       Pinot Gris and Bordeaux-style Red Blend",
       x = "Predicted Rating",
       y = "Density") +
  scale_fill_brewer(palette = "Set1") +
  theme_minimal() +
  theme(legend.title = element_blank())


test_data <- test_data %>%
  mutate(predicted_rating = predicted_points_test,
         rich_present = ifelse(rich > 0, "Rich Present", "Rich Absent"),
         light_present = ifelse(light > 0, "Light Present", "Light Absent"))

# You can choose to focus on one descriptor at a time or compare them; here's how to plot for 'rich'
ggplot(test_data, aes(x = predicted_rating, fill = rich_present)) +
  geom_density(alpha = 0.5) +
  labs(title = "Probability Distribution of Predicted Ratings for 'Rich' Descriptor",
       x = "Predicted Rating", y = "Density") +
  scale_fill_brewer(palette = "Set1") +
  theme_minimal() +
  theme(legend.title = element_blank())


ggplot(test_data, aes(x = predicted_rating, fill = light_present)) +
  geom_density(alpha = 0.5) +
  labs(title = "Probability Distribution of Predicted Ratings for 'Light' Descriptor",
       x = "Predicted Rating", y = "Density") +
  scale_fill_brewer(palette = "Set1") +
  theme_minimal() +
  theme(legend.title = element_blank())
