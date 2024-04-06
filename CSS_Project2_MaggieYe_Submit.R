# Project 6: Randomization and Matching
# Maggie Ye

### Setup
library(tidyverse)
library(MatchIt)
library(ggplot2)
library(cobalt)
library(gridExtra)
library(optmatch)
library(dplyr)

options(scipen = 999)

ypsps <- read.csv("ypsps.csv")
head(ypsps)

### Q3. Randomization

# Generate a vector that randomly assigns each unit to treatment/control
set.seed(42)
ypsps <- ypsps %>%
  mutate(treatment = as.numeric(rbernoulli(length(unique(interviewid)), p = 0.5)))

# I use 'parent_Govt4All' as a baseline covariate. 
baseline_cov <- ypsps %>% 
  select(parent_Govt4All, treatment)

# Visualize the distribution by treatment/control
ggplot(baseline_cov, aes(x = parent_Govt4All, fill = factor(treatment))) +
  geom_bar(position = "stack", alpha = 0.7) +
  labs(x = "parent_Govt4All", y = "Count", fill = "Treatment", title = "Distribution of parent_Govt4All by Treatment/Control") +
  scale_fill_manual(values = c("skyblue", "black"), labels = c("Control", "Treatment")) +
  theme_minimal()

# Simulate 10,000 times

n_simulations <- 10000
simulation_results <- matrix(nrow = n_simulations, ncol = 2)

for (i in 1:n_simulations) {
  df <- ypsps %>%
    select(interviewid, parent_Govt4All) %>%
    mutate(treatment = rbinom(n(), 1, 0.5))  
  proportion_treatment <- mean(df$treatment) 
  proportion_Govt4All <- mean(df$parent_Govt4All[df$treatment == 1]) 
  simulation_results[i, 1] <- proportion_treatment
  simulation_results[i, 2] <- proportion_Govt4All
}

simulation_results_df <- as.data.frame(simulation_results)
names(simulation_results_df) <- c("Proportion_Treatment", "Proportion_Govt4All")

# Plot
par(mfrow = c(1, 2))
hist(simulation_results_df$Proportion_Treatment, breaks = 30, main = "Distribution of Treatment Proportions",
     xlab = "Treatment", ylab = "Frequency")
hist(simulation_results_df$Proportion_Govt4All, breaks = 30, main = "Distribution of parent_Govt4All in Treatment Group",
     xlab = "Parent_Govt4All", ylab = "Frequency")

# Difference in means for each simulation

simulation_results_df <- simulation_results_df %>%
  mutate(
    Diff_Means = Proportion_Govt4All - Proportion_Treatment
  )

ggplot(simulation_results_df, aes(x = seq_along(Diff_Means), y = Diff_Means)) +
  geom_point(alpha = 0.5, color = "darkred") +
  geom_line(alpha = 0.75) +
  geom_smooth(method = "lm", se = FALSE, color = "blue", linetype = "dashed") +
  labs(x = "Simulation Index", 
       y = "Difference in Means", 
       title = "Trend of Balance Across Simulations") +
  theme_bw()

###  Questions

## What do you see across your simulations? 
# The distribution for treatment & my baseline variable are both roughly normal.treatment assignment and baseline covariates are overall balanced but there are variation.

## Why does independence of treatment assignment and baseline covariates not guarantee balance of treatment assignment and baseline covariates?}
# Random assignment of treatment helps to ensure that confounding variables are equally distributed between the treatment and control groups on average. 
# However, in any single randomization, it's possible—purely by chance—to end up with an imbalance in one or more covariates between the groups. This is especially true with smaller sample sizes. Even when the treatment assignment is independent of the covariates, chance can lead to an imbalance.


### Q4 Propensity Score Matching

# 4.1 One Model
formula <- college ~ student_communicate + student_PubAff + student_Newspaper + parent_Govt4All
model <- glm(formula, data = ypsps, family = binomial())

# propensity scores
ypsps$prop_score <- predict(model, type = 'response')

# matching
match_data <- matchit(formula, data = ypsps, method = 'nearest', distance = 'logit', replace = TRUE)

# Plot the balance for the top covariates
bal <- bal.tab(match_data, un = TRUE)
plot(bal)

# Summary statistics
ps_balance <- bal.tab(match_data, un = TRUE, vars = "prop_score")
print(ps_balance)

##  All 4 covariates have standardized mean difference of p-score ≤ .1.
matched_data <- match.data(match_data)
att_model <- lm(student_ppnscal ~ college + student_communicate + student_PubAff + student_Newspaper + parent_Govt4All, data = matched_data)
att_summary <- summary(att_model)
print(att_summary)
att_estimate <- coef(att_summary)["college", "Estimate"]
print(paste("Estimated att for college:", att_estimate))

# 4.2 Simulations

# setup - renaming & cleaning
new_df <- ypsps %>%
  rename_with(~paste0("post_", .), contains("1973") | contains("1983") | contains("1982"))
post_variables <- names(new_df) %>%
  keep(~str_starts(., "post_")) # Define post_variables here before using it
pre_df <- new_df %>%
  select(-any_of(c(post_variables, "interviewid", "treatment", "college"))) %>%
  filter_all(any_vars(!is.na(.))) 
pre_variables <- colnames(pre_df)

# Simulate random covariate selection 10,000 times
results <- matrix(nrow = 10000, ncol = 3)
colnames(results) <- c("att", "proportion", "improvement")

for (i in 1:10000) {
  suppressWarnings({
    num_var <- sample(1:length(pre_variables), 1)
    random_covariates <- sample(pre_variables, num_var)
    df <- new_df %>%
      select(college, student_ppnscal, all_of(random_covariates)) %>%
      filter(complete.cases(.)) 
    model <- glm(as.formula(paste("college ~", paste(random_covariates, collapse = "+"))), data = df)
    match_att <- matchit(as.formula(paste("college ~", paste(random_covariates, collapse = "+"))), data = df, family = binomial(), estimand = "att")
    match_summ <- summary(match_att, un=F)
    balanced_covariates <- match_summ$sum.matched[abs(match_summ$sum.matched[, "Std. Mean Diff."]) < 0.1, ]
    balanced_proportion <- length(balanced_covariates) / length(random_covariates)
    match_exact_att_data <- match.data(match_att)
    covariates <- random_covariates
    matched_df <- match_exact_att_data
    smd_before <- sapply(df[, covariates], function(x) {
      (mean(x[df[["college"]] == 1],na.rm=T) - mean(x[df[["college"]] == 0],na.rm=T)) / 
        sqrt((var(x[df[["college"]] == 1]) + var(x[df[["college"]] == 0])) / 2)
    })
    smd_after <- sapply(df[, covariates], function(x) {
      (mean(x[matched_df[["college"]] == 1],na.rm=T) - mean(x[matched_df[["college"]] == 0],na.rm=T)) / 
        sqrt((var(x[matched_df[["college"]] == 1]) + var(x[matched_df[["college"]] == 0])) / 2)
    })
    average_percent_improvement <- mean((smd_before - smd_after) / smd_before * 100, na.rm = TRUE)
    model <- lm(as.formula(paste("student_ppnscal ~ college +", paste(random_covariates, collapse = "+"))), data = df)
    lm_summary <- summary(model)
    att <- lm_summary$coefficients["college","Estimate"]
  })
  results[i, ] <- c(att, balanced_proportion, average_percent_improvement)
}

# Plot att v. proportion
results_data <- as.data.frame(results)
subsample <- results_data[sample(nrow(results_data), 1000), ]
ggplot(subsample, aes(att, proportion)) +
  geom_point()+
  geom_smooth(method = "lm", se = FALSE, color = "black") + 
  scale_fill_gradient(low = "lightblue", high = "red") +
  labs(title = "att vs. proportion of balanced covariates", x = "% balanced covariates", y = "att estimate") +
  theme_minimal()

hist(results_data$proportion)
hist(results_data$att)

match_list <- list()

# Iterate for 10 times
for (i in 1:10) {
  num_var <- sample(1:length(pre_variables), 1)
  random_covariates <- sample(pre_variables, num_var)
  df <- new_df %>%
    select(interviewid, college, student_ppnscal, all_of(random_covariates))
  match_att <- matchit(as.formula(paste("college ~", paste(random_covariates, collapse = "+"))), data = df, family = binomial(), estimand = "att")
  match_list[[i]] <- love.plot(match_att)}
grid.arrange(grobs = match_list, ncol = 3)

# Number of simulations with a higher proportion of balanced covariates
average_balance <- mean(results_data$proportion)
above_average_balance <- results_data %>%
  filter(proportion > average_balance)
nrow(above_average_balance)

# 4.2 
# Questions: 

## How many simulations resulted in models with a higher proportion of balanced covariates? Do you have any concerns about this?
# 52%. This is a bit high. I'm concerned that there might be overfitting and bias in covariate selection. Also, there could be unobserved variables that confound the result.  

## Analyze the distribution of the atts. Do you have any concerns about this distribution?
# ATT graph is positive skewed. My concern is potential underestimation of the true treatment effect. 

## Do your 10 randomly chosen covariate balance plots produce similar numbers on the same covariates? Is it a concern if they do not?
# It's hard to tell. Yes, as that could indicate bias in my approach. 

# Q5 

results_2 <- matrix(nrow = 10000, ncol = 3)
colnames(results_2) <- c("att", "proportion", "improvement")

for (i in 1:10000) {
  suppressWarnings({
    num_var <- sample(1:length(pre_variables), 1)
    random_covariates <- sample(pre_variables, num_var)
    df <- new_df %>%
      select(college, student_ppnscal, all_of(random_covariates)) %>%
      filter(complete.cases(.))
    match_att <- matchit(as.formula(paste("college ~", paste(random_covariates, collapse = "+"))),
                         data = df,
                         method = "nearest",
                         distance = "glm",
                         link = "logit",
                         discard = "control",
                         replace = FALSE,
                         ratio = 2)
    match_summ <- summary(match_att, un = FALSE)
    balanced_covariates <- match_summ$sum.matched[abs(match_summ$sum.matched[, "Std. Mean Diff."]) < 0.1, ]
    balanced_proportion <- length(balanced_covariates) / length(random_covariates)
    match_exact_att_data <- match.data(match_att)
    covariates <- random_covariates
    matched_df <- match_exact_att_data
    smd_before <- sapply(df[, covariates], function(x) {
      (mean(x[df[["college"]] == 1], na.rm = TRUE) - mean(x[df[["college"]] == 0], na.rm = TRUE)) / 
        sqrt((var(x[df[["college"]] == 1]) + var(x[df[["college"]] == 0])) / 2)
    })
    smd_after <- sapply(df[, covariates], function(x) {
      (mean(x[matched_df[["college"]] == 1], na.rm = TRUE) - mean(x[matched_df[["college"]] == 0], na.rm = TRUE)) / 
        sqrt((var(x[matched_df[["college"]] == 1]) + var(x[matched_df[["college"]] == 0])) / 2)
    })
    average_percent_improvement <- mean((smd_before - smd_after) / smd_before * 100, na.rm = TRUE)
    model <- lm(as.formula(paste("student_ppnscal ~ college +", paste(random_covariates, collapse = "+"))), data = df)
    lm_summary <- summary(model)
    att <- lm_summary$coefficients["college", "Estimate"]
  })
  results_2[i, ] <- c(att, balanced_proportion, average_percent_improvement)
}

# Plot att v. proportion
results_data_2 <- as.data.frame(results_2)
subsample <- results_data_2[sample(nrow(results_data_2), 100), ]
ggplot(subsample, aes(att, proportion)) +
  geom_point()+
  geom_smooth(method = "lm", se = FALSE, color = "green") +  # Add trend line without confidence intervals
  scale_fill_gradient(low = "blue", high = "red") +
  labs(title = "att vs. proportion of balanced covariates", x = "proportion of covariates above 0.1 threshold", y = "att estimate") +
  theme_minimal()

match_list <- list()

# Iterate for 10 times
for (i in 1:10) {
  num_var <- sample(1:length(pre_variables), 1)
  random_covariates <- sample(pre_variables, num_var)
  df <- new_df %>%
    select(interviewid, college, student_ppnscal, all_of(random_covariates))
  match_att <- matchit(as.formula(paste("college ~", paste(random_covariates, collapse = "+"))), 
                       data = df, 
                       method = "nearest",
                       distance = "glm",
                       link = "logit",
                       discard = "control",
                       replace = FALSE,
                       ratio = 2)
  match_list[[i]] <- love.plot(match_att)}
grid.arrange(grobs = match_list, ncol = 3)

hist(results_data_2$proportion)

hist(results_data_2$att)

# Density plots 
old_method_plot <- ggplot(results_data, aes(x = improvement)) +
  geom_density(fill = "lightblue", alpha = 0.3) +
  labs(title = "Distribution of Percent improvement (Old Method)", x = "Percent improvement", y = "Density") +
  theme_minimal()

new_method_plot <- ggplot(results_data_2, aes(x = improvement)) +
  geom_density(fill = "black", alpha = 0.3) +
  labs(title = "Distribution of Percent improvement (New Method)", x = "Percent improvement", y = "Density") +
  theme_minimal()

combined_plot <- ggplot() +
  geom_density(data = results_data, aes(x = improvement, fill = "Old Method"), alpha = 0.5) +
  geom_density(data = results_data_2, aes(x = improvement, fill = "Nearest Neighbor"), alpha = 0.5) +
  labs(title = "Distribution of Percent improvement", x = "Percent improvement", y = "Density") +
  scale_fill_manual(values = c("Old Method" = "lightblue", "Nearest Neighbor" = "black"), labels = c("Old Method", "Nearest Neighbor")) +
  theme_minimal()

print(combined_plot)

#count number of simulations where balanced covariate proportion was higher

average_balance <- mean(results_data_2$proportion)

above_average_balance <- results_data_2 %>%
  filter(!is.na(proportion)) %>% 
  filter(proportion > average_balance)

num_above_average_balance <- nrow(above_average_balance)
print(num_above_average_balance)

# 5.2 Questions: 
## Does your alternative matching method have more runs with higher proportions of balanced covariates?
# No
## Use a visualization to examine the change in the distribution of the percent improvement in balance in propensity score matching vs. the distribution of the percent improvement in balance in your new method. Which did better? Analyze the results in 1-2 sentences.
# As shown in the figure,

# 6 Discussion Questions
## Why might it be a good idea to do matching even if we have a randomized or as-if-random design?
# Matching can still be beneficial even in randomized or as-if-random designs because it can help reduce bias and improve the precision of treatment effect estimate by reducing variability in covariates between treatment groups and ensuring better balance in observed covariates between treatment groups.

## The standard way of estimating the propensity score is using a logistic regression to estimate probability of treatment. Given what we know about the curse of dimensionality, do you think there might be advantages to using other machine learning algorithms (decision trees, bagging/boosting forests, ensembles, etc.) to estimate propensity scores instead?
# Yes. They can be useful in handling nonlinearity and interaction effects. They might also be useful when we deal with high-dimensional data by automatically performing variable selection. 


