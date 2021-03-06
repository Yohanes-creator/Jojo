#- set working directory 
#setwd("script")
options(scipen = 999)

# load necessary packages
library(neuralnet)
#library(nnet)
#library(NeuralNetTools)
library(plyr)
library(dplyr)
library(ggplot2)
#library(maptools)
library(astsa)
#library(leaflet)
library(caret)
library(knitr)
library(tidyr)
library(ROCR)
library(bigmemory)
library(doParallel)
registerDoParallel(cores=8)


##### step 1: collecting data
##### step 2: exploring and preparing the data
# read in data and examine structure
data <- read.csv("dataset/Gempa_Tsunami_BMKG_Bathy_Jarak.csv")

# Encode as a one hot vector multilabel data
data <- cbind(data, class.ind(as.factor(data$TypeMag)))

# data-time type
data$Date <- as.POSIXct(strptime(data$Date, "%m/%d/%Y"))

# check NA 
colSums(sapply(data, is.na))


# attach year/month/day as group variables 
date.data <- sapply(as.character(data$Date), function(x) unlist(strsplit(x, "-")))

data$Year <- as.numeric(date.data[seq(1, 3*nrow(data), by = 3)])
data$Month <- as.numeric(date.data[seq(2, 3*nrow(data), by = 3)])
data$Day <- as.numeric(date.data[seq(3, 3*nrow(data), by = 3)])

# - Univariate Analysis
# Tsunami and no tsunami in data set - Fair representation of both outcomes
kable(table(data$MT),
      col.names = c("Tsunami", "Frequency"), align = 'l')

# convert MT to factor
data$FlagTsu = data$MT
data$MT = as.factor(data$MT)
levels(data$MT) = make.names(levels(factor(data$MT)))

# Visualiztion for tsunami 
data %>% 
  group_by(Year) %>% 
  summarise(Avg.num = n(), 
            Avg.mt = mean(MT, na.rm = T)) %>%
  ggplot(aes(x = Year, y = Avg.num)) + 
  geom_col(fill = "blue") + 
  stat_smooth(col = "red", method = "loess") + 
  labs(x = "Year",
       y = "Total Observations Tsunami Each Year",
       title = "Total Observations Tsunami Each Year (2008-2018)",
       caption = "Source: Significant Tsunami 2008-2018 by BMKG") + 
  theme_bw()

# Visualiztion for earthquake
data %>% 
  group_by(Year) %>% 
  summarise(Avg.num = n(), 
            Avg.mag = mean(Mag, na.rm = T)) %>%
  ggplot(aes(x = Year, y = Avg.num)) + 
  geom_col(fill = "blue") + 
  stat_smooth(col = "red", method = "loess") + 
  labs(x = "Year",
       y = "Total Observations Earthquake Each Year",
       title = "Total Observations Earthquake Each Year (2008-2018)",
       caption = "Source: Significant Earthquake 2008-2018 by BMKG") + 
  theme_bw()

# Check out the average magnitude of all earthquakes happened each year.
data %>% 
  group_by(Year) %>% 
  summarise(Avg.num = n(), Avg.mag = mean(Mag, na.rm = T)) %>%
  ggplot(aes(x = Year, y = Avg.mag)) + 
  geom_col(fill = "blue") + 
  labs(x = "Year",
       y = "Average Magnitude Each Year",
       title = "Total Observations Earthquake Each Year (2008-2018)",
       caption = "Source: Significant Earthquake 2008-2018 by BMKG") +  
  theme_bw()


# independent variable
ggplot(gather(data[7:ncol(data)]), aes(value)) + 
  geom_histogram(bins = 5, fill = "blue", alpha = 0.6) + 
  facet_wrap(~key, scales = 'free_x')

str(data)

train.aba <- cbind(data[, 7:ncol(data)])

# Scale data
scl <- function(x){ (x - min(x))/(max(x) - min(x)) }
train.aba[, 1:ncol(train.aba)] <- data.frame(lapply(train.aba[, 1:ncol(train.aba)], scl))
head(train.aba)

##### step 3: training a model on the data
n <- names(train.aba)
f <- as.formula(paste("FlagTsu~", paste(n[!n %in% c("FlagTsu")], collapse = " + ")))
f

nn <- neuralnet(f,
                data = train.aba,
                hidden = c(16,8,4,2),
                act.fct = "logistic",
                linear.output = FALSE,
                lifesign = "minimal", threshold = 0.1)
#stepmax=1e7
summary(nn)

##### step 4: evaluating model performance
# visualize the network topology
plot(nn)

# plotnet
par(mar = numeric(4), family = 'serif')
plotnet(nn)

# Compute predictions
predicted.nn <- neuralnet::compute(nn, train.aba[,1:26])

# Extract results
result.predicted.nn <- predicted.nn$net.result
head(result.predicted.nn)

# Accuracy (training set)
original_values <- max.col(train.aba[, 27])
result.predicted.nn_2 <- max.col(result.predicted.nn)
mean(result.predicted.nn_2 == original_values, na.rm = TRUE)

##### step 5: improving model performance
# Crossvalidate
set.seed(10)
k <- 10
outs <- NULL
#proportion <- 0.80

 
pbar <- create_progress_bar('text')
pbar$init(k)

for(i in 1:k){
  index <- c()
  for (l in 1:length(train.aba$Year)){
    if(train.aba$Year[l] < 1.0){
      index <- c(index, l)
    }
  }
  #index <- sample(1:nrow(train.aba), round(proportion*nrow(train.aba)))
  cross.train <- train.aba[index, ]
  cross.test  <- train.aba[-index, ]
  cross.nn    <- neuralnet(f,
                           data = cross.train,
                           hidden = c(16,8,4,2),
                           act.fct = "logistic",
                           linear.output = FALSE, threshold = 0.08)
  
  # Compute predictions
  predicted.nn <- neuralnet::compute(cross.nn, cross.test[, 1:26])
  
  # Extract results
  result.predicted.nn <- predicted.nn$net.result
  
  # Accuracy (test set)
  original_values <- max.col(cross.test[, 27])
  result.predicted.nn_2 <- max.col(result.predicted.nn)
  outs[i] <- mean(result.predicted.nn_2 == original_values)
  pbar$step()
}

summary(cross.nn)

# Average accuracy
mean(outs, na.rm = TRUE)

results <- data.frame(actual = cross.test[27], prediction = predicted.nn$net.result)

#- Area Under Curve
plot(performance(prediction(results$prediction, results$FlagTsu),
                 "tpr", "fpr"))

# use probability cut off 0.5 for classification
results$prediction = ifelse(results$prediction > 0.5, 1,0)

#- confusion matrix
confusionMatrix(factor(results$prediction),
                factor(results$FlagTsu))

# save the model to disk
saveRDS(cross.nn, "output/final_model_ann_predict_jarak.rds")

