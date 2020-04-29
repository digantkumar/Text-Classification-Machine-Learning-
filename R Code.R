#'* Digant Kumar*
#'* 119220141*
#'* Data Mining Project*

# Location of the Newsgroup file (Change accordingly)
path = "C://Users//Digant//Desktop//Digant Study docs//Semester 2//Data Mining Project//Newsgroups"

########################################################################
                ###'* Exploration of the dataset* ###
########################################################################

# Function to store all the words from the Newsgroup folder into a single vector
Tokenizer <- function(path){
files = list.files(path)
all_words = c() #Empty vector to store all the words
index = 1

#Traversing the folders 1 by 1
for (i in 1:length(files)){
  p = paste(path,"//", files[i], sep="")
  folder = list.files(p)
  
  #Traversing all the files in the respective folders
  for (j in 1:length(folder)){
    q = paste(p,"//",folder[j],sep="")
    data = file(q, open = "r")
    lines = readLines(data) # Reads the entire content in the file
    
    for (line in 1:length(lines)){
      if (lines[line] == "" | lines[line] == " "){next}
      
      words = strsplit(lines[line]," ")[[1]] #Splitting the words on spaces
      for (k in 1:length(words)){
        if (words[k] == "" | words[k] == " "){next}
        
        all_words[index] = words[k] #Storing all the words
        index = index + 1
      }
    }
  }
}
return(all_words)
}

words <- Tokenizer(path) # Function call to get all the words, and their frequency

# Table containing all the worlds with its count
words_table = data.frame(sort(table(words),decreasing = T))

# TOP 200 most popular words
Top_200_words = words_table[1:200,]

# Filtering by length (min. size = 4) and (max. size = 20)
filtered_words = c() # Contains all the words filtered by length > 3 and less than 20
new_index = 1
for (i in 1:length(words)){
  if (nchar(words[i]) < 4 | nchar(words[i]) > 20){next}
  filtered_words[new_index] = words[i]
  new_index = new_index + 1
}
filtered_words = data.frame(sort(table(filtered_words),decreasing = T))

# Top 200 words after filtering by length
Top_200_filtered_words = filtered_words[1:200,]

# Comparing the top 200 words with and without filtering
comparison = cbind(Top_200_words, Top_200_filtered_words)
View(comparison)

# Function to read in words file by file within the Newsgroup folder
# and return its count
dat <- function(filepath){
  data <- file(filepath, open = 'r') #Opens the file in readmode
  lines <- readLines(data)
  vec1 <- c() #Empty vector to store all the words
  index <- 1
  for(line in lines){
    if(line == "" | line == " "){
      next}
    words <- strsplit(line, " ")[[1]]
    # If is the word is "" or " " we skip it else we store the word in vec1
    for(j in 1:length(words)){
      if (words[j] == "" | words[j] == " "){next}
      
      vec1[index] = words[j]
      index = index + 1
    }
  }
  close(data) # To avoid warning message
  return(data.frame(table(vec1)))
}

# Function to read words Folder by folder along with its classifer(folder number)
folderRead <- function(folder, classifier){
  df <- data.frame() # Dataframe to store words along with the file path
  files <- list.files(folder)
  for(f in files){
    path <- paste(folder, f, sep = "//")
    data <- dat(path)
    file_num <- rep(path,nrow(data))
    class <- rep(classifier, nrow(data))
    data <- cbind(data, file_num, class)
    df <- rbind(df, data)
  }
  return(df)
}

#
folders <- list.dirs(path) #Lists the directories in the path
class_index <- 1
final_df <- data.frame()
for(f in folders){
  if(f == path){next} 
  df <- folderRead(f,class_index)
  class_index = class_index + 1
  final_df = rbind(final_df, df)
}

### Function to create the bag of words

BagOfWords <- function(newsgroup_sum){
  words <- as.vector(unique(final_df$vec1)) # Calculating the unique words in the file
  newsgroup.df <- t(data.frame(table(words))) # Transposing the data.frame
  colnames(newsgroup.df) <- newsgroup.df[1,] # Changing the column name to features
  newsgroup.df <- newsgroup.df[-c(1,2),]
  file <- as.factor(final_df$file_num)
  file <- levels(file)
  newsgroup_class <- c(rep(0, nrow(newsgroup.df))) # Our response variable
  newsgroup.df <- cbind(newsgroup.df, newsgroup_class)
  class_index <- which(colnames(newsgroup.df) == "newsgroup_class")
  for(f in file){
    current_file <- which(final_df$file_num == f)
    row <- rep(0, ncol(newsgroup.df)) # Rownames
    for(c in current_file){
      match <- which(colnames(newsgroup.df) == final_df$vec1[c])
      row[match] <- final_df$Freq[c]
      row[class_index] <- final_df$class[c]
    }
    newsgroup.df <- rbind(newsgroup.df, row)
    newsgroup.df[which(colnames(newsgroup.df) == "newsgroup_class")] = 1
  }
  return(newsgroup.df)
}

cat("Creating Bag of words...\nApproximate loading time is 3 minutes...!\n")
newsgroup_df <- data.frame(BagOfWords(final_df))
print("Bag of words ready!!!")

#install.packages("taRifx") {Library for converting class of columns} 
library( taRifx )

# All the columns came out as factor variables, hence changing them first to character
# and then to numeric variables, barring out class column. 
newsgroup_df <- japply( newsgroup_df, which(sapply(newsgroup_df, class)=="factor"), as.character )
newsgroup_df$newsgroup_class = as.factor(newsgroup_df$newsgroup_class)
newsgroup_df <- japply( newsgroup_df, which(sapply(newsgroup_df, class)=="character"), as.numeric )
newsgroup_df$newsgroup_class = as.factor(newsgroup_df$newsgroup_class)


############################################################################
                      ###'* BASIC EVALUATION * ###
############################################################################

##### Splitting the dataset into Training(70%) and Testing(30%) ######

# Set seed so we may get the same result everytime, 6405 = Course Code
set.seed(6405)
# Shuffling the dataset
index = sample(1:nrow(newsgroup_df),nrow(newsgroup_df),replace=F)
shuffle = newsgroup_df[index,]
x = shuffle
x$newsgroup_class = NULL
y = shuffle$newsgroup_class
set.seed(6405)
itrain = sample(1:nrow(newsgroup_df),round(.70*(nrow(newsgroup_df))),replace = F)
dat.train = newsgroup_df[itrain,]
dat.test = newsgroup_df[-itrain,]

###### KNN ######

set.seed(6405)
x.train = x[itrain,]
x.test = x[-itrain,]
y.train = y[itrain]
y.test = y[-itrain]

library(class) # Contains knn
library(caret) # For confusion matrix, recall, precision and F1 score
ko = knn(x.train,x.test,y.train)
knn.confmat <- confusionMatrix(ko, y.test)
knn.confmat$table # Confusion Matrix
knn.confmat$overall[1] # Accuracy
knn.confmat$byClass # Recall = Sensitivity is the first column,
                    # Pos Pred. Value/Precision is the third column.
plot(ko,xlab="Class",ylab="Classification by row",main="KNN")

#'* Accuracy - 39.167% with k = 1 nearest neighbour *


###### Random Forest ######

set.seed(6405)
library(randomForest)
cat("Random Forest Processing...\nLoading time is 3 minutes")
rf.out <- randomForest(x.train,y.train)
rf.pred <- predict(rf.out,x.test)
rf.confmat <- confusionMatrix(rf.pred, y.test)
rf.confmat$table               # Confusion Matrix
rf.confmat$overall[1]           # Accuracy
rf.confmat$byClass # Recall = Sensitivity is the first column,
                  # Pos Pred. Value/Precision is the third column.
plot(rf.pred,xlab="Class",ylab="Classification by row",main="Random Forest")
#'* Accuracy - 83.33% *


####### Naive Bayes ######


set.seed(6405)
itrainshuffle = sample(1:nrow(newsgroup_df),nrow(newsgroup_df),replace = F)
shuffle = newsgroup_df[itrainshuffle,]
set.seed(6405)
itraining = sample(1:nrow(shuffle),round(.70*nrow(shuffle)),replace=F)
training = shuffle[itraining,]
testing = shuffle[-itraining,]

# Storing the rows which contain class1, classs2, class3 and class4
class1 = training[which(training$newsgroup_class == "1"),-28187]
class2 = training[which(training$newsgroup_class == "2"),-28187]
class3 = training[which(training$newsgroup_class == "3"),-28187]
class4 = training[which(training$newsgroup_class == "4"),-28187]

# Computing the probabality of all the classes
prob.class1 = nrow(class1)/nrow(training)
prob.class2 = nrow(class2)/nrow(training)
prob.class3 = nrow(class3)/nrow(training)
prob.class4 = nrow(class4)/nrow(training)

# Initializing all the probabilities as 0
prob1 = prob2 = prob3 = prob4 = 0
index = 1
x <- colnames(testing)
y.hat <- c()
y.index <- 1
voc <- 28186  # Total number of unique words in the bag of words 
cat("Naive Bayes Processing...Running time is 20 minutes")

for(i in 1:nrow(testing[,-28187])){
  row <- testing[i,-28187]
  a = which(row!=0) # Taking only those rows which contain non zero columns
  for (j in a){
    word = x[j] # Inputs words one by one
    # Summing the columns in the respective class if the word is found
    cls1_sum = sum(class1[,which(colnames(class1)==word)]) 
    cls2_sum = sum(class2[,which(colnames(class2)==word)])
    cls3_sum = sum(class3[,which(colnames(class3)==word)])
    cls4_sum = sum(class4[,which(colnames(class4)==word)])
    total1 <- sum(colSums(class1[,-28187]))  # Running on the training set
    total2 <- sum(colSums(class2[,-28187]))
    total3 <- sum(colSums(class3[,-28187]))
    total4 <- sum(colSums(class4[,-28187]))
    p1 <- log((cls1_sum + 1)/(total1 + voc))   # La Place smoothing
    p2 <- log((cls2_sum + 1)/(total2 + voc))   
    p3 <- log((cls3_sum + 1)/(total3 + voc))
    p4 <- log((cls4_sum + 1)/(total4 + voc))
    prob1 = prob1 + p1           
    prob2 = prob2 + p2
    prob3 = prob3 + p3
    prob4 = prob4 + p4
    
  }
  # Checking the probability of all classes and the feature goes in the class with the
  # highest probability
  y.hat[y.index] <- which.max(c((prob1 + log(prob.class1)) ,(prob2 + log(prob.class2)),
                                (prob3 + log(prob.class3)), (prob4 + log(prob.class4))))
  y.index <- y.index + 1
}

# Performance Metrics

nb.confmat <- confusionMatrix(y.hat, testing$newsgroup_class)
nb.confmat$overall[1] # Accuracy
nb.confmat$table      # Confusion matrix
nb.confmat$byClass    # Contains Recall, Precision & F1 score
#'* Accuracy 35% *

########################################################################
                    ###'*    ROBUST EVALUATION     * ###
########################################################################

###'*(1) APPLYING PRE PROCESSING TECHNIQUES *###                  
## Creating bag of words again, this time using pre-processing techniques
# Removing stopwords, numbers, punctuations, words of length less than 3
# and converting every word to lower case

library(tm) # For a list of stopwords
stopwords = stopwords(kind="en")

newdat <- function(filepath){
  data <- file(filepath, open = 'r')
  lines <- readLines(data)
  vec1 <- c()
  index <- 1
  for(line in lines){
    if(line == "" | line == " "){
      next}
    words <- strsplit(line, " ")[[1]]
    words = removeWords(words,stopwords) #Removing the stopwords
    words = gsub(pattern = "\\W", replace = "", words) # Removing punctuations
    words = gsub(pattern = "\\d", replace = "", words) # Removing digits
    words = tolower(words)            # Converting all the words to lower case
    words = gsub('\\b\\w{1,3}\\b','',words) # Removing words of length < 3
    words = stripWhitespace(words)          # Removing any whitespaces
    
    # If is the word is "" or " " we skip it else we store the word in vec1
    for(j in 1:length(words)){
      if (words[j] == "" | words[j] == " "){next}
      
      vec1[index] = words[j]
      index = index + 1
    }
  }
  close(data)
  return(data.frame(table(vec1)))
}

folderRead <- function(folder, classifier){
  df <- data.frame()
  files <- list.files(folder)
  for(f in files){
    path <- paste(folder, f, sep = "//")
    data <- newdat(path)
    file_num <- rep(path,nrow(data))
    class <- rep(classifier, nrow(data))
    data <- cbind(data, file_num, class)
    df <- rbind(df, data)
  }
  return(df)
}

folders <- list.dirs(path)
class_index <- 1
final_df <- data.frame()
for(f in folders){
  if(f == path){next}
  df <- folderRead(f,class_index)
  class_index = class_index + 1
  final_df = rbind(final_df, df)
}

# Bag of words function
BagOfWords <- function(newsgroup_sum){
  words <- as.vector(unique(final_df$vec1))
  newsgroup.df <- t(data.frame(table(words)))
  colnames(newsgroup.df) <- newsgroup.df[1,]
  newsgroup.df <- newsgroup.df[-c(1,2),]
  file <- as.factor(final_df$file_num)
  file <- levels(file)
  newsgroup_class <- c(rep(0, nrow(newsgroup.df)))
  newsgroup.df <- cbind(newsgroup.df, newsgroup_class)
  class_index <- which(colnames(newsgroup.df) == "newsgroup_class")
  for(f in file){
    current_file <- which(final_df$file_num == f)
    row <- rep(0, ncol(newsgroup.df))
    for(c in current_file){
      match <- which(colnames(newsgroup.df) == final_df$vec1[c])
      row[match] <- final_df$Freq[c]
      row[class_index] <- final_df$class[c]
    }
    newsgroup.df <- rbind(newsgroup.df, row)
    newsgroup.df[which(colnames(newsgroup.df) == "newsgroup_class")] = 1
  }
  return(newsgroup.df)
}

cat("Creating new Bag of words...\nApproximate loading time is 1 minute...!\n")
newsgroup_df2 <- data.frame(BagOfWords(final_df))

# Again the columns are factor variables and hence converting them to numeric
newsgroup_df2 <- japply( newsgroup_df2, which(sapply(newsgroup_df2, class)=="factor"), as.character )
newsgroup_df2$newsgroup_class = as.factor(newsgroup_df2$newsgroup_class)
newsgroup_df2 <- japply( newsgroup_df2, which(sapply(newsgroup_df2, class)=="character"), as.numeric )
newsgroup_df2$newsgroup_class = as.factor(newsgroup_df2$newsgroup_class)

set.seed(6405)
index = sample(1:nrow(newsgroup_df2),nrow(newsgroup_df2),replace=F)
shuffle2 = newsgroup_df2[index,]
x = shuffle2
x$newsgroup_class = NULL
y = shuffle2$newsgroup_class

set.seed(6405)
# Shuffling the data again
itr = sample(1:nrow(newsgroup_df2),round(.70*nrow(newsgroup_df2)),replace=F)
xtr = x[itr,]
xtst = x[-itr,]
ytr = y[itr]
ytst = y[-itr]
dat.tr <- shuffle2[itr,]
dat.tst = shuffle2[-itr,]

###'* Performing feature selection using random forest * ###

rf.fs.out <- randomForest(xtr,ytr)
varImpPlot(rf.fs.out) # Plot shows the 30 most important features in the data

imp <- as.data.frame(varImp(rf.fs.out))
imp <- data.frame(overall = imp$Overall,names = rownames(imp))
var_imp = imp[order(imp$overall,decreasing=T),]

# Selecting the top 1000 most significant features 
imp_variables = as.numeric(rownames(var_imp)[1:1000])
newsgroup_df3 = newsgroup_df2[,imp_variables]
newsgroup_df3$newsgroup_class = newsgroup_df2$newsgroup_class

set.seed(6405)
# Shuffling the dataset
index = sample(1:nrow(newsgroup_df3),nrow(newsgroup_df3),replace=F)
shuffle3 = newsgroup_df3[index,]
x = shuffle3
x$newsgroup_class = NULL
y = shuffle3$newsgroup_class

itrain = sample(1:nrow(newsgroup_df3),round(.70*nrow(newsgroup_df3)),replace = F)
x.train = x[itrain,]
x.test = x[-itrain,]
y.train = y[itrain]
y.test = y[-itrain]
dat.train = shuffle3[itrain,]
dat.test = shuffle3[-itrain,]

##########################################################################
                   ###'* Hyperparameter Tuning *###
##########################################################################

#'* Performing Grid Searching to optimize hyperparameters*

library(mlr)

set.seed(6405)
lrn.trees = makeLearner("classif.rpart")
lrn.knn = makeLearner("classif.knn")
lrn.rf = makeLearner("classif.randomForest")
lrn.svm = makeLearner("classif.ksvm")

# Defining the hyperparameters
ps.trees = makeParamSet(makeIntegerParam("minsplit",lower=1,upper=100),
                  makeIntegerParam("maxdepth",lower=2,upper=50),
                  makeDiscreteParam("cp",values=seq(0.001,0.006,0.002)))
ps.knn = makeParamSet(makeIntegerParam("k",lower=1,upper=30))
ps.rf = makeParamSet(makeDiscreteParam("ntree",values=c(100,300,500,700,900)),
                     makeDiscreteParam("mtry",values=c(100,400,700,1000)))
ps.svm = makeParamSet(makeNumericParam("C", lower=-10, upper=10, trafo = function(x)2^x),
                      makeNumericParam("sigma",lower=-10,upper=10,trafo = function(x)2^x))

ctrl = makeTuneControlGrid() # Exhaustive Grid search
rdesc = makeResampleDesc("CV",iters=3) # Performing 3-fold cross validation

# The class to be predicted
task = makeClassifTask(data=dat.train,target="newsgroup_class")

# Tuning the parameters
set.seed(6405) # Setting seed everytime as it is performing 3 Cross Validations
res.trees = tuneParams(lrn.trees,task=task, resampling = rdesc, par.set=ps.trees, 
                       control=ctrl, measures = acc) # 300 iterations
set.seed(6405)
res.knn = tuneParams(lrn.knn, task, resampling = rdesc, par.set=ps.knn, 
                       control=ctrl, measures = acc) # 30 iterations
set.seed(6405)
res.rf = tuneParams(lrn.rf, task, resampling = rdesc, par.set=ps.rf, 
                     control=ctrl, measures = acc) # 20 iterations
set.seed(6405)
res.svm = tuneParams(lrn.svm, task, resampling = rdesc, par.set=ps.svm, 
                     control=ctrl, measures = acc) # 100 iterations

# Optimal results from hyperparameter tuning
print(res.trees)
#'* The optimal parameters are minsplit = 78, maxdepth = 23, cp=0.005 with acc. 97.5% *
print(res.knn)
#'* The optimal parameters is at k=1, with mean accuracy 54.60 *
print(res.rf)
#'* The optimal parameters are ntree = 900, mtry = 100, acc = 98.57 * 
print(res.svm)
#'* The optimal parameters are C=219, sigma = 0.000977 and acc = 95.35*
par(mfrow=c(2,2))
plot(res.trees$x$minsplit,res.trees$x$maxdepth,xlab="Min. split",ylab="Max. Depth",col="black",
     pch=16,main="Decision Trees")
plot(res.knn$x,res.knn$y,xlab="K value",ylab="Accuracy",col="black",pch=16,
     main="Knn")
plot(res.rf$x$ntree, res.rf$x$mtry, xlab="No. of trees",ylab="mtry",main="Random Forest",
     pch=16, col="black")
plot(res.svm$x$C, res.svm$x$sigma, xlab="Cost", ylab="Sigma", main="SVM",
     col="black",pch=16)

##########################################################################
              ###'* K - fold Cross Validation *###
##########################################################################


library(e1071) # Contains SVM
library(rpart) # Contains Tree

K = 10
folds = cut(1:nrow(newsgroup_df3),K,labels = F)
acc.knn = acc.rf = acc.svm = acc.tree = numeric(K)
for (k in 1:K){
  i = which(folds==k)
  xtrain = x[-i,]
  xtest = x[i,]
  ytrain = y[-i]
  ytest = y[i]
  newdat.train = shuffle3[-i,]
  newdat.test = shuffle3[i,]
  ko = knn(xtrain,xtest,ytrain,k=res.knn$x$k)
  tb.cv.knn = table(ko,ytest)
  acc.knn[k] = sum(diag(tb.cv.knn))/sum(tb.cv.knn)
  rf.out <- randomForest(xtrain,ytrain,ntree = res.rf$x$ntree, mtry=res.rf$x$mtry)
  rf.pred <- predict(rf.out,xtest)
  tb.cv.rf = table(rf.pred,ytest)
  acc.rf[k] = sum(diag(tb.cv.rf))/sum(tb.cv.rf)
  svmo <- svm(xtrain,ytrain,cost = res.svm$x$C, sigma = res.svm$x$sigma)
  svm.pred <- predict(svmo, newdata=xtest)
  tb.cv.svm = table(svm.pred,ytest)
  acc.svm[k] = sum(diag(tb.cv.svm))/sum(tb.cv.svm)
  tree.out <- rpart(newsgroup_class~.,data = newdat.train,minsplit=res.trees$x$minsplit,
                    maxdepth = res.trees$x$maxdepth, cp=res.trees$x$cp)
  tree.pred <- predict(tree.out,newdat.test,type="class")
  tb.cv.tree = table(tree.pred,newdat.test$newsgroup_class)
  acc.tree[k] = sum(diag(tb.cv.tree))/sum(tb.cv.tree)
  }

mean(acc.knn) # Cross Validated KNN Mean Accuracy on the testing set : 62.75%
mean(acc.rf) # Cross Validated Random Forest Mean Accuracy on the testing set: 99.25%
mean(acc.svm) # Cross Validated SVM Mean Accuracy on the testing set: 93.25%
mean(acc.tree) # Cross Validated Decision Tree Mean Accuracy on the testing set: 98.5%


plot(acc.knn,t="b",main="Cross Validated KNN Accuracy")
plot(acc.rf,t="b",main="Cross Validated RF Accuracy")
plot(acc.svm,t="b",main="Cross Validated SVM Accuracy")
plot(acc.tree,t="b",main="Cross Validated DT Accuracy")

boxplot(acc.knn, acc.rf, acc.svm, acc.tree,
        main="Overall CV prediction accuracy",
        names=c("kNN","RF","SVM","TREE"))

###################################################################
                         ###'*Hold Out *###
###################################################################

### Decision Trees ###

tree.out <- rpart(newsgroup_class~.,data = dat.train,minsplit=res.trees$x$minsplit,
                  maxdepth=res.trees$x$maxdepth, cp=res.trees$x$cp)
tree.pred <- predict(tree.out,dat.test,type="class")

# Performance Metrics
tree.confmat <- confusionMatrix(y.test, tree.pred)
tree.confmat$overall[1]
#'* Accuracy 98.33%*
tree.confmat$table      # Confusion matrix
tree.confmat$byClass    # Contains Recall (6th column), Precision (5th column)
                        # and F1 score (7th column) per class

### Support Vector Machines ####

svm.k <- svm(x.train,y.train,cost = res.svm$x$C, sigma=res.svm$x$sigma)
svm.pred <- predict(svm.k, newdata=x.test)

# Performance Metrics
svm.confmat <- confusionMatrix(y.test, svm.pred)
svm.confmat$overall[1]
#'* Accuracy 97.5% on SVM*
svm.confmat$table     # Confusion Matrix
svm.confmat$byClass   # Contains Recall (6th column), Precision (5th column)
                      # and F1 score (7th column) per class


### KNN ###
ko = knn(x.train,x.test,y.train,k=res.knn$x$k)
plot(ko)

# Performance Metrics
knn2.confmat = confusionMatrix(y.test, ko)
knn2.confmat$overall[1]
#'* Accuracy 67.5% *
knn2.confmat$table      # Confusion matrix
knn2.confmat$byClass    # Contains Recall (6th column), Precision (5th column)
                        # and F1 score (7th column) per class

### Random Forest ###
rf.out <- randomForest(x.train,y.train,ntree = res.rf$x$ntree, mtry=res.rf$x$mtry)
rf.pred <- predict(rf.out,x.test)


# Performance Metrics
rf2.confmat = confusionMatrix(y.test, rf.pred)
rf2.confmat$overall[1]
#'* Accuracy 99.1667% *
rf2.confmat$table       # Confusion matrix
rf2.confmat$byClass     # Contains Recall (6th column), Precision (5th column)
                        # and F1 score (7th column) per class

cbind(tree.confmat$overall[1], svm.confmat$overall[1], knn2.confmat$overall[1],
      rf2.confmat$overall[1]) ## Comparing the final accuracies

# We can see that Random Forest and Decision Trees are doing the best job of classification
# while knn is doing the worst job in classifying words on this dataset.

