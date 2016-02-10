randForestTrain <- function(finMicro){

  #Finmicro is the dataframe that contains recurrence in the first column 
  #load required packages
  
  for (package in c('caret', 'ROCR','dplyr','plyr','randomForest')) {
    if (!require(package, character.only=T, quietly=T)) {
       if(!require(package)){
        stop(paste0("could not install the package: ",package))
         return(NULL)
      }
    }
  }
  #bb variable stores the results of different repeats
  bb <- data.frame()
  
  #mylevels keeps the data
  mylevels <- c("0","1")

  #check if first column is a factor if not change it to factor
  if(!is.factor(finMicro[,1]))
  {
    print("First Column must be a factor with levels 0 and 1. \n Converting first column to a factor")
    finMicro[,1] <- factor(finMicro[,1],levels = mylevels)
  }
  

for (noofrepeats in 1:100){
  print(paste0("Running iteration: ", noofrepeats))
  k = 5 # Number of Folds in cross validation
  
  # prediction and testset data frames that we add to 
  # with each iteration over the folds
  
  prediction <- data.frame()
  testsetCopy <- data.frame()
  
  #Creating a progress bar to know the status of CV
  progress.bar <- create_progress_bar("text")
  progress.bar$init(k)
  
  #We will average the auc value and f1 in each iteration, so initialize them to zeros
  aucval <-0
  f1 <-0
  
  #Creat folds for the data
  folds <- createFolds(factor(finMicro$Recurrence),k=k,list=FALSE)
  
  for (i in 1:k){
    
    #subset the training set from the data
    trainingset <- finMicro[!(folds == i), , drop=FALSE]
    testset <- finMicro[(folds == i), , drop=FALSE]  
    
    #Generate a formula for different variables in the dataframe
    frmla <- as.formula(paste(colnames(finMicro)[1], paste(colnames(finMicro)[2:ncol(finMicro)], sep = "", 
                                  collapse = " + "), sep = " ~ "))
    # train a random forest model
    mymodel <- randomForest(frmla, data=trainingset, node=3, ntree = 100)
    
    # predict from the trained random forest model
    # the result is a probility of the sample taking the value 1
    temp <- as.data.frame(predict(mymodel, testset[,-1],type="prob"))
    
    # define prediction to calculate AUC
    predf <- prediction(temp[,2],as.factor(testset[,1]))
    
    # calculate auc
    auc.tmp <- performance(predf,"auc");
    aucval  <- aucval + as.numeric(auc.tmp@y.values)
    
    #calculate f-measure
    ftemp <- performance(predf,"f");
    
    f1 <- f1+max(ftemp@y.values[[1]],na.rm=TRUE)
    
    # change the probability in the prediction to 0 and 1
    prediction <- rbind(prediction, as.data.frame(as.numeric(temp[,2] > 0.5)))
    
    # append this iteration's test set to the test set copy data frame
    testsetCopy <- rbind(testsetCopy, as.data.frame(testset[,1]))
    
    progress.bar$step()
  }

  # add predictions and actual values together in the result dataframe
  result <- as.data.frame(cbind((prediction), testsetCopy[, 1]))
  names(result) <- c("Predicted", "Actual")

  #calculate confusion matrix using confusion matrix in caret pacakge
  aa <-  confusionMatrix(factor(result[,1],levels=mylevels),
  factor(result[,2],levels=mylevels), positive='1')

  #insert values in the bb dataframe to return
  bb[noofrepeats,1] <- aa$overall[1]
  bb[noofrepeats,2] <- aa$byClass[1]
  bb[noofrepeats,3] <- aa$byClass[2]
  bb[noofrepeats,4] <- aa$byClass[3]
  bb[noofrepeats,5] <- aa$byClass[4]
  bb[noofrepeats,6] <- aa$overall[2]
  bb[noofrepeats,7] <- aucval/k
  bb[noofrepeats,8] <- f1/k
  
  # print(aa$overall[1])
  #readline(prompt="Press [enter] to continue")
}

#Rename the columns to what they mean in the dataframe
  
names(bb)<- c("Accuracy","Sensitivity","Specificity",
              "Pos_Pred_Val","Neg_Pred_Val","Kappa","AUC","F-Measure")

return(bb)
}
