KMeans Customer Segmentation
===================================================

Created by Chia, Jonathan on Apr 09, 2022

_Notes on cluster creation for behavioral customer segmentation_

* [The Goal](#goal)
* [PCA](#pca)
* [K-Means](#kmeans)

# The Goal <a name="goal"></a>
========

Create distinguishable clusters of customers. Each cluster of customers should have different behaviors. 

  

Because there are so many different behaviors, segmenting customers manually would be very complex. Let's say we segment them by what products they purchase. We may end up with a lot of easy to name groups, but the problem we would find is that two different brands might have very similar behaviors because these brands appeal to new customers. 

The reason we use machine learning is because the algorithm can use math to find groups that are distinct and separate from each other in terms of behaviors. Kmeans essentially plots all the different customers in a multi-dimensional space, and then finds these groups.


One big problem we have is that we have so many variables that we start to have dimensionality problems (curse of dimensionality); there are some variables that are not very useful, some variables that create noise, and these extra variables make it much harder for the algorithm to work efficiently and accurately.


# PCA <a name="pca"></a>
===

Principal Component Analysis helps to solve the problems we face with multi-dimensional data. It can reduce noise, reduce non-essential variables, and still preserve most of the variance. 

### Determining if PCA is needed

I don't remember what the best way is to decide, but you can run Kmeans with PCA and without it and then compare the results. It all depends on your data and your problem. In our case, we tried it with PCA and we liked the results we got from it.

### **Normalizing before running PCA**

PCA decides on dimensionality reduction based on variance. Make sure everything is scaled from 0 to 1 so that each variable is weighted equally

### Finding the right number of PCA columns to use

Permutation Test:

Check to make sure the variance in each column is not due to random chance

Null hypothesis: simulate the null hypothesis by shuffling columns so that each row (each point in the dimensional space) is random. Thus the PCA we get is now a random null distribution for if the data is random.

Alternative hypothesis: If we reject the null hypothesis then we know that the PCA we found was from real variance not random variance

```r
find_best_pcs <- function(df,N_perm) {  
	print("Finding PCA Variance")  
	PC <- prcomp(df, center=TRUE, scale=FALSE)  
	expl_var <- PC$sdev^2/sum(PC$sdev^2)  # converting the standard deviation back to the variance
	print("Conducting Permutation Test for best PCs")  

	expl_var_perm <- matrix(NA, ncol = length(PC$sdev), nrow = N_perm)  
	for(i in 1:N_perm) {    	
		print(paste("Permutation Number",i))    
		data_perm <- apply(df,2,sample)    # shuffle the data in each column
		PC_perm <- prcomp(data_perm, center=TRUE, scale=FALSE)    # apply PCA to the data_perm (this data represents the null distribution)
		expl_var_perm[i,] <- PC_perm$sdev^2/sum(PC_perm$sdev^2)  # converting from sd to variance
	} 

	pval <- apply(t(expl_var_perm) >= expl_var,1,sum) / N_perm  # running a hypothesis test comparing the variance from the actual data versus the variance in the permuted (null distribution) data
	out <- list(    "best_pc" = head(which(pval>=0.05),1)-1,    "expl_var" = expl_var,    "expl_var_perm" = expl_var_perm,    "pval_perm" = pval)    

	return(out)
}
```
  

  

# K-Means <a name="kmeans"></a>
=======

See the below link for a visual explanation of the algorithm:

[https://medium.com/dataseries/k-means-clustering-explained-visually-in-5-minutes-b900cc69d175](https://medium.com/dataseries/k-means-clustering-explained-visually-in-5-minutes-b900cc69d175)

### Finding the right K

Remember, the goal is to create distinguishable clusters. The more clusters we can make the better (so a higher k is generally better), but we have to be careful. With a higher number of groups, the groups can become less distinguishable. For example, let's say we have 4 groups, and groups 1 and 2 are pretty close to each other. Let's say we run it with 5 groups, and the 5th group takes some from group 1 and some from group 2. Now groups 1, 2, and 5 are all really close, with 5 bridging 1 and 2. These groups may not be as distinguishable now. 

### Testing if the algorithm made distinguishable enough clusters

If the clusters are distinguishable enough, a basic algorithm such as a decision tree should be able to see clustered data, learn the differences in behavior, and then 'replicate' the clustering algorithm.

  

Let's look at the code for a better explanation of this 'replication':

1.  First we cluster the data and assign labels to all the customers
2.  Then we split the data into a training and testing set
3.  Finally, we run a decision tree on the training data

If the clusters are distinguishable enough, the decision tree should be able to learn the differences in the clusters. To check if the decision tree learned well, we then show the test data (without the cluster labels) to the decision tree. The decision tree then assigns the labels to the test data.

We then use the confusion matrix to compare the test data's actual labels versus the decision tree's assigned labels.

  

If accuracy is very high, then we know that each cluster was different enough that the decision tree didn't get confused when assigning cluster labels to customers.

```r
cluster_comparison <- function(temp,n_clusts) {   
	# 1. create temp data and cluster 
	print("Clustering Data") 
	temp$temp_clust <- (temp %>% kmeans(n_clusts))$cluster 

	# 2. Train Test split
	 print("TrainTest Split") 
	train <- temp[tts(nrow(temp),.8,100),] 
	test <- temp[-tts(nrow(temp),.8,100),]   
	

	# 3. Train and Predict 
	print("Rpartition") 
	rpm <- rpart(as.factor(temp_clust) ~ .,train) 
	pp <- predict(rpm,newdata = select(test,-temp_clust)) 
	pp <- sapply(1:nrow(test),function(x) which.max(pp[x,])) 
	pp <- confusionMatrix(as.factor(pp),as.factor(test$temp_clust)) 
	#sum(diag(pp$table)/sum(pp$table))}
```
  

  

  
---
Document generated by Confluence on Apr 09, 2022 16:54

[Atlassian](http://www.atlassian.com/)
