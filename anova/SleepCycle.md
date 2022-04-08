---
layout: page
title: SleepCycle
---

Jonathan Chia, Krystine Osumo, Gregory Pollock 

# Introduction 
The purpose of our experiment was to gain information about what affects sleep quality using the Sleep Cycle application’s measure of sleep quality. Because we conducted an observational study, we can only conclude correlation and assume that Jonathan’s data is representative of a typical college student. We looked at the effects of total time slept and the season of the year on sleep quality. The principle questions of interest were: “Does the amount you sleep affect your overall sleep quality?” and “Does the season of the year affect your overall sleep quality?” Our initial assumptions were that Sleep quality will increase when you sleep more, student will have better sleep quality in spring/summer compared to fall/winter, there may not be a difference between 89 hours of sleep versus more than 9 hours, and that there will be an interaction between season and amount of sleep. 
 
# Design and Data Collection 
Our three null hypotheses assume: that there is no difference in mean sleep quality between Fall, Winter and Spring / Summer seasons, that there is no difference in mean sleep quality when sleeping between 5 - 7.25 hours, 7.25 - 8 hours , 8 - 9 hours  and sleeping more than 9 hours, and that there is no interaction between the seasons and the amount of sleep.  As our alternative hypothesis, we assume that at least one mean sleep quality measure differs from the rest of the seasons; we also assume that at least one mean sleep quality measure differs from the rest of the amount of sleep. We also assume that there is an interaction between the two factors.    Model We performed a 2 factor Anova Test with Sleep Quality as our response variable. The two factors are: Hours Jonathan Slept (4 levels: 5 - 7.25 hrs, 7.25 - 8 hrs, 8-9 hrs, 9+ hrs) and Seasons (3 levels: Fall, Spring and Summer, Winter). Below is our model where Yijk represents the observation, μ represents the grand average of all levels, 𝛼i  represents the season effect with i level, βj represents the amount of sleep effect with j levels, 𝛼iβj  represents the interaction effect between season effect and amount of sleep effect and εijk  represents the effect of random error. 

Yijk= μ + 𝛼i + βj + 𝛼iβj + εijk 

![power](/anova/SleepCycle_files/Sleep_Powercurve.jpg)
_Because this is an observational study, we used power analyses to find what differences in sleep quality are statistically possible to detect. The within.variance is the variance of the response variable, the sleep quality. The between variance was calculated assuming that we want to detect a difference of at least 2.5% between Spring/Summer and Fall/Winter as well as detect a difference of at least 5% between each level of time sleeping.  With at least 9 replications in each cell, we can detect a difference of 2.5% in sleep quality with a power of about 90%._

# Data Collection               
Data was gathered through the Sleep Cycle iPhone application. The user is instructed to place the phone on a bedside table while the app measures movement with a microphone to provide a measure of sleep quality with high movement or irregular movement patterns suggesting bad sleep. We used the Time in Bed, Sleep Quality and Date measurements from the app to gather our data. We also created season and time sleeping (timslpfac) factors. A sample of our data is included in the appendix. Note that the data has limitations because one can decide when to sleep and wake up, so we have to assume the Time in Bed factor is random in order to do analysis. 

![materials](/anova/SleepCycle_files/Sleep_Materials.jpg)

# Data Analysis                
After creating boxplots, we saw that a logarithmic transformation would be unnecessary since the largest variance divided by the smallest variance, was less than 2. Therefore, we concluded that they have equal variance. We also checked the normality of the observations and also the normality of residuals by looking at their Histograms. The results showed that the histogram of the observations was left skewed but did not show any extreme outliers. The histogram of the residuals was slightly left skewed but overall it resembled a normal curve. With the data satisfying the assumptions, we then performed the Analysis of Variance. 

Since we had an unbalanced experiment, we used the Type III Anova Test. Figure 1.1 looks at Time Sleeping taking into account the Season effect and we see that the p-value for Time Sleeping is sufficiently small. Figure 1.2 does the same for Season taking into account the Time Sleeping effect. The new calculations, in Figure 1.2, show that with the time sleeping effect taken into account, we saw that the p-value for Season became smaller, while the p-value for Time Sleeping remained unchanged. With Time Sleeping being the most significant, Figure 1.2 is the most representative Anova table, and we can conclude that both factors are significant. 

![anova](/anova/SleepCycle_files/Sleep_Anovatable.jpg)


![anova](/anova/SleepCycle_files/Sleep_interact.jpg)
_While checking the Anova Table, we found that the pvalue for interaction is 0.1543. Thus, we can conclude that while there is an interaction, it is not significant._


![anova](/anova/SleepCycle_files/Sleep_confintervals.jpg)
_The confidence interval support what the initial box plot showed. There appears to be a linear relationship between time sleeping and sleep quality. The means of the levels of Amount of Sleep increase as the amount of sleep increase._  

# Conclusions                  
The data suggest sleep quality improves in a somewhat linear fashion. We concluded that Season has an effect on overall sleep quality and reject that null hypothesis based on a p-value of  2.775x10-12. We also concluded that Time Sleeping has an effect on overall sleep quality based on a p-value of 2.2x10-16. There is an interaction between Season and the Time Sleeping, but it is not significant (p-value = 0.1543). We can conclude that there is correlation for Time Sleeping and sleep quality and correlation for Season and sleep quality, but we are unable to generalize our findings to all college students due to our one-person population.  
 
For this project, we have assumed that Jonathan represents a typical college student; however, everyone sleeps differently so these results are only catered to him. We were unable to measure some factors that are commonly known to affect sleep. These factors include stress, outside noises, diet or phone light before sleeping. Additionally, consistent sleep patterns probably have an effect on sleep quality, meaning the previous days’ sleep can affect the next day. However, since we have a large sample data, we believe it is enough data to balance out the effects. If we had more time and resources, we could have more students use the Sleep Cycle App and report their data since a broader sample could better represent the population of college students. The best method would be to run a sleep study experiment, with randomized treatments from a random sample. Then we could conclude causation. 
 
# Appendix

![data](/anova/SleepCycle_files/Sleep_datasample.jpg)

![](/anova/SleepCycle_files/Sleep_resids.jpg)
_The Histogram for Sleep Quality (the response) does not show any significant outliers._ 
_The Histogram of Residuals is approximately normal._ 

![](/anova/SleepCycle_files/Sleep_boxplots.jpg)
_We started the data with two fall semesters, but we noticed that Jonathan’s sleep quality was irregular for the first fall semester due to higher stress and slight insomnia. Therefore, we decided that for this study, it would be better to not take into account the first fall semester._

# R Code
see *[R Studio](https://rstudio.cloud/project/252662)*.
