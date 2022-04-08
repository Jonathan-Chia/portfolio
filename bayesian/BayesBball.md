---
layout: page
title: BayesBball
---
I am making a dashboard in shiny that will input my free throws and update my free throw true shooting percentage using bayesian statistics. 

## Problem of Interest
Why are we doing this study?
  
  define RV of interest: number of free throws made out of 10
  define the parameter of interest: free throw true shooting percentage
  assumptions: we have to assume independence of each free throw. In reality, when you make five in a row, your confidence increases and      will affect the next shots.
  
## Define Model

 ```r
  Prior Probability Distribution: $$ \pi(\theta) ~ Beta(\alpha, \beta) $$
```
note: LATEX doesn't work so insert pictures instead

My best guess for theta (shooting percentage) is E(theta) = 0.7. Therefore, alpha should be bigger than beta. Because my guess is that 95% of the time I will get between 0.5 to 0.9, my standard deviation will be about 0.2 (as alpha and beta get bigger, variance decreases).  
    
    INSERT PICTURE OF BETA DISTRIBUTION WITH 95% including 0.5 to 0.9
    
  Likelihood: f(x|theta) ~ Binomial
  
## Derive the Posterior Distribution

  Posterior: pi(theta|x) ~ Beta(x + alpha, n - x + beta)
  
## Apply to Problem of Interest

  Make decisions baded on posterior
  summarize posterior dist to help
  

<iframe src="https://jonathan-chia.shinyapps.io/BayesBball/" style="width:750px; height: 750px;">
<embed src="https://jonathan-chia.shinyapps.io/BayesBball/" style="width:750px; height: 750px;">

  
