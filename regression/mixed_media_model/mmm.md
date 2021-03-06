Created by Chia, Jonathan on Apr 09, 2022

  
# Introduction

Mixed Media Models: fancy regression models that use special transformations and inputs to adjust for nuances in mixed marketing

Using a Mixed Media Model will allow us to predict the outcome of changes to our marketing portfolio. We will be able to test and optimize various scenarios once our model is built. An essential part of MMM being accurate/useful is fluctuations in spend. It will be very difficult for the model to understand the influence of marketing spends that never change. 


  

## Key # 1:

Spend has a diminishing + lagged effect on sales. MMM models use adstock, geodecay, hill, and other functions to transform the Beta coefficient

### Robyn
Facebook has open source MMM code called "Robyn".  

[https://towardsdatascience.com/automated-marketing-mix-modeling-with-facebooks-robyn-fd79e60b489d](https://towardsdatascience.com/automated-marketing-mix-modeling-with-facebooks-robyn-fd79e60b489d)

[https://facebookexperimental.github.io/Robyn/docs/analysts-guide-to-MMM/?fbclid=IwAR36QpuqTPP1eR2hgXwat-gZrltfRnzi3jVhUAGF7VKFnR6BEnRmesmu-Tw](https://facebookexperimental.github.io/Robyn/docs/analysts-guide-to-MMM/?fbclid=IwAR36QpuqTPP1eR2hgXwat-gZrltfRnzi3jVhUAGF7VKFnR6BEnRmesmu-Tw)

Takeaways from this article:

*   Instead of using spend as our independent variables, it may be better to use total touchpoints. (Circulation for Print, ads served for digital media). The article makes the argument that the spend in most cases does not accurately represent the outreach of the marketing. We will need all of these touchpoints to be at the day or week level. If we wanted this to be a whole company model, we could include the TV viewership in addition to all of our digital, print, and email marketing. The problem with that is the TV viewership would overshadow the everything else. It would be very difficult to observe the effect of our marketing test budget ~30K a week. 
*   Robyn uses 'Prophet' which will take care of seasonality

  

  
### Bayesian MMM
Great presentation on Bayesian MMM. contains chunks of code and lines up with our goal of acquisition. 

  

[https://towardsdatascience.com/bayesianmmm-state-of-the-art-media-mix-modelling-9207c4445757](https://towardsdatascience.com/bayesianmmm-state-of-the-art-media-mix-modelling-9207c4445757)

  

  

### Doordash MMM  

[https://doordash.engineering/2020/07/31/optimizing-marketing-spend-with-ml/](https://doordash.engineering/2020/07/31/optimizing-marketing-spend-with-ml/)

  
---
Document generated by Confluence on Apr 09, 2022 16:54

[Atlassian](http://www.atlassian.com/)
