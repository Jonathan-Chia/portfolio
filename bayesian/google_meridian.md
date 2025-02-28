# Google Meridian vs. PyMC Marketing - Quick Comparison

## Introduction

MMM is one of the hottest martech tools, especially with increasing privacy and cookie deprecation concerns. <Reference Gartner report>. Google just dropped their own MMM tool called Meridian, and it is set to disrupt the marketing industry. As a PyMC enthusiast, I was curious how Meridian compares to PyMC Marketing.  

The differences:

* Google Meridian is better at accounting for causal relationship between marketing -> search -> sales.
* PyMC Marketing requires more technical bayesian knowledge but is more flexible.
* Google created Meridian to strengthen their ecosystem and increase trust in Google Ads. PyMC create PyMC Marketing to strengthen their ecosystem and get more consulting gigs. 

The similarities:

* Both allow users to incorporate previous business knowledge such as incrementality testing to improve the performance of the models.
* Both use Bayesian framework, which is ideal for MMM analysis.
* Both are open-source, giving transparency and potential to improve organically.


## Differences

Google Meridian 

Knowing their motivations is very important for understanding the potential weaknesses and strengths of their tools. For example, PyMC may not be motivated to provide a full-fledged solution, because they want you to hire them as consultants. Meanwhile, Google is very motivated to get their models to measure Search accurately, but may not be as motivated to account for nuances in other channels. For the most part though, we can trust that Meridian will be a good tool. If companies are improving marketing efficiency, they will spend more in advertising. 





## Similarities



## Conclusion

I like how PyMC is a neutral party providing an MMM tool. I worry that Google could shape their MMM to always give search a bit more credit. I will look into this in a future post. 

Which should you use? Use Meridian unless you really need more flexibility in setting prior distributions, applying transformations, or accounting for more nuanced changes in the seasonality of your business. 



### Sources




Below are my notes on it.

Questions for Garth:

* How do we incorporate reach and frequency into DD 2.0? 

"Paid search campaigns that target brand-specific queries are very different from those that target more generic product related queries. It is best to include these campaigns as separate media channels in the model."

* Should I keep these separate in my model?




