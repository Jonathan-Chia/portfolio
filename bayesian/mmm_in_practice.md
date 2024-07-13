# 3 Takeaways From Building an MMM with Analytic Edge Consultants

## Introduction - Analytic Who? 

It is wildly helpful to have consultants who have built thousands of MMMs. 

## 1. You Should Spend 80% of Your Time Understanding the Data and Finding Good Features

Choosing the right features that reflect the business is key. 

For example, I started out using marketing spends. 

* I initially started using marketing spends as the inputs, but it's actually better to use raw impressions/clicks because CPM's change.

* For control variables, competitor data, holidays, inflation/economic metrics, pricing, promotions, and product quality data can all be very useful.
  - For inflation/economic metrics, I pulled data from the [FRED](https://fred.stlouisfed.org/) database.
  - I found that the monthly metrics didn't help that much since I was using a weekly model. They just didn't change enough month over month to give detailed info to the model.
  - I ended up using weekly gas prices for my inflation metric because it was a weekly metric and it helped improve model performance.
 
* Start out with a "Data Review". Look at correlations with data inputs and your KPI and you'll automatically know which variables are likely good features.

## 2. Learn from Experts

Without the consultants I would've been stuck.

Here are some insights I've learned:

* MMM's are usually on a weekly grain and include 2 years of data.
* For statistical significance, we looked at models that had R-squared over 80% and a MAPE and holdout MAPE below 10% (using a holdout of a month to three months).
  - These goals can change depending on the type of MMM you are building. We relied on our consultants to help us know if our metrics were good compared to other models they had seen with the same KPIs.
  - Our consultants told us that a model that aligns with business intuition and has pretty decent statistical significance is better than a model with perfect statistical significance but misaligns with business intuition.
* MMM's generally give different results than MTA because MMM's account for non-digital channels, control factors, and MTA's can suffer from lack of incrementality (campaign can take credit for touching a customer who would have made the purchase regardless of seeing the ad or not).
* Sometimes you have to build a full funnel MMM. We need to understand each part of the funnel: Awareness, Consideration, Purchase, and Loyalty.
  - To understand awareness, we built a website traffic MMM.
  - To understand consideration, we built a new customer generation MMM.
  - To understand purchase, we will build a placed orders MMM, then a cart value MMM (what if you have been upselling better?), and then a sales MMM.
  - To understand loyalty, we will build a customer lifetime value MMM.
  - Each model can help you in building the next model because you have a better understanding of every part of the sales funnel.
  - In my case, we built the website traffic MMM first and then jumped straight to Sales MMM. The CEO wanted some directional insights for our budget, so we jumped ahead to give an 80% answer.
* A bayesian MMM algorithm is highly recommended because you can input business knowledge as priors in the model.
  - For example, we ran an incrementality test to gauge our programmatic lower funnel and found it wasn't very incremental. Thus, we expect the MMM to give programmatic lower funnel a lower contribution and ROI, so we input priors to calibrate the model to align with this knowledge.
* Incrementality tests are the gold standard for truth (assuming they were run correctly). The more you have, the better you can calibrate your MMM.
* You won't have a perfect MMM in the beginning. You have to put your best MMM into production, and then continually improve it by learning from the difference in forecasts and what actually happened.
* The goal is to learn and make better business decisions. MMM is just one tool in the toolbox.
  - For example, if you are a small company that advertises across three channels, you probably don't need an MMM. You can just run an incrementality test for each channel.
  - Our consultants consistently reminded me that I'm not here to build models, but to guide good business decisions using all my tools.
* MTA is a dying solution because of cookie deprecation, so MMM's are becoming even more crucial to have.
* Our consultants looked at our MMM's and compared them with other similar brands' MMM results to help us validate that we are in the right ballpark.

These insights are worth thousands of dollars (because that is what we paid them). 

## 3. Marketing Mix Modeling is as much of an Art as a Science

How do you know when you are done? How do you know which model to pick? 
