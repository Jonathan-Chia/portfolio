# MMM Showdown: Google Meridian vs. PyMC Marketing

<div style="display: flex; justify-content: left; gap: 10px;">
  <img src="pymc-marketing-logo.png" alt="" width="300" height="200">
  <img src="meridian-logo.png" alt="" width="300" height="200">
</div>


## Introduction

Marketing Mix Modeling (MMM) has become one of the most sought-after tools in the martech space, especially as privacy concerns and cookie deprecation reshape the advertising landscape. According to a recent Gartner report, ["64% of senior marketing leaders have adopted MMM solutions"](https://martech.org/unlocking-the-power-of-marketing-mix-modeling-solutions/) as they seek reliable, privacy-compliant ways to measure campaign effectiveness and allocate budgets.

Enter [Google Meridian](https://developers.google.com/meridian), Google’s newly launched MMM tool, which promises to disrupt the industry with its advanced capabilities and seamless integration with Google’s ecosystem. As a PyMC enthusiast, I couldn’t help but wonder: how does Meridian stack up against [PyMC Marketing](https://www.pymc-marketing.io/en/stable/), the open-source Bayesian modeling framework that has been a favorite among Bayesian marketers? Let’s dive into the differences and similarities between these two tools.


## The Differences

### 1. Google Meridian Excels at Accounting for Organic Search and Reach/Frequency

Google Meridian is designed to leverage Google’s vast data ecosystem, making it particularly strong at accounting for organic search as a confounding variable. This is a game-changer for marketers who rely heavily on SEO and want to understand how organic and paid search interact. Additionally, Meridian allows users to incorporate [reach and frequency](https://developers.google.com/meridian/docs/advanced-modeling/reach-frequency) metrics. MMMs generally "rely on impressions as input, neglecting the fact that individuals can be exposed to ads multiple times, and the impact can vary with exposure frequency." This level of granularity is a significant advantage for marketers who want to optimize their campaigns based on how often and how widely their ads are seen.

### 2. PyMC Marketing Requires More Technical Bayesian Knowledge but Offers Greater Flexibility

PyMC Marketing, built on the PyMC library, is a powerful tool for those with a strong grasp of Bayesian statistics. While it has a steeper learning curve, it offers unparalleled flexibility in modeling. Users can set [custom priors](https://www.pymc-marketing.io/en/stable/notebooks/general/prior_predictive.html) and build [highly tailored models](https://www.pymc-marketing.io/en/stable/notebooks/mmm/mmm_components.html) that fit their unique business needs. This makes PyMC Marketing ideal for organizations with specific modeling requirements or those who want full control over their MMM framework.

### 3. Different Goals: Ecosystem Strengthening vs. Consulting Opportunities

Google’s primary motivation for developing Meridian is clear: to strengthen its advertising ecosystem and increase trust in Google Ads. By offering a robust MMM tool, Google can provide advertisers with more accurate insights into their campaigns, ultimately driving more spend on its platform. On the other hand, PyMC Marketing was created to expand the PyMC ecosystem and create opportunities for consulting and customization. It’s a tool for the open-source community, designed to empower users to build and refine their own models.

## The Similarities

### 1. Both Tools Allow Incorporation of Prior Business Knowledge

One of the key strengths of both Google Meridian and PyMC Marketing is their ability to incorporate prior business knowledge into the modeling process. For example, results from incrementality tests or historical campaign performance data can be used to inform the models, improving their accuracy and relevance. This feature is crucial to [overcoming multicollinearity issues in MMMs](https://www.pymc-marketing.io/en/stable/notebooks/mmm/mmm_roas.html) (when sales are good, usually all marketing spend is increased). 

### 2. Both Use a Bayesian Framework

At their core, both tools rely on a Bayesian framework, which is [ideal for MMM analysis](https://developers.google.com/meridian/docs/basics/rationale-for-causal-inference-and-bayesian-modeling). Bayesian methods allow for probabilistic reasoning, meaning they can handle uncertainty and incorporate new data as it becomes available. This makes both Meridian and PyMC Marketing well-suited for the dynamic and often noisy world of marketing data.

### 3. Both Are Open-Source and Transparent

Transparency is a major advantage of both tools. Google Meridian, despite being developed by a tech giant, is open-source, allowing users to inspect the code and understand how the models work. Similarly, PyMC Marketing is built on the open-source [PyMC library](https://www.pymc.io/welcome.html), giving users full visibility into the modeling process. This openness not only builds trust but also encourages community contributions, leading to continuous improvement over time.

## Which Tool Should You Choose?

The choice between Google Meridian and PyMC Marketing ultimately depends on your organization’s needs and technical expertise. If you’re looking for a tool that integrates seamlessly with Google’s ecosystem, accounts for organic search, and incorporates reach/frequency data, Google Meridian is likely the better choice. However, if you value flexibility, have the technical know-how to build custom models, and want full control over your MMM framework, PyMC Marketing is the way to go.

Both tools represent the future of MMM in a privacy-first world, offering marketers powerful ways to measure and optimize their campaigns. Whether you choose Meridian or PyMC Marketing, you’ll be well-equipped to navigate the evolving landscape of advertising.


