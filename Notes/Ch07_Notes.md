---
title: "Chapter 7: Ulysses' Compass"
subtitle: "Statistical Rethinking Notes"
author: "Caitlin S. Ducate"
date: "12 June, 2020"
output: 
  html_document: 
    keep_md: yes
    toc: yes
---

# Video Lectures

## Lecture 07

* Note: first half covered last of content related to Ch06

### Goals

* Understand *overfitting* and *underfitting*
* Introduct *regularization*
* Cross-validation & information criteria
  - Estimate predictive accuracy
  - Estimate overfitting risk
  - Understand how overfitting relates to complexity
  - Identify influential observations
* See that prediction & causal inference are different objectives

### Notes

* *Stargazing*: using asterisks (i.e. p-values) to select model terms
  - p-values are *not* designed to do this! They have nothing to say about predictive accuracy of the model
* Overfitting & Underfitting: the Scylla & Charybdis of Statistics
  - Will always have to navigate the narrow strait between the two
* The problem with parameters
  - Every time to add a parameter, the model will fit the sample better
  - So, cannot use fit to sample as indicator of model accuracy
  - *Underfitting*: learning too little from the data
    + Fit and predict poorly
  - *Overfitting*: learning too much from the data
    + Fit better and predict worse
    + Multilevel models not as susceptible to overfitting--one reason why they are great
* Variance "explained": $R^2$

     $R^2 = \frac{var(outcome)-var(residuals)}{var(outcome)} = 1 - \frac{var(residuals)}{var(outcome)}$
  - You can get an $R^2 = 1$ if you have as many parameters as data points 
* Model fitting: formal definitions
  - *Underfitting*: insensitive to data
  - *Overfitting*: very sensitive to exact data
  - *Regular fit*: sensitive only to the regular features of the sample, not the features idiosyncratic to the sample
  
## Lecture 08

* Information theory
  - Describes degree of uncertainty when outcome is known
  - Information entropy: uncertainty in a probability distribution is average (minus) log-probability of an event
  
      $H(p) = -Elog(p_i) = -\sum p_i(log(p_i) - log(q_i))$
    + Entropy to accuracy
    
      $D_{KL}(p, q) = \sum_{i} p_i(log(p_i) - log(q_i))$
      * Two probability distributions: *p*, *q*
      * *p* is true, *q* is model
      * Distance from *q* to *p* describes the **Divergence** or how good of a model *q* is
  - Simpler models are more accurate because they have higher entropy--they expect more things to possibly happen and are thus less surprised by more things
* Estimating divergence
  - Because we don't know *p*, we need to estimate
  - Can use the log-score (sum of log probabilities of each observation) to score accuracy of a model: $S(q) = \sum log(q_i)$
  - In practice of Bayesian, there is a *distribution* of log scores, the **log-pointwise-predictive density**
  
    $lppd(\gamma, \Theta) = \sum_i log(\frac{1}{S}) \sum_s p(y_i|\theta_s)$

* Regularizing
  - Use informative, conservative priors to reduce overfitting
    + Model will learn less from sample but will perform better out of sample
    + This is why weakly-informative priors perform better than flat priors--model will perform better out of sample
  - Why don't scientists do this more?
    + Not taught
    + Makes significant results harder
    + Scientists not judged on accuracy
* How to estimate out-of-sample deviance?
  - In theory: cross validation
    + Leave out some observations
    + Fit model on remaining observations
    + Score model fit based on how well it fits the left-out data
    + Average over many permutations of left-out sets to get an estimate of out-of-sample accuracy
    + Most common way is to leave one out at a time--but for large samples, this takes forEVER
      * Can use Pareto-smoothed importance sampling (PSIS): `LOO` function in `rethinking`, also the `loo` package
  - Also in theory: Information criteria
    + Approach originally by Akaike information criterion, but makes a lot of restrictive assumptions & so is no longer useful
    + WAIC (Widely Applicable Information Criterion) does not assume Gaussian posterior: $WAIC(y,\Theta)=-2(lppd - \Sigma_i var_{\theta}log p(y_i|\Theta))$
  - Both perform similarly & very well
    + When they disagree, it means there is some high-leverage point that they are handling differently--indicates you shouldn't trust either method and dig into your data
* How to use CV & WAIC
  - Avoid "model selection" (unless you *only* care about prediction)
  - Practice model comparison
    + Multiple models for causal inference
    + Multiple models for explanations
  - Use the `compare()` function in `rethinking` to compare `quap()` models
    + Use cautiously--the model which makes the best predictions is not always the most explanatory model




# Book Notes


