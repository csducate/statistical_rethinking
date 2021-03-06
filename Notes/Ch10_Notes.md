---
title: 'Chapter 10: Big Entropy and the Generalized Linear Model'
subtitle: "Statistical Rethinking Notes"
author: "Caitlin S. Ducate"
date: "29 June, 2020"
output: 
  html_document: 
    keep_md: yes
    toc: yes
---

# Video Lectures 

##  Lecture 11

### Maximum Entropy
* Principle of Bayesian inference: some distributions have a lot more ways that they can come about than other distributions
  - Think combinatorics: $W = \frac{N!}{n_1!n_2!n_3!n_4!n_5!}$
  - More dispersed distributions have *way* more ways of happening than less dispersed distributions
    + Flat distributions are the most dispersed and thus have the highest entropy
    + Thus, we want the flattest possible distribution that is also realistic & matches pre-determined constraints (**Maxent Principle**)
* Posterior distribution is the flattest distribution (highest entropy) based on the data

### Generalized Linear Models
* Goals: Connect a linear model to an outcome variable
* Strategy:
  1. Pick an outcome distribution
  2. Model parameters using links to linear models
  3. Compute posterior
* How to pick a data distribution
  - Mostly use exponential family--have maxent (maximum entropy) interpretations in some circumstances
    + Distances & Durations: Exponential, Gamma, 
    + Counts: Binomial, Poisson, Multinomial (ext. of binomial), Geometric (like exp but for discrete counts)
    + Monsters: Ranks and ordered categories
    + Mixtures: Beta-binomial, gamma-Poisson, zero-inflated processes, occupancy models
  - Can arise from natural processes
  - Resist "histomancy"--don't pick a likelihood based on histogram
* Working with the log-odds scale
  - log-odds is $log(\frac{p_i}{1-p_i})$
  - Anchor points for interpretation
    + log-odds of 0 == p of 0.5
    + log-odds of +-1 == p of 0.73
    + log-odds of +-3 == p of 0.95
    + log-odds of +-4 == always or never
* Alternatives to logit link
  - Probit (common in econ)
  - Complementary-log-log (clog)
* Picking priors
  - Prior predictive simulation--hard to intuit what a reasonable logit prior would be


  

# Book Notes



