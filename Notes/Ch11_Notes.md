---
title: 'Chapter 11: God Spiked the Integers'
subtitle: "Statistical Rethinking Notes"
author: "Caitlin S. Ducate"
date: "29 June, 2020"
output: 
  html_document: 
    keep_md: yes
    toc: yes
---

# Lecture Notes

## Lecture 12

* For doing model comparison with logit models, need to include `log_lik = TRUE` to `ulam()`
* Relative and absolute effects
  - Parameters are on a relative effect scale
  - Predictions are on an absolute effect scale
  - Think epidemiology: smoking *vastly* increases risk of getting lung cancer relative to someone who doesn't smoke, but the absolute risk of getting lung cancer is still very rare
  - The difference becomes exaggerated for rare events--very easy to increase relative effect of something while keeping the base rare (absolute risk) relatively untouched
* Aggregated Binomial
  - Mathematically the same as doing 0/1 Bernoulli trials but aggregate over the trials
* Simpson's Paradox
  - When you add a parameter, reverses an effect found before the added parameter
  - Purely a statistical phenomenon--can happen indefinitely, but we need theory & DAGs to understand what matters
* Intro to Poisson distributions
  - Counts without an upper limit but with a constant expected value
  - Has a single parameter: events per unit of time or distance
  - Use a log-link function--forces linear model to be positive because the outcome (count value) has to be positive
    + Maps all negative numbers to [0, 1], and all positive numbers mapped to [1, $\infty$]


# Book Notes


