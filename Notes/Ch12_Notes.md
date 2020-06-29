---
title: 'Chapter 12: Monsters & Mixtures'
subtitle: "Statistical Rethinking Notes"
author: "Caitlin S. Ducate"
date: "29 June, 2020"
output: 
  html_document: 
    keep_md: yes
    toc: yes
---

# Lecture Notes

## Lecture 13

* More complicated GLMs:
  - **Monsters**: Specialized, complex distributions, such as ordered categories and ranks
  - **Mixtures**: Blends of stochastic processes
    + Varying means, probabilities, rates
    + Varying process: zero-inflation (with many ways to get zero), hurdles
* Zero-inflated Poisson models
  - How to handle "false zeros", where you get a value of 0, but it does not mean there really was 0 of that phenomenon
  - **Note**: Super important for crime--need to handle true 0s (no crime) and false 0s (no *reported* crime)
  - Mixed model which is Bernoulli + Poisson
  - See **Overthinking** box on pg. 383
  



# Book Notes







