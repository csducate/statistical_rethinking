---
title: "Chapter 7: Ulysses' Compass"
subtitle: "Statistical Rethinking Notes"
author: "Caitlin S. Ducate"
date: "13 August, 2020"
output: 
  html_document: 
    keep_md: yes
    toc: yes
---

# Lecture Notes

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

## 7.1: The Problem with Parameters
* This chapter is about trade-offs between simplicity & accuracy
  - hence *Ulysses' Compass*, sailing him precariously between Scylla (overfitting) & Charybdis (underfitting)
* The Problem with R^2
  - Formula: $R^2 = \frac{var(outcome) - var(residuals)}{var(outcome)} = 1 - \frac{var(residuals)}{var(outcome)}$
  - $var(outcome)$ is just the variance of the outcome that the model can reproduce--everything else is "residual"
  - As you add parameters, $R^2$ increases no matter what
  - $R^2$ gets closer to 1 as the variance of the residuals shrinks, but residual variance will always shrink with more parameters because it becomes easier and easier to map each data point

### Example 1: Brain Volume & Body Mass

* Setup:

```r
# Setup
install.packages(c("devtools", "mvtnorm", "loo", "coda"), dependencies = TRUE)
library(devtools)

# Most recent package as of 10.30.19
install_github("rmcelreath/rethinking",ref="Experimental")
```


```r
library(rethinking)
```


```r
sppnames <- c("afarensis", "africanus", "habilis", "boisei", "rudolfensis", "ergaster", "sapiens")
brainvolcc <- c(438, 452, 612, 521, 752, 871, 1350)
masskg <- c(37.0, 35.5, 34.5, 41.5, 55.5, 61.0, 53.5)
d <- data.frame(species = sppnames, brain = brainvolcc, mass = masskg)
```

* Next, rescale the predictor and outcome variables
  - This improves ability to get model to fit and makes it easier to specify priors
  - However, do so responsibly--some variables (e.g. brain volume) won't make sense with a 0 reference point

```r
# Convert mass to a z-score
d$mass_std <- (d$mass - mean(d$mass))/sd(d$mass)

# Rescale brain volume in reference to the largest brain
d$brain_std <- d$brain / max(d$brain)
```

* Set the priors

```r
m7.1 <- quap(
  alist(
    brain_std ~ dnorm(mu, exp(log_sigma)),
    mu <- a + b*mass_std,
    a ~ dnorm(0.5, 1),
    b ~ dnorm(0, 10), 
    log_sigma ~ dnorm(0, 1)
  ), data = d
)
```

* Computing $R^2$ to kill it:


```r
set.seed(12)
s <- sim(m7.1)
r <- apply(s, 2, mean) - d$brain_std
resid_var <- var2(r)
outcome_var <- var2(d$brain_std)
1 - resid_var / outcome_var
```

```
## [1] 0.4774589
```

Write a function to do this $R^2$ calculation, following the principle that if you're going to do things many times in the same way, you should write a function.


```r
R2_is_bad <- function(quap_fit){
  s <- sim(quap_fit, refresh = 0)
  r <- apply(s, 2, mean) - d$brain_std
  1 - var2(r)/var2(d$brain_std)
}
```

Now to make new models, each with a higher polynomial degree than the last.


```r
m7.2 <- quap(
  alist(
    brain_std ~ dnorm(mu, exp(log_sigma)),
    mu <- a + b[1]*mass_std + b[2]*mass_std^2,
    a ~ dnorm(0.5, 1),
    b ~ dnorm(0, 10),
    log_sigma ~ dnorm(0, 1)
  ), data = d, start = list(b = rep(0, 2)) # <- tells quap() how long the b vector is
)
```


```r
m7.3 <- quap(
  alist(
    brain_std ~ dnorm(mu, exp(log_sigma)),
    mu <- a + b[1]*mass_std + b[2]*mass_std^2 + 
              b[3]*mass_std^3,
    a ~ dnorm(0.5, 1),
    b ~ dnorm(0, 10),
    log_sigma ~ dnorm(0, 1)
  ), data = d, start = list(b = rep(0, 3))
)

m7.4 <- quap(
  alist(
    brain_std ~ dnorm(mu, exp(log_sigma)),
    mu <- a + b[1]*mass_std + b[2]*mass_std^2 +
              b[3]*mass_std^3 + b[4]*mass_std^4,
    a ~ dnorm(0.5, 1),
    b ~ dnorm(0, 10),
    log_sigma ~ dnorm(0, 1)
  ), data = d, start = list(b = rep(0, 4))
)

m7.5 <- quap(
  alist(
    brain_std ~ dnorm(mu, exp(log_sigma)),
    mu <- a + b[1]*mass_std + b[2]*mass_std^2 +
              b[3]*mass_std^3 + b[4]*mass_std^4 +
              b[5]*mass_std^5,
    a ~ dnorm(0.5, 1),
    b ~ dnorm(0, 10),
    log_sigma ~ dnorm(0, 1)
  ), data = d, start = list(b = rep(0, 5))
)
```

Model m7.6 has to have it's standard deviation fixed to a constant value of 0.001 (see below for reason)


```r
m7.6 <- quap(
  alist(
    brain_std ~ dnorm(mu, 0.001),
    mu <- a + b[1]*mass_std   + b[2]*mass_std^2 +
              b[3]*mass_std^3 + b[4]*mass_std^4 +
              b[5]*mass_std^5 + b[6]*mass_std^6,
    a ~ dnorm(0.5, 1),
    b ~ dnorm(0, 10)
  ), data = d, start = list(b = rep(0, 6))
)
```

Now plot each model.

```r
# Plot M7.1
post <- extract.samples(m7.1)
mass_seq <- seq(from = min(d$mass_std), to = max(d$mass_std), length.out = 100)
# Make a function to do this for the others
plot_R2 <- function(model) {
  post <- extract.samples(model)
  l <- link(model, data = list(mass_std = mass_seq))
  mu <- apply(l, 2, mean)
  ci <- apply(l, 2, PI)
  plot(brain_std ~ mass_std, data = d)
  lines(mass_seq, mu)
  shade(ci, mass_seq)
}
plot_R2(m7.1)
```

![](Ch07_Notes_files/figure-html/unnamed-chunk-8-1.png)<!-- -->

```r
# Plot M7.2
post <- extract.samples(m7.2)
plot_R2(m7.2)
```

![](Ch07_Notes_files/figure-html/unnamed-chunk-8-2.png)<!-- -->

```r
# Plot m7.3
post <- extract.samples(m7.3)
plot_R2(m7.3)
```

![](Ch07_Notes_files/figure-html/unnamed-chunk-8-3.png)<!-- -->

```r
# Plot m7.6 (yes I skipped a few)
plot_R2(m7.6)
```

![](Ch07_Notes_files/figure-html/unnamed-chunk-8-4.png)<!-- -->

Because the 6th-degree polynomial passes exactly through each point, it has a perfect $R^2 = 1$. This is also why $\sigma$ had to be fixed to 0.001--if it were estimated, it would shrink to 0 because there would *be* no residual variance.

Consider model fitting as a form of data compression. You want to keep enough that you can recreate the important bits but do not waste storage space on the noise.

## 7.2 Entropy & Accuracy

* **Information**: The reduction in uncertainty derived from learning an outcome
* What makes a good measure of uncertainty?
  - Is continuous
  - Increases as the number of possible events increases
  - Is additive
* Information Entropy meets all of these criteria:
  - $H(p) = -Elog(p_i) = -\sum p_ilog(p_i)$
    + $n$ is the different possible events
    + $i$ is each event
    + $p_i$ is the probability of each event
  - The uncertainty in a probability distribution is the average log-probability of an event

* Example: if the true probabilities of rain and shine are $p_r = 0.3$ and $p_s = 0.7$, then:


```r
p <- c(0.3, 0.7)
-sum(p * log(p))
```

```
## [1] 0.6108643
```

  - BUT if we lived in Abu Dhabi, the probabilities of rain and shine might be $p_r = 0.01$ and $p_s = 0.99$:


```r
p <- c(0.01, 0.99)
-sum(p * log(p))
```

```
## [1] 0.05600153
```
  
* **Divergence*: the aditional uncertainty induced by using probabilities from one distribution to describe another distribution
  - also know as K-L (Kulback-Leibler) divergence
  - Tells us how much uncertainty we introduce when we guess incorrectly about the true distribution of events
    + $D_{KL}(p, q) = -\sum_i p_i(log(p_i) - log(q_i)) = \sum_i p_ilog(\frac{p_i}{q_i})$
  - It's the average distance in the log probability between the target ($p$) and the model ($q$)
  - Note that $H(p, q) \neq H(q, p)$, which means that we can minimize K-L divergence if we use a high-entropy model
* What's the point of all of this?
  - We needed a way to measure the distance of a model from our target: K-L divergence
  - We needed a way to estimate the divergence with real models where we don't actually *know* the target
* What do do when we don't know $p$:
  - When comparing the divergence of two models, $Elog(p_i)$ subtracts out, just leaving the log-probabilities of the models
    + $S(q) = \sum_i log(q_i)$ is the total score for model $q$
    + This formula only makes sense when compared to another model--absolute value of entropy score is, on its own, uninterpretable
  - In a Bayesian framework, you sum the log-probability over the entire posterior (rather than just the main point estimate)
    + In other words, You need to log the average probability for each observation across the posterior
  - In `rethinking` package, use the function `lppd`: **log-pointwise-predictive-density**
  

```r
set.seed(1)
lppd(m7.1, n = 1e4)
```

```
## [1]  0.6098646  0.6483419  0.5496094  0.6234911  0.4648118  0.4347581 -0.8444553
```
  - Output: the log-probability score for each observation; if you sum across them, you get the total log-probability score for the model + data
    + Larger numbers are better
    + Can also calculate **Deviance**, which is the lppd score multiplied by -2, so smaller values are better (this is there for historical reasons, not mathematical ones)
  - Overthinking: Caluculating lppd in a Bayesian framework

```r
set.seed(1)
logprob <- sim(m7.1, ll = TRUE, n = 1e4)
n <- ncol(logprob)
ns <- nrow(logprob)
f <- function(i) log_sum_exp(logprob[,i]) - log(ns)
(llpd <- sapply(1:n, f))
```

```
## [1]  0.6098646  0.6483419  0.5496094  0.6234911  0.4648118  0.4347581 -0.8444553
```

```r
sapply(list(m7.1, m7.2, m7.3, m7.4, m7.5, m7.6), function(m) sum(lppd(m)))
```

```
## [1]  2.424818  2.646561  3.694617  5.316186 14.112180 39.509528
```

More complex models have higher lppd scores, but that will always be true. So we can't score models based on how they perform on training data, but rather how they perform on test data. Training data will always get better as you add parameters, but out-of-sample test data will *not*. The true model should minimize out-of-sample deviance, but it will not necessarily be the one that minimizes deviance *the most*--prediction and accuracy are two different goals.

### Overthinking: Simulated training and testing--DO NOT EVALUATE

```r
N <- 20
kseq <- 1:5
dev <- sapply(kseq, function(k) {
  print(k);
  # For sim_train_test, N is number of cases simulated and k is number of parameters to fit
  r <- mcreplicate(1e4, sim_train_test(N = N, k = k), mc.cores = 4);
  c(mean(r[1,]), mean(r[2,]), sd(r[1,]), sd(r[2,]))
})
```


## 7.3 Golem Taming: Regularization

* Overfitting occurs when the model gets overexcited by the data
  - This is especially likely to happen with flat priors--model assumes all parameters are equally plausible
  - Can use **regularizing priors* to make the model more skeptical
* Skeptical, regularizing priors perform worse on training data but perform better on test or out-of-sample data

## 7.4 Predicting Predictive Accuracy

* **Cross-validation**: leaving out small chunks of observations (a fold) from the sample & seeing how the model performs
  - **Leave-one-out cross-validation (LOOCV)**: cross-validation that re-runs the model, leaving out a different data point each time; can compute on `quap()` models using `cv_quap`
  - The problem: LOOCV can be time-consuming if you are computing a posterior each time you leave out one of many data points
  - **Pareto-smoothed importance sampling LOOC (LOOIS or PSIS)**: an approximation of LOOCV that looks at the variance of each observation to assess impact on posterior
    + Note: Called PSIS in published 2nd edition but LOOIS in Sept 2019 manuscript
    + Weights each point to indicate which ones are mostly likely to undermine reliability of estimate
    + Because it is computed point by point, can use it to get an approximation of the standard error: $S_{PSIS} = \sqrt(Nvar(psis_i))$, where $N$ is number of observations & $psis_i$ is the PSIS estimate for observation
    
* **Information Criteria**
  - From machine learning, we know that for OLS with flat priors, the expected overfitting penalty is about twice the number of parameters
  - Akaike Information Criterion (AIC) was the first to approximate this penalty relationship
    + Is of historical interest now--needs a few strong assumptions to work
  - Widely Applicable Info Criterion (WAIC) is trying to guess the K-L divergence & is used instead of AIC
    + Takes the lppd deviance and adds a penalty proportional to the variance in the posterior distribution
    + $WAIC(y, \Theta) = -2(lppd - \sum_i var_{\Theta}logp(y_i|\Theta)) = -2lppd +2\sum_i var_{\Theta}(logp(y_i|\Theta))$
  - Though some claim that AIC & WAIC overfit, they evaluate model fit as N -> $\infty$, where they can predict parameter estimates very precisely
    + If you mostly care about prediction, not explanation, these will do a great job--and if you have too many parameters, it will just estimate the parameters as being near zero
  - Bayesian Information Criterion (BIC) is not drawn from information theory but rather uses the log of the average likelihood of a linear model
    + The average likelihood is the denominator of Bayes theorem--the likelihood averaged over the prior
  - Ratio of average likelihoods is the Bayes Factor
* Comparing LOOCV, PSIS, & WAIC, they all perform similarly with ordinary linear models with approximately Gaussian distributions & with no strong observations that unduly influence the posterior.
  - LOOCV & PSIS have higher variance in their estimation of KL divergence, but WAIC is more biased
  - Useful to compare WAIC & PSIS--if they differ greatly, probably can't trust either

## 7.5 Model Comparison

* Summary: How to pick the best model?
  - Fit to sample ins't a good metric because it will always favor more complex models
  - Information divergence is the right way to *measure* model accuracy, but will also favor more complex models when used alone
  - Can use regularizing priors to be skeptical of extreme parameters, then use CV, PSIS, or WAIC to make a guess about out-of-sample predictive accuracy
  - However, don't want to use CV, PSIS, or WAIC to *select* a model but rather to compare *several* models
    + While we care about predictive accuracy, we also often care about causal inference, which are different goals
* Get models from Ch. 6 for comparison

```r
set.seed(71)

N <- 100 # Number of plants

# Simulate initial heights
h0 <- rnorm(N, 10, 2)

# Assign treatments--half have the treatment (1) and half don't (0)
treatment <- rep(0:1, each = N/2)

# Simulate presence of fungus
## For each of N plants, there is an initial 50% chance of having fungus; this number goes down to 10% if the plant is treated with an antifungal
fungus <- rbinom(N, size = 1, prob = 0.5 - treatment*0.4)
# Final height 
## A function of initial height + a random number centered around 5 (for no fungus) and 2(if there is fungus)
h1 <- h0 + rnorm(N, 5 - 3*fungus)

# Put values in clean data frame
d <- data.frame(h0 = h0, h1 = h1, treatment = treatment, fungus = fungus)
precis(d)
```

```
##               mean        sd      5.5%    94.5%    histogram
## h0         9.95978 2.1011623  6.570328 13.07874 ▁▂▂▂▇▃▂▃▁▁▁▁
## h1        14.39920 2.6880870 10.618002 17.93369     ▁▁▃▇▇▇▁▁
## treatment  0.50000 0.5025189  0.000000  1.00000   ▇▁▁▁▁▁▁▁▁▇
## fungus     0.23000 0.4229526  0.000000  1.00000   ▇▁▁▁▁▁▁▁▁▂
```

```r
m6.6 <- quap(
  alist(
    h1 ~ dnorm(mu, sigma),
    mu <- h0*p,
    p ~ dlnorm(0, 0.25),
    sigma ~ dexp(1)
  ),
  data = d
)

precis(m6.6)
```

```
##           mean         sd     5.5%    94.5%
## p     1.426628 0.01759834 1.398503 1.454754
## sigma 1.792106 0.12496794 1.592383 1.991829
```

```r
m6.7 <- quap(
  alist(
  h1 ~ dnorm(mu, sigma),
  mu <- h0*p,
  p <- a + bT*treatment + bF*fungus,
  a ~ dlnorm(0, 0.25),
  bT ~ dnorm(0, 0.5),
  bF ~ dnorm(0, 0.5),
  sigma ~ dexp(1)
  ),
  data = d
)
precis(m6.7)
```

```
##               mean         sd        5.5%       94.5%
## a      1.482826890 0.02452189  1.44363617  1.52201761
## bT     0.001060274 0.02987665 -0.04668839  0.04880894
## bF    -0.267912117 0.03654977 -0.32632571 -0.20949852
## sigma  1.408655093 0.09859123  1.25108727  1.56622292
```

```r
m6.8 <- quap(
  alist(
    h1 ~ dnorm(mu, sigma), 
    mu <- h0 * p,
    p <- a + bT*treatment,
    a ~ dlnorm(0, 0.2),
    bT ~ dnorm(0, 0.5),
    sigma ~ dexp(1)
  ),
  data = d
)
precis(m6.8)
```

```
##             mean         sd       5.5%     94.5%
## a     1.38035157 0.02517700 1.34011386 1.4205893
## bT    0.08499494 0.03429912 0.03017831 0.1398116
## sigma 1.74641704 0.12193300 1.55154456 1.9412895
```

* Now calculate WAIC for each

```r
set.seed(11)
WAIC(m6.7)
```

```
##       WAIC      lppd  penalty  std_err
## 1 361.4371 -177.1597 3.558906 14.17059
```

```r
set.seed(77)
compare(m6.6, m6.7, m6.8, func = WAIC)
```

```
##          WAIC       SE    dWAIC      dSE    pWAIC       weight
## m6.7 361.8803 14.26260  0.00000       NA 3.845703 1.000000e+00
## m6.8 402.7753 11.28120 40.89493 10.51667 2.645589 1.317589e-09
## m6.6 405.9174 11.66153 44.03705 12.24953 1.582861 2.738272e-10
```

* Interpreting the findings
  - `WAIC` is the WAIC value
  - `SE` is the standard error of the WAIC value
  - `dWAIC` is the difference between each model's WAIC and the best WAIC--means that best model has a value of 0
  - `dSE` is the SE of the above difference
  - `pWAIC`is the penalty term of WAIC
    + Close to but slightly below the number of dimensions in the posterior of each model
  - `weight`

```r
# Manually calculating dWAIC
set.seed(91)
waic_m6.7 <- WAIC(m6.7, pointwise = TRUE)$WAIC
waic_m6.8 <- WAIC(m6.8, pointwise = TRUE)$WAIC

n <- length(waic_m6.7)
diff <- waic_m6.7 - waic_m6.8
sqrt(n*var(diff))
```

```
## [1] 10.39596
```

```r
# 99% interval around dWAIC
40.9 + c(-1, 1)*10*2.6
```

```
## [1] 14.9 66.9
```

```r
# Plot the deviance estimates
plot(compare(m6.6, m6.7, m6.8))
```

![](Ch07_Notes_files/figure-html/unnamed-chunk-15-1.png)<!-- -->

The triangle with line segments between each of the models indicates the difference in error & standard error of that difference for one model vs. another. In this case, it says that `m6.8` has better *predictive* accuracy, but tells us nothing about causal inference because we know that `m6.7` is the true model.

What about `m6.8` vs `m6.6`?


```r
set.seed(92)
waic_m6.6 <- WAIC(m6.6, pointwise = TRUE)$WAIC
diff <- waic_m6.6 - waic_m6.8
sqrt(n*var(diff))
```

```
## [1] 4.858914
```

Huh...not very different...


```r
set.seed(93)
compare(m6.6, m6.7, m6.8)@dSE
```

```
##           m6.6     m6.7      m6.8
## m6.6        NA 12.22586  4.934353
## m6.7 12.225864       NA 10.465182
## m6.8  4.934353 10.46518        NA
```

Metaphors to think about information criteria
  * IC is like a horse race--if one horse wins by a lot, it is probably the best horse, but if it's a photo-finish, it is hard to know whether the horse that won was the best horse or if there were unmeasured variables at play
  * IC is also like throwing stones--because you can't control all of the conditions of a given throw, need to compare the relative distances to figure out which stone goes furthest *on average*

### 7.5.2 Handling Outliers

```r
# Load in the waffle marriage data
data("WaffleDivorce")
d <- WaffleDivorce
d$A <- standardize(d$MedianAgeMarriage)
d$D <- standardize(d$Divorce)
d$M <- standardize(d$Marriage)
```


```r
# Build the models
m5.1 <- quap(
  alist(
    D ~ dnorm(mu, sigma),
    mu <- a + bA * A,
    a ~ dnorm(0, 0.2),
    bA ~ dnorm(0, 0.5),
    sigma ~ dexp(1)
  ), data = d
)

m5.2 <- quap(
  alist(
    D ~ dnorm(mu, sigma),
    mu <- a + bM * M,
    a ~ dnorm(0, 0.2),
    bM ~ dnorm(0, 0.5),
    sigma ~ dexp(1)
  ), data = d
)

m5.3 <- quap(
  alist(
    D ~ dnorm(mu, sigma),
    mu <- a + bM*M + bA*A,
    a ~ dnorm(0, 0.2),
    bM ~ dnorm(0, 0.5),
    bA ~ dnorm(0, 0.5),
    sigma ~ dexp(1)
  ), data = d
)
```


```r
# Compare models using PSIS
set.seed(24071847)
compare(m5.1, m5.2, m5.3, func = PSIS)
```

```
## Some Pareto k values are high (>0.5). Set pointwise=TRUE to inspect individual points.
## Some Pareto k values are high (>0.5). Set pointwise=TRUE to inspect individual points.
```

```
## Some Pareto k values are very high (>1). Set pointwise=TRUE to inspect individual points.
```

```
##          PSIS       SE     dPSIS       dSE    pPSIS       weight
## m5.1 127.5665 14.69485  0.000000        NA 4.671425 0.8340176286
## m5.3 130.8062 16.15696  3.239685  1.808671 6.578663 0.1650769863
## m5.2 141.2178 11.56557 13.651299 10.923820 4.057204 0.0009053851
```

Notice that it throws up an error: `Some Pareto k values are high (>0.5)`. This is because some values are strong, influential outliers. When they are included in the analysis, they make out-of-sample prediction worse because it is unlikely that other samples would have similar outliers.


```r
PSIS_m5.3 <- PSIS(m5.3, pointwise = TRUE)
```

```
## Some Pareto k values are high (>0.5). Set pointwise=TRUE to inspect individual points.
```

```r
WAIC_m5.3 <- WAIC(m5.3, pointwise = TRUE)
plot(PSIS_m5.3$k, WAIC_m5.3$penalty, xlab = "PSIS Pareto k", ylab = "WAIC penalty", col = rangi2, lwd = 2)
```

![](Ch07_Notes_files/figure-html/unnamed-chunk-21-1.png)<!-- -->

The massive outlier is Idaho. Old standard practice would simply drop it, but better to do **robust regression**, which is less surprised by outliers than regression with a standard Gaussian distribution. For instance, you can replace a Gaussian with a Student's T. T distributions have heavier tails controlled by a $\nu$ parameter, which usually isn't great to estimate because there aren't enough extreme values, so we have to pick a value ourselves.


```r
m5.3t <- quap(
  alist(
    D ~ dstudent(2, mu, sigma),
    mu <- a + bM*M + bA*A,
    a ~ dnorm(0, 0.2),
    bM ~ dnorm(0, 0.5),
    bA ~ dnorm(0, 0.5),
    sigma ~ dexp(1)
  ), data = d
)

PSIS(m5.3t)
```

```
##       PSIS      lppd  penalty  std_err
## 1 133.6478 -66.82391 6.794408 11.87374
```









