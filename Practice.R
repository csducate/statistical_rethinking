# Setup
install.packages(c("devtools","mvtnorm","loo","coda"),dependencies=TRUE)
library(devtools)

# Most recent package as of 10.30.19
install_github("rmcelreath/rethinking",ref="Experimental")
library(rethinking)


dbinom(6, size=9, prob = 0.5)

options(scipen = 999)

# BUILDING MODEL FOR GLOBE-TOSSING EXAMPLE
# GRID EXAMPLE
# P. 40
# define grid
p_grid <- seq( from=0 , to=1 , length.out=20 )
# define prior
prior <- rep( 1 , 20 )
# compute likelihood at each value in grid
likelihood <- dbinom( 6 , size=9 , prob=p_grid )
# compute product of likelihood and prior
unstd.posterior <- likelihood * prior
# standardize the posterior, so it sums to 1
posterior <- unstd.posterior / sum(unstd.posterior)
# Plot posterior distribution
plot( p_grid , posterior , type="b" ,
      xlab="probability of water" , ylab="posterior probability" )
mtext( "20 points" )

# Prior 2
# define grid
p_grid <- seq( from=0 , to=1 , length.out=20 )
# define prior
prior <- ifelse( p_grid < 0.5 , 0 , 1 )
# compute likelihood at each value in grid
likelihood <- dbinom( 6 , size=9 , prob=p_grid )
# compute product of likelihood and prior
unstd.posterior <- likelihood * prior
# standardize the posterior, so it sums to 1
posterior <- unstd.posterior / sum(unstd.posterior)
# Plot posterior distribution
plot( p_grid , posterior , type="b" ,
      xlab="probability of water" , ylab="posterior probability" )
mtext( "20 points" )

# Prior 3
# define grid
p_grid <- seq( from=0 , to=1 , length.out=20 )
# define prior
prior <- exp( -5*abs( p_grid - 0.5 ) )
# compute likelihood at each value in grid
likelihood <- dbinom( 6 , size=9 , prob=p_grid )
# compute product of likelihood and prior
unstd.posterior <- likelihood * prior
# standardize the posterior, so it sums to 1
posterior <- unstd.posterior / sum(unstd.posterior)
# Plot posterior distribution
plot( p_grid , posterior , type="b" ,
      xlab="probability of water" , ylab="posterior probability" )
mtext( "20 points" )

# MOVE TO QUADRATIC APPROXIMATION
library(rethinking)
globeQuap <- quap(
  alist(
    W ~ dbinom(W+L, p),
    p ~ dunif(0, 1)
  ),
  data=list(W = 6, L = 3)
)
precis(globe.qa)
## Analytical calculation
W <- 6
L <- 3
curve(dbeta(x, W+L, L+1), from = 0, to=1)
curve( dnorm( x , 0.67 , 0.16 ) , lty=2 , add=TRUE )
## MCMC
n_samples <- 100000
p <- rep( NA, n_samples )
p[1] <- 0.5
W <- 6
L <- 3
for ( i in 2:n_samples ) {
  p_new <- rnorm( 1 , p[i-1] , 0.1 )
  if ( p_new < 0 ) p_new <- abs( p_new )
  if ( p_new > 1 ) p_new <- 2 - p_new
  q0 <- dbinom( W , W+L , p[i-1] )
  q1 <- dbinom( W , W+L , p_new )
  p[i] <- ifelse( runif(1) < q1/q0 , p_new , p[i-1] )
}

dens( p , xlim=c(0,1) )
curve( dbeta( x , W+1 , L+1 ) , lty=2 , add=TRUE )

#========================================================================================
#
# CHAPTER 3
# 
#========================================================================================

# Probability of being a vampire given positive test
Pr_Positive_Vampire <- 0.95
Pr_Positive_Mortal <- 0.01
Pr_Vampire <- 0.001
Pr_Positive <- Pr_Positive_Vampire * Pr_Vampire +
  Pr_Positive_Mortal * ( 1 - Pr_Vampire )
( Pr_Vampire_Positive <- Pr_Positive_Vampire*Pr_Vampire / Pr_Positive )

# Grid approximation
p_grid <- seq( from=0 , to=1 , length.out=1000 )
prob_p <- rep( 1 , 1000 )
prob_data <- dbinom( 6 , size=9 , prob=p_grid )
posterior <- prob_data * prob_p
posterior <- posterior / sum(posterior)
samples <- sample( p_grid , prob=posterior , size=1e4 , replace=TRUE )
plot(samples)
library(rethinking)
dens( samples )

# add up posterior probability where p < 0.5
sum( posterior[ p_grid < 0.5 ] )
sum( samples < 0.5 ) / 1e4 ## OR
sum(samples < 0.5) / length(samples)

# Add up posterior probability where p > 0.5 & p < 0.75
sum( samples > 0.5 & samples < 0.75 ) / length(samples)

# Bottom 80%
quantile( samples , 0.8 )

# Middle 80%
quantile( samples , c( 0.1 , 0.9 ) )

# Data with 3 waters in 3 tosses (globe-tossing example)
p_grid <- seq( from=0 , to=1 , length.out=1000 )
prior <- rep(1,1000)
likelihood <- dbinom( 3 , size=3 , prob=p_grid )
posterior <- likelihood * prior
posterior <- posterior / sum(posterior)
samples <- sample( p_grid , size=1e4 , replace=TRUE , prob=posterior )
PI(samples, prob = 0.5)
HPDI(samples, prob = 0.5)

# maximum a posteriori (MAP) estimate
p_grid[ which.max(posterior) ]
# Mode from the posterior
chainmode( samples , adj=0.01 )
mean( samples ) 
median( samples )
hist(samples, breaks = 10000)
plot(samples)

# Generating dummy data (p. 62)
dbinom( 0:2 , size=2 , prob=0.7)
# 1 random draw
rbinom( 1 , size=2 , prob=0.7 )
# 10 random draws
rbinom( 10 , size=2 , prob=0.7 )
# 10,000 random draws
dummy_w <- rbinom( 1e5 , size=2 , prob=0.7 )
# Prop matches actual probability given binom distr w/ prob 0.7
table(dummy_w)/1e5
# Play with smaller and larger # of samples
## You can see sample variation 
dummy_w <- rbinom( 1e5 , size=9 , prob=0.7 )
table(dummy_w)/1e5 # divide table by # of samples
simplehist( dummy_w , xlab="dummy water count" )

# Sim predictions of 1e4 samples of 9 tosses
w <- rbinom( 1e4 , size=9 , prob=0.6 )
simplehist(w)

# Stopped on pg. 71

#========================================================================================
#
# CHAPTER 5: The Many Variables & Spurious Waffles
# 
#========================================================================================

library(rethinking)
data("WaffleDivorce")
d = WaffleDivorce

# Standardize variables
d$A = scale(d$MedianAgeMarriage)
d$D = scale(d$Divorce)

# P.130 example
# Note: need to call variables in the data by their names
m5.1 = quap(
  alist(
    D ~ dnorm(mu, sigma),
    mu <- a + bA*A,
    a ~ dnorm(0, 0.2),
    bA ~ dnorm(0, 0.5),
    sigma ~ dexp(1)
  ),
  data = d
)
precis(m1)

# Prior predictive checks
set.seed(10)
prior = extract.prior(m5.1)
mu = link(m5.1, post=prior, data = list(A=c(-2, 2)))
plot(NULL, xlim=c(-2, 2), ylim = c(-2, 2))
for (i in 1:50) {
  lines(c(-2, 2), mu[i,], col=col.alpha("black", 0.4))
}

# Posterior computation
A_seq <- seq(from = -3, to = 3.2, length.out = 30)
mu <- link(m5.1, data = list(A=A_seq))
muMean <- apply(mu, 2, mean)
muPI <- apply(mu, 2, PI)
precis(m5.1)
plot(m5.1)

# Modelling the relationship between marriage rate and divorce
d$M = scale(d$Marriage)
m5.2 = quap(
  alist(
    D ~ dnorm(mu, sigma),
    mu <- a + bM*M,
    a ~ dnorm(0, 0.2),
    bM ~ dnorm(0, 0.5),
    sigma ~ dexp(1)
  ),
  data = d
)
precis(m5.2)
plot(m5.2)

# Modelling the relationship bewteen marriage rate, age of marriage, & divorce
m5.3 <- quap(
  alist(
    D ~ dnorm(mu, sigma),
    mu <- a + bM*M + bA*A,
    a ~ dnorm(0, 0.2),
    bM ~ dnorm(0, 0.5),
    bA ~ dnorm(0, 0.5),
    sigma ~ dexp(1)
  ), 
  data = d
)
precis(m5.3)
plot(coeftab(m5.1, m5.2, m5.3), par = c("bA", "bM"))

# Simulation the divorce data
N <- 50
# If you just use rnorm with a sample size, draws a sample from the standard normal curve
## This is useful bc values are automatically standardized
age <- rnorm(N)
# Simulated a vector of z-scores, where each zscore was dependent on the age of the person in the age vector
mar <- rnorm(N, age)

# Predictor Residual Plots
m5.4 <- quap(
  alist(
    M ~ dnorm(mu, sigma),
    mu <- a + bA,
    a ~ dnorm(0, 0.2),
    bA ~ dnorm(0, 0.5),
    sigma ~ dexp(1)
  ),
  data = d
)
precis(m5.4)
## Calculate the residual
mu <- link(m5.4)
muMean <- apply(mu, 2, mean)
muResid <- d$M - muMean

# Counterfactual Plots
## Prepare new counterfactual data
M_seq <- seq(-2, 3, length.out = 30)
pred_data <- data.frame(M = M_seq, A = 0)

# compute counterfactual mean divorce (mu)
mu <- link(m5.3, data = pred_data)
muMean <- apply(mu, 2, mean)
muPI <- apply(mu, 2, PI)

# simulate counterfactual divorce outcomes
D_sim <- sim(m5.3, data = pred_data, n=1e4)
D_PI <- apply(D_sim, 2, PI)

# display predictions; hide raw data with type "n"
plot(D ~ M, data = d, type = "n")
mtext("Median age marriage(std) = 0")
lines(M_seq, muMean)
shade(muPI, M_seq)
shade (D_PI, M_seq)

# ------
# Section 5.2
# ------

library(rethinking)
data("milk")
d <- milk
str(d)

# Standardize the variables
d$kCal <- scale(d$kcal.per.g)
d$neoCor <- scale(d$neocortex.perc)
d$M <- scale(log(d$mass))

# Run bivariate regression with vague priors
## This model will throw an error because of missing values in the neocortex data
m5.5_draft <- quap(
  alist(
    kCal ~ dnorm(mu, sigma),
    mu <- a + bN*neoCor,
    a ~ dnorm(0, 1), 
    bN ~ dnorm(0, 1),
    sigma ~ dexp(1)
  ), data = d
)

# Make a new data frame without missing values
## Looks for rows with complete cases for 3 variables
dcc <- d[complete.cases(d$kCal, d$neoCor, d$M), ]

# Try the model again
m5.5_draft <- quap(
  alist(
    kCal ~ dnorm(mu, sigma),
    mu <- a + bN*neoCor,
    a ~ dnorm(0, 1), 
    bN ~ dnorm(0, 1),
    sigma ~ dexp(1)
  ), data = dcc
)

# Prior predictive checks
prior <- extract.prior(m5.5_draft)
xseq <- c(-2, 2)
mu <- link(m5.5_draft, post = prior, data = list(neoCor=xseq))
plot(NULL, xlim = xseq, ylim = xseq)
for(i in 1:50){
  lines(xseq, mu[i, ], col=col.alpha("black", 0.3))
}
# Produces crazy impossible outcomes


# Tweak the priors to be more reasonable
m5.5 <- quap(
  alist(
    kCal ~ dnorm(mu, sigma),
    mu <- a + bN*neoCor,
    a ~ dnorm(0, 0.2), 
    bN ~ dnorm(0, 0.5),
    sigma ~ dexp(1)
  ), data = dcc
)

# Plot again
prior <- extract.prior(m5.5)
xseq <- c(-2, 2)
mu <- link(m5.5, post = prior, data = list(neoCor=xseq))
plot(NULL, xlim = xseq, ylim = xseq)
for(i in 1:50) {
  lines(xseq, mu[i,], col=col.alpha("black", 0.3))
}

# Not a strong bivariate relationship....
precis(m5.5)

# Plot the posterior
xseq <- seq(from=min(dcc$neoCor) - 0.15, to=max(dcc$neoCor) + 0.15, length.out = 30)
mu <- link(m5.5, data=list(neoCor=xseq))
muMean <- apply(mu, 2, mean)
muPI <- apply(mu, 2, PI)
plot(kCal ~ neoCor, data = dcc)
lines(xseq, muMean, lwd=2)
shade(muPI, xseq)
# Not very strong relationship indeed...

# New model! with mother body mass
m5.6 <- quap(
  alist(
    kCal ~ dnorm(mu, sigma),
    mu <- a + bM*M,
    a ~ dnorm(0, 0.2),
    bM ~ dnorm(0, 0.5),
    sigma ~ dexp(1)
  ), data = dcc
)
precis(m5.6)

# Plot the posterior
xseq <- seq(from=min(dcc$M) - 0.15, to=max(dcc$M) + 0.15, length.out = 30)
mu <- link(m5.6, data = list(M=xseq))
muMean <- apply(mu, 2, mean)
muPI <- apply(mu, 2, PI)
plot(kCal ~ M, data = dcc)
lines(xseq, muMean, lwd = 2)
shade(muPI, xseq)

# Multivariate model with both predictors
m5.7 <- quap(
  alist(
    kCal ~ dnorm(mu, sigma),
    mu ~ a + bN*neoCor + bM*M,
    a ~ dnorm(0, 0.2),
    bN ~ dnorm(0, 0.5),
    bM ~ dnorm(0, 0.5),
    sigma ~ dexp(1)
  ), data = dcc
)
precis(m5.7)

# Plot the coefficients in each model
plot(coeftab(m5.5, m5.6, m5.7), pars=c("bM", "bN"))

pairs(~kCal + M + neoCor, dcc)


# ------
# Section 5.3: Categorical Variables
# ------

# Load in Kalahari height data
data("Howell1")
d <- Howell1
str(d)

# Recode male variable as 1s and 2s to make it an index variable
d$sex <- ifelse(d$male==1, 2, 1)
str(d$sex)

m5.8 <- quap(
  alist(
    height ~ dnorm(mu, sigma),
    mu <- a[sex],
    a[sex] ~ dnorm(178, 20),
    sigma ~ dunif(0, 50)
  ), data = d
)
precis(m5.8, depth = 2)

# Calculate difference between males and females on height
post <- extract.samples(m5.8)
post$diff_fm <- post$a[, 1] - post$a[ , 2]
precis(post, depth = 2)
