import pandas as pd
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri, r, conversion
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter

# Activate the pandas conversion
pandas2ri.activate()

# Import R packages
base = importr('base')
rjags = importr('rjags')
coda = importr('coda')

# Define the JAGS model
model_string = """
model {
  for (i in 1:N) {
    y[i] ~ dnorm(mu[i], tau)
    mu[i] <- alpha + beta * x[i]
  }
  alpha ~ dnorm(0, 0.01)
  beta ~ dnorm(0, 0.01)
  tau <- pow(sigma, -2)
  sigma ~ dunif(0, 100)
}
"""

# Sample data
data = {
    'x': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'y': [2.1, 2.5, 3.7, 4.0, 5.1, 6.3, 6.8, 8.0, 9.1, 9.9]
}
df = pd.DataFrame(data)

N = df.shape[0]
x = df['x'].values
y = df['y'].values

# Convert data to R objects
robjects.globalenv['N'] = N
robjects.globalenv['x'] = robjects.FloatVector(x)
robjects.globalenv['y'] = robjects.FloatVector(y)

# Debug: Check if the data and model_string are correctly assigned in R
print(f'N in R: {robjects.globalenv["N"]}')
print(f'x in R: {robjects.globalenv["x"]}')
print(f'y in R: {robjects.globalenv["y"]}')

# Assign the model string in R
robjects.r('model_string <- "{}"'.format(model_string.replace("\n", "\\n")))

# Debug: Check if the model_string is correctly assigned in R
print(f'model_string in R: {robjects.r("model_string")}')

# Create and compile the JAGS model
base.eval('library(rjags)')
base.eval('model <- jags.model(textConnection(model_string), data = list(N=N, x=x, y=y), n.chains = 1, n.adapt = 1000)')

# Burn-in phase
base.eval('update(model, 1000)')

# Sample from the posterior
try:
    base.eval('samples <- coda.samples(model, c("alpha", "beta", "sigma"), n.iter = 5000)')
    print("Samples created successfully")
except robjects.rinterface_lib.embedded.RRuntimeError as e:
    print(f"Error in creating samples: {e}")

# Convert samples to a list of MCMC objects
try:
    mcmc_list = robjects.r('as.mcmc.list(samples)')
    print("MCMC list created successfully")
except robjects.rinterface_lib.embedded.RRuntimeError as e:
    print(f"Error in creating MCMC list: {e}")


# Convert MCMC list to pandas DataFrame
samples_df = pd.DataFrame()
for chain in mcmc_list:
    with conversion.localconverter(robjects.default_converter + pandas2ri.converter):
        chain_df = conversion.rpy2py(chain)
    samples_df = pd.concat([samples_df, chain_df], axis=1)

print(samples_df)
