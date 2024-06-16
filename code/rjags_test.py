import os

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import FloatVector, IntVector
from rpy2.robjects.conversion import localconverter

import pandas as pd
from rpy2.robjects import pandas2ri

DIR = "~/Documents/Code/premier-league-modelling"

pandas2ri.activate()

# Import the base and rjags packages
base = importr('base')
rjags = importr('rjags')
coda = importr('coda')

# Define the model string in R
model_string = """
model {
  for (i in 1:N) {
    y[i] ~ dpois(lambda[i])
    log(lambda[i]) <- b0 + b1 * x[i]
  }
  b0 ~ dnorm(0, 0.01)
  b1 ~ dnorm(0, 0.01)
}
"""

# Define the data
N = 100
x = [i for i in range(1, N + 1)]
y = [i + 2 for i in range(1, N + 1)]  # example data
# bugs_model_filepath = os.path.join(DIR, "model_files/model_test.bug")
bugs_model_filepath = "model_files/model_test.bug"
with open(bugs_model_filepath) as file:
    model_test_bugs = file.read()

# Create R vectors
robjects.globalenv['N'] = N
robjects.globalenv['x'] = IntVector(x)
robjects.globalenv['y'] = IntVector(y)
base.eval(robjects.r("""model <- jags.model(textConnection(model_string), data = list(N=N, x=x, y=y), n.chains = 3, n.adapt = 1000)"""))


# Verify that the data and model string are correctly assigned in R
print(base.eval(robjects.r('N')))
print(base.eval(robjects.r('x')))
print(base.eval(robjects.r('y')))
# print(base.eval(robjects.r('model_string')))

# Create a text connection for the model string
text_connection_code = """
tc <- textConnection(model_string)
"""
robjects.r(text_connection_code)
print("Text connection created")

# model_init_code = """
# model <- jags.model(textConnection(model_string), data = list(N=N, x=x, y=y), n.chains = 1, n.adapt = 1000)
# """
#
# try:
#     model = robjects.r(model_init_code)
#     print("Model initialized successfully")
# except Exception as e:
#     print("Error initializing model:", e)
#
# # Check if model initialization was successful
# print("Model object:", model)
#
# # Check for errors in R
# error_message = base.geterrmessage()
# print("Error message from R:", error_message)
#
# # If model is not None, proceed
# if model:
#     # Update the model (Burn-in)
#     update_code = """
#     update(model, 1000)
#     """
#     try:
#         robjects.r(update_code)
#         print("Model updated successfully (Burn-in)")
#     except Exception as e:
#         print("Error updating model:", e)
#
#     # Sample from the posterior
#     samples_code = """
#     samples <- coda.samples(model, c("b0", "b1"), n.iter = 10000)
#     """
#     try:
#         samples = robjects.r(samples_code)
#         print("Samples generated successfully")
#     except Exception as e:
#         print("Error generating samples:", e)
#
#     # Convert the samples to a DataFrame
#     with localconverter(robjects.default_converter + pandas2ri.converter):
#         df_samples = pandas2ri.rpy2py(samples)
#
#     # Print the summary of the samples
#     print(df_samples.describe())
# else:
#     print("Model initialization failed. Cannot proceed with sampling.")

# Burn-in phase
base.eval(robjects.r('update(model, 1000)'))

# Sample from the posterior
samples = base.eval(robjects.r('coda.samples(model, c("b0", "b1"), n.iter = 5000)'))

with localconverter(robjects.default_converter + pandas2ri.converter):
        df_samples = pandas2ri.rpy2py(samples)
print(df_samples)
