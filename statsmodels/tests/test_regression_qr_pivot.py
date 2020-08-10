import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.sandbox.regression.predstd import wls_prediction_std

from statsmodels.datasets import longley
#from statsmodel.regression.tests.results.results_regression import Longley

np.random.seed(9876789)

# Test regression with no multicollinearity
nsample = 100
x = np.linspace(0, 10, 100)
X = np.column_stack((x, x**2))
beta = np.array([1, 0.1, 10])
e = np.random.normal(size=nsample)

X = sm.add_constant(X)
y = np.dot(X, beta) + e

model = sm.OLS(y, X)
results = model.fit()
print(results.summary())
results.qr = model.fit(method="qr")
print(results.qr.summary())
results_pivot = model.fit(method="qr-pivot")
print(results_pivot.summary())


assert all((results.params - results_pivot.params) < 1e-8), 'Some of the params are not identical'
assert all((results.pvalues - results_pivot.pvalues) < 1e-8), 'Some of the params are not identical'

predict = results.predict(X)
predict_pivot = results_pivot.predict(X)


###################################
data = longley.load(as_pandas=False)

data.exog = sm.add_constant(data.exog, prepend=False)
ols_m = sm.OLS(data.endog, data.exog)
res = ols_m.fit()
res_qr = ols_m.fit(method='qr')
res_pivot = ols_m.fit(method='qr-pivot')

print(res.summary())
print(res_qr.summary())
print(res_pivot.summary())
print('end')

# Test multiple regression with multicollinearity X3 is highly correlated with

X3 = sm.add_constant(np.column_stack((x, x**2, x + x**2 + np.random.normal(size=nsample)/10000)))

model3 = sm.OLS(y, X3)
results3 = model3.fit()
print(results3.summary())
print("Condition Number is high: %d" %results3.condition_number)

results3_pivot = model3.fit(method='qr-pivot', tol=1e-1)
print(results3_pivot.summary())
print("Condition Number is high: %d" %results3_pivot.condition_number)

Y = [1,3,4,5,2,3,4]
X = range(1,8)
X = sm.add_constant(X)
wls_model = sm.WLS(Y,X, weights=list(range(1,8)))
results = wls_model.fit()
results_pivot = wls_model.fit(method='qr-pivot')

print(results.params)
print(results_pivot.params)
#array([ 2.91666667,  0.0952381 ])
print(results.tvalues)
print(results_pivot.tvalues)

assert all((results.params - results_pivot.params) < 1e-8), 'Some of the params are not identical'
assert all((results.pvalues - results_pivot.pvalues) < 1e-8), 'Some of the params are not identical'

# test GLM with logistic regression
spector_data = sm.datasets.spector.load_pandas()
spector_data.exog = sm.add_constant(spector_data.exog)

#stert with a sinmple example
logit_model = sm.GLM(spector_data.endog, spector_data.exog, family=sm.families.Binomial())
logit_res = logit_model.fit()
print(logit_res.summary())

logit_res_pivot = logit_model.fit(wls_method='qr-pivot')
print(logit_res_pivot.summary())

# add
X4 = spector_data.exog.copy()
X4['MC'] = X4.iloc[:, 1]+X4.iloc[:, 2]+X4.iloc[:, 3]

logit_model_MC = sm.GLM(spector_data.endog, X4, family=sm.families.Binomial())
logit_res_MC = logit_model_MC.fit()
print(logit_res_MC.summary())

logit_res_MC_pivot = logit_model_MC.fit(wls_method='qr-pivot')
print(logit_res_MC_pivot.summary())