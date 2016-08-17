# fitting noisy data to an exponential model
model(x, p) = p[1]*exp(-x.*p[2])
modeldot = Function[ (x,p)->exp(-x.*p[2]), (x,p)->-p[1].*x.*exp(-x.*p[2]) ]

# some example data
srand(12345)
xpts = linspace(0,10,20)
ydata = model(xpts, [1.0 2.0]) + 0.01*randn(length(xpts))
p0 = [0.5, 0.5]

fit = curve_fit(model, xpts, ydata, p0)
@assert norm(fit.param - [1.0, 2.0]) < 0.05

# can also get error estimates on the fit parameters
errors = estimate_errors(fit)
@assert norm(errors - [0.017, 0.075]) < 0.01

# use known model derivatives
fit = curve_fit(model, modeldot, xpts, ydata, p0)
@assert norm(fit.param - [1.0, 2.0]) < 0.05

# with weights
wt = collect(linspace(0.4,0.6,20))
fit = curve_fit(model, xpts, ydata, wt, p0)
@assert norm(fit.param - [1.0, 2.0]) < 0.05
fit = curve_fit(model, modeldot, xpts, ydata, wt, p0)
@assert norm(fit.param - [1.0, 2.0]) < 0.05
