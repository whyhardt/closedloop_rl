import arviz as az

idata = az.from_netcdf('benchmarking/params/traces.nc')

print(az.summary(idata))

az.plot_trace(idata)
import matplotlib.pyplot as plt
plt.show()