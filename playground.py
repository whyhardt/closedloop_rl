import arviz as az

idata = az.from_netcdf('traces.nc')

print(az.summary(idata))