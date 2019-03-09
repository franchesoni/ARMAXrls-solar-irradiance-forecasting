# Solar irradiance forecasting using ARMAX models as RLS filters and satellite data.
We implement an ARMAX algorithm in a Recursive Least Squares fashion to forecast solar irradiance. The exogenous variables are the variability of the series and satellite cloud estimation.

* Use data2npy.py to convert the raw data to numpy arrays
* Use get_results_bf.py to compute vectors used when displaying results. We used this to compute the .csv files containing the errors, available at http://les.edu.uy/pub/ARMAX-RLS.csv
* Display results with show_results.py and show_results_models.py

If you find this useful please cite our publication