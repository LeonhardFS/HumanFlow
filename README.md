# Prediction of missing HumanFlow data

This project aims to predict missing sensor data as stated in the 2016 Harvard IACS Computational Challenge (http://computefest.seas.harvard.edu/student-challenge). Sensors are located as shown in the image below.
![Map of sensors](https://raw.githubusercontent.com/LeonhardFS/HumanFlow/master/data/map.png?token=ANeRaQpi0JG8i3IFCtxLGF_AUUw2efYcks5WumkYwA%3D%3D)

### Model
For our winning model, we used an inverse distance weighted averaging methode (see IDWmodel.py) which was then fed as input to a linear regression model(LR_model.py). 
Further improvement could be achieved using moving averages (see LR_MA_model.py / LR_MA_model_advanced.py) and exponential decaying (see LR_MA_model_advanced_weighting.py).
Using optimized parameter settings, this lead us to victory!


*(c) 2016 N.Drizard, L.Spiegelberg / Team Merkozy*


