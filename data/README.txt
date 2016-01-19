Student Computational Challenge

Using the Internet of Things for Human Flow Analytics
Tuesday, January 19 - Wednesday, January 20, 2016

Prediction task:
You are provided with a highly incomplete data set containing sensor readings of human movement in a building. The problem is to accurately predict the missing data by exploiting the recurring temporal and spatial patterns in the way people move around within the building. 

The data set includes readings from 56 motion detector sensors in a floor of an office building, recorded over a period of ~4 months. The sensor stream contains the number of people passing by a sensor for each minute. The sensors are prone to failure at any time of the day, and it can take up to 24 hours for a sensor to be replaced. The task is to predict the number of people passing by the sensors during every hour a sensor had failed. The participants will be evaluated based on the accuracy of predictions on the provided test set. The evaluation measure used is the Mean Absolute Error:
MEA = 1/N Σ_i |truecount_i - predictedcount_i|,
where N is the total number of test points, truecount_i is the acutual number of people who passed by a sensor, and the predictedcount_i is the predicted number of people who passed by.

Web page: http://computefest.seas.harvard.edu/student-challenge

----
DATA SET DETAILS:

For the first day (Jan 19, 12 pm) of the contest, we are providing sample data sets for first 1 month the sensors were deployed. The complete data set spanning ~4 months will be released on the second day (Jan 20, 2 pm). We have included the following files:

1. map.png
Contains the layout of 56 sensors in a floor of an office building. The sensors cover the corridors and public spaces in the building, and also include portions close to stair cases. The sensors are denoted by their ids.

2. train.txt
Contains sensor data with missing values. The rows contain a time stamp for every minute. There is one column for each sensor, containing the number of people detected by the sensor. An entry of '-1' indicates that a sensor has failed and that the value is missing. Each row includes the following fields.
<timestamp, sensor1-output, sensor2-output, ...>

Each time stamp contains the day, hour and minute at which the sensor reading was recorded: DHHMM

3. test.txt
 Contains time periods of interest where the sensors had failed. Each period spans 1 hour. The participants will be evaluated based on how accurately they are able to predict the number of people passing by the sensor during these periods. Each row in the file contain the following fields:
<sensor-id, start-time, end-time, ?>

The participants will have to predict the number of people passing by the sensor *between start-time and end-time*. Both the start and end time stamps are in the same format as above: DHHMM

4. sensor-coordinates.txt
Contain the location coordinates of each sensors. Each row contains the following fields:
<sensor-id, x-coordinate, y-coordinate>
----

----
SUBMISSION FORMAT:
A submission file with predicted counts must be in the following format:

Index,Count
1,23
2,11
3,45
...

where the first column contains the test point index, and the second column contains your predicted count. A row "a,b" would indicate that your prediction for test point 'a' is 'b'. DO NOT FORGET TO INCLUDE THE HEADER LINE. *AVOID ADDITIONAL WHITE SPACES*. ALL PREDICTIONS MUST BE NON-NEGATIVE INTEGERS (Please do not have fractional predictions). The file should contain a total of 3686 lines + a header line.
----

All the best!