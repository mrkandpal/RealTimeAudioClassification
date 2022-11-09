# RealTimeAudioClassification

This repository houses a code that uses the default microphone of a computer to analyse and classify audio content using the YAMNet tensorflow model. The tflite version of YAMNet is used in this work. 

The code analyses ambient audiio in 2 second increments and prints the 3 most-likely classification outputs to the console, along with the confidence rating for each classification. 

Running instructions:
The code does not require any user input. The audio analysis loop uses a simple while loop, and the desired number of seconds for which the code must be run can be changed using the 'loopLength' variable before each run. As mentioned above, the output is printed to the console. 
