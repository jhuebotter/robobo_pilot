# Robobo Pilot

Robobo Pilot is an addition to the Robobo.py library used to create programs for the Robobo educational robot (http://www.theroboboproject.com) in the Python language.

Robobo control is available manually via console or automatically via model by using only the infrared sensors.

## Installation

The scripts run with **Python 3.6** or greater and use **numpy** and **pandas**. Robobo.py also depends on the **websocket-client** library, that must be installed.

```
pip install websocket-client, numpy, pandas
```

First download or clone this repository.

```
git clone https://github.com/jhuebotter/robobo_pilot.git
```

Change into the robobo_pilot directory and get the Robobo.py library.

```
cd robobo_pilot
git clone https://github.com/mintforpeople/robobo.py.git
```

## Basic usage

To use this code to control a Robobo please connect the Robobo with a phone using **not** the developer app. Then open either the manual_control.py or model_control.py and enter the correct ip address shown the paired device. If you hit start on the app and run either script, the robot is ready to move.

For **manual control** run:
```
python -i manual_control.py
```
A Robobo object called robobo should be available and connected. You may now use any of the commands specified at https://github.com/mintforpeople/robobo.py. For example:

```
robobo.sayText("Hello world")
robobo.moveWheelsByTime(40, -40, 3)
```

For **model control** run:
```
python model_control.py
```
By default, a dummy model is loaded and starts moving the robot. In general, the function load_model() is waiting for further models to be implemented. Each model object should have able to action = model.predict(state) where state is a numpy array of infrared sensor readings normalized between 0.0 and 1.0 in the same order as the SENSORS constant and action is a list or tuple of two values between -1.0 and 1.0.



