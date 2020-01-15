# This imports the system libraries needed to add "robobo.py" to the system path
import sys, os
# This adds "robobo.py" to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '.', 'robobo.py'))
# This imports the Robobo library    
from Robobo import Robobo
# Additional imports
import numpy as np
import time
import pandas as pd

# Define what sensor values should be read out and in what order
#SENSORS = ['Back-R', 'Back-C', 'Back-L', 'Front-RR', 'Front-R', 'Front-C', 'Front-L', 'Front-LL']
SENSORS = ['Back-R', 'Back-L', 'Front-RR', 'Front-C', 'Front-LL']

# How fast should the robot go? between 1 and 100
MAX_SPEED = 30.

# where the results are saved
RESULT_PATH = "results/"

# set the ip of the robot in network
IP = '192.168.2.174'


class DummyModel():

    def __init__(self, **kwargs):

        # initialize with any kwargs if wished

        for k, v in kwargs.items():
            setattr(self, k, v)

    def predict(self, state):

        # Takes a state and gives back a list or tuple of (rSpeed, lSpeed) with each value between -1. and 1.
        state_dict = dict(zip(SENSORS, state))

        front = [state_dict[x] for x in SENSORS if 'Front' in x]
        back = [state_dict[x] for x in SENSORS if 'Back' in x]

        if np.max(state) < 0.3:
            action = [1., 1.]
        elif np.max(front) < np.max(back):
            action = [1., 1.]
        elif state_dict['Front-RR'] <= state_dict['Front-LL']:
            action = [-1, 1]
        elif state_dict['Front-RR'] > state_dict['Front-LL']:
            action = [1, -1]

        return action


def initialize(ip=IP):
    
    #  This creates a instance of the Robobo class with the indicated ip address
    
    robobo = Robobo(ip)
    robobo.connect()
    robobo.moveWheels(0, 0)

    #robobo.sayText('Here we go!')
    
    t0 = time.time()
    
    return robobo, t0


def get_state(robobo):

    # This returns the selected sensor readings scaled between 0 (far) and 1 (close)
    
    done = False
    while not done:
        readings = robobo.readAllIRSensor()
        if readings:
            state = np.log([int(readings[s]) for s in SENSORS])/10

            old_min = 0.2
            old_max = 1.
            new_min = 0.
            new_max = 1.

            state = [(((old_value - old_min) * (new_max - new_min)) / (old_max - old_min)) + new_min for old_value in
                      state]

            state = np.clip(state, 0.0, 1.0)

            return state
        time.sleep(0.01)


def get_action(model, state):

    # this takes a pretrained model and the state and gives back the rSpeed, lSpeed
    
    action = model.predict(state)

    return round(action[0] * MAX_SPEED), round(action[1] * MAX_SPEED)


def load_model(type):

    # Loads models of different kind. Models need to be able to action = predict(state)

    if type == 'dummy':
        model = DummyModel()
    else:
        raise NotImplementedError(f"model {type} not implemented")

    return model

def get_reward(right, left, state, delta):

    # calculate a reward for the results

    s_trans = abs(left) + abs(right)
    #print("s_trans = {s_trans}".format(s_trans=s_trans))
    s_rot = (np.max([left, right]) - np.min([left, right])) / (np.abs(left) + np.abs(right))
    #print("s_rot = {s_rot}".format(s_rot=s_rot))
    v_sens = np.min(state)
    #print("v_sens = {v_sens}".format(v_sens=v_sens))

    reward = s_trans * (1 - s_rot) * (1 - v_sens)

    if np.isnan(reward):
        reward = 0

    return reward


def write_results(results, dir):

    # write the results to a file

    try:
        results.to_csv(f"{dir}robot_results.csv")
    except:
        print("could not save results")


def main(save_results=True):

    # Here the magic happens

    # initialize some stuff
    model = load_model('dummy')
    results = pd.DataFrame([])
    robobo, t0 = initialize()
    t_1 = t0
    cum_reward = 0

    # make dir for results
    dir = f"{RESULT_PATH}{round(t0)}/"
    os.makedirs(dir, exist_ok=True)

    # first we just go full speed until we get initial sensor readings
    rSpeed, lSpeed = int(MAX_SPEED), int(MAX_SPEED)
    print(rSpeed)
    robobo.moveWheels(rSpeed, lSpeed)
    state = np.zeros(len(SENSORS))
    new_state = get_state(robobo)

    # then we start using the model
    done = False
    while not done:
        if not np.array_equal(new_state, state):
            t = time.time() - t0
            delta = t - t_1
            t_1 = t

            reward = get_reward(rSpeed, lSpeed, state, delta)
            cum_reward += reward
            result = dict(t=t, delta=delta, state=state, reward=reward, cum_reward=cum_reward,
                          rSpeed=rSpeed, lSpeed=lSpeed)
            state_dict = dict(zip(SENSORS, list(state)))
            result = {**result, **state_dict}
            print(result)
            results = results.append(result, ignore_index=True)
            if save_results:
                write_results(results, dir)
            if np.max(state) > 0.9:
                done = True

            state = new_state
            rSpeed, lSpeed = get_action(model, state)
            robobo.moveWheels(rSpeed, lSpeed)
        else:
            # we check up to 100 times per second if there was a sensor update
            time.sleep(0.01)
        new_state = get_state(robobo)

    # if we crash, we stop
    robobo.stopMotors()



if __name__ == "__main__":

    main()

    
