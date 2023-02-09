
import sys
from os import path

current_dir = path.dirname(path.abspath(__file__))
node_dir = current_dir
while path.split(current_dir)[-1] != r'Heron':
    current_dir = path.dirname(current_dir)
sys.path.insert(0, path.dirname(current_dir))

from q_learning_agent import QLearner
import numpy as np
from Heron.communication.socket_for_serialization import Socket
from Heron import general_utils as gu, constants as ct

game: str
agent: QLearner
running = False
mode: bool
q_file: str
action_set = ['Action=Move:Forwards', 'Action=Move:Back', 'Action=Rotate:CW', 'Action=Rotate:CCW', 'Action=Nothing:Nothing']
previous_action = None
previous_observation = None


def get_parameters(_worker_object):
    global agent
    global mode
    global q_file
    global action_set

    try:
        parameters = _worker_object.parameters
        mode = parameters[0]
        q_file = parameters[1]
        alpha = parameters[2]
        gamma = parameters[3]
        start_epsilon = parameters[4]
        epsilon_decay = parameters[5]
        end_epsilon = parameters[6]
    except:
        return False

    _worker_object.savenodestate_create_parameters_df(mode=mode, q_file=q_file, alpha=alpha, gamma=gamma,
                                                      start_epsilon=start_epsilon, epsilon_decay=epsilon_decay,
                                                      end_epsilon=end_epsilon)

    agent = QLearner(alpha=alpha, gamma=gamma, starting_epsilon=start_epsilon, actions_set=action_set,
                     epsilon_decay=epsilon_decay, minimum_epsilon=end_epsilon, q_table=None)
    return True


def initialise(_worker_object):
    if not get_parameters(_worker_object):
        return False

    return True


def work_function(data, parameters, savenodestate_update_substate_df):
    global running
    global agent
    global previous_observation
    global previous_action
    global action_set

    topic = data[0].decode('utf-8')

    message = data[1:]  # data[0] is the topic
    message = Socket.reconstruct_data_from_bytes_message(message)

    observation_in = False
    if 'Start' in topic:
        running = True
    if 'Observation' in topic:
        observation_in = True

    result = [np.array([ct.IGNORE])]
    if running:
        if observation_in:
            try:
                reward = int(message['Reward'])
                observation_state_index = message['State Index']

                # First learn with previous observation, previous action and current observation
                if previous_observation is not None and previous_action is not None:
                    agent.learn(previous_observation, previous_action, reward, observation_state_index)

                # Then calculate action from current observation
                action = agent.act(observation_state_index)
                # and send it
                result = [np.array([action])]

                # Finally assign previous observation and previous action
                previous_action = action
                previous_observation = observation_state_index
            except Exception as e:
                print(e)
        else:
            result = [np.array([np.random.choice(action_set)])]
        observation_in = False
    return result


# The on_end_of_life function must exist even if it is just a pass
def on_end_of_life():
    global q_file
    global agent

    if running:
        agent.q_table.savez_to_numpy(q_file)


if __name__ == "__main__":
    worker_object = gu.start_the_transform_worker_process(work_function=work_function,
                                                          end_of_life_function=on_end_of_life,
                                                          initialisation_function=initialise)
    worker_object.start_ioloop()
