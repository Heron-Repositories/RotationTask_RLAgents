
import sys
from os import path

current_dir = path.dirname(path.abspath(__file__))
node_dir = current_dir
while path.split(current_dir)[-1] != r'Heron':
    current_dir = path.dirname(current_dir)
sys.path.insert(0, path.dirname(current_dir))

import numpy as np
import copy
from Heron.communication.socket_for_serialization import Socket
from Heron import general_utils as gu, constants as ct
import commands_to_unity as cu
from Heron.gui.visualisation_dpg import VisualisationDPG

visualisation_dpg: VisualisationDPG
game: str
observation_type: str
initialised = False
visualisation_on = False


def get_parameters(_worker_object):
    global game
    global observation_type
    global visualisation_dpg
    global visualisation_on

    try:
        parameters = _worker_object.parameters
        visualisation_on = parameters[0]
        game = parameters[1]
        observation_type = parameters[2]
    except:
        return False

    visualisation_dpg = VisualisationDPG(_node_name=_worker_object.node_name, _node_index=_worker_object.node_index,
                                         _visualisation_type='Single Pane Plot', _buffer=100,
                                         _x_axis_label='Latest Actions',
                                         _y_axis_base_label='Cumulative Reward',
                                         _base_plot_title='Cumulative Reward over Actions')

    _worker_object.savenodestate_create_parameters_df(visualisation=visualisation_on, game=game, observation_type=observation_type)
    return True


def initialise(_worker_object):
    global initialised

    if not get_parameters(_worker_object):
        return False

    return True


def work_function(data, parameters, savenodestate_update_substate_df):
    global visualisation_on
    global observation_type

    visualisation_dpg.visualisation_on = parameters[0]

    topic = data[0]

    message = data[1:]  # data[0] is the topic
    message = Socket.reconstruct_data_from_bytes_message(message)


    # Generate the result
    result = [np.array([ct.IGNORE])]

    return result


# The on_end_of_life function must exist even if it is just a pass
def on_end_of_life():
    if initialised:
        cu.kill_unity()


if __name__ == "__main__":
    worker_object = gu.start_the_transform_worker_process(work_function=work_function,
                                                          end_of_life_function=on_end_of_life,
                                                          initialisation_function=initialise)
    worker_object.start_ioloop()
