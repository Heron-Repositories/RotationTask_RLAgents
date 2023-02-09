import os
import sys
from os import path

current_dir = path.dirname(path.abspath(__file__))
while path.split(current_dir)[-1] != r'Heron':
    current_dir = path.dirname(current_dir)
sys.path.insert(0, path.dirname(current_dir))

from Heron import general_utils as gu
Exec = os.path.abspath(__file__)
# </editor-fold>

# <editor-fold desc="The following code is called from the GUI process as part of the generation of the node.
# It is meant to create node specific elements (not part of a generic node).
# This is where a new node's individual elements should be defined">
"""
Properties of the generated Node
"""
BaseName = 'Qlearner For Rat In Virtual Rotation'   # The base name can have spaces.
NodeAttributeNames = ['Parameters', 'Start', 'Reward and Observations', 'Action']

NodeAttributeType = ['Static', 'Input', 'Input Dict', 'Output']

ParameterNames = ['Mode', 'Q file to save or use', 'alpha', 'gamma', 'starting epsilon', 'epsilon decay',
                  'finishing epsilon']
ParameterTypes = ['list', 'str', 'float', 'float', 'float', 'float', 'float']

ParametersDefaultValues = [['Learning', 'Testing'], '', 0.1, 0.9, 0.9, 0.0001, 0.1]

# The following line needs to exist with the correct name for the xxx_worker.py script
WorkerDefaultExecutable = os.path.join(os.path.dirname(Exec), 'qlearner_for_rat_in_virtual_rotation_worker.py')
# </editor-fold>


# <editor-fold desc="The following code is called as its own process when the editor starts the graph.
#  You can refactor the name of the xxx_com variable but do not change anything else">
if __name__ == "__main__":
    qlearner_for_rat_in_virtual_rotation_com = gu.start_the_transform_communications_process(NodeAttributeType, NodeAttributeNames)
    gu.register_exit_signals(qlearner_for_rat_in_virtual_rotation_com.on_kill)
    qlearner_for_rat_in_virtual_rotation_com.start_ioloop()

# </editor-fold>