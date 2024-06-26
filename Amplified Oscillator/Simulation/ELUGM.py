
import sys
from os import environ
from os import getcwd
import string

sys.path.append(environ["PYTHON_MODULE_PATH"])


import CompuCellSetup


sim,simthread = CompuCellSetup.getCoreSimulationObjects()
        
# add extra attributes here
        
CompuCellSetup.initializeSimulationObjects(sim,simthread)
# Definitions of additional Python-managed fields go here
        
#Add Python steppables here
steppableRegistry=CompuCellSetup.getSteppableRegistry()
        
from ELUGMSteppables import ELUGMSteppable
steppableInstance=ELUGMSteppable(sim,_frequency=1)
steppableRegistry.registerSteppable(steppableInstance)

# from ELUGMSteppables import MitosisSteppable
# MitosisSteppableInstance=MitosisSteppable(sim,_frequency=1)
# steppableRegistry.registerSteppable(MitosisSteppableInstance)

from ELUGMSteppables import ExtraFields
extraFields=ExtraFields(_simulator=sim,_frequency=1)
steppableRegistry.registerSteppable(extraFields)
        
CompuCellSetup.mainLoop(sim,simthread,steppableRegistry)