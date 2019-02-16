from unityagents import UnityEnvironment
from functools import partial
from mappo.environment import resources

def create_environment(path, graphics=False):
    return UnityEnvironment(
                            file_name=path, 
                            no_graphics=(not graphics),
                            )

create_tennis_environment = partial(create_environment, path=resources.tennis)
