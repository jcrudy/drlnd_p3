from mappo.agent import MuSigmaLayer, NormalPolicy, Agent, Trainer
from matplotlib import pyplot as plt
import os


if __name__ == '__main__':
    from torch import nn
    from mappo.environment.base import create_tennis_environment
    
    weights_path = os.path.join('checkpoint.pth')
    plot_path = os.path.join('plot.png')
    
    environment = create_tennis_environment()
    
    hidden_size = 400
    state_size = 24
    action_size = 2
    actor_network_1 = nn.Sequential(
                                  nn.Linear(state_size, hidden_size),
                                  nn.ReLU(),
                                  nn.Linear(hidden_size, hidden_size),
                                  nn.ReLU(),
                                  MuSigmaLayer(hidden_size, action_size),
                                  )
    actor_network_2 = nn.Sequential(
                                  nn.Linear(state_size, hidden_size),
                                  nn.ReLU(),
                                  nn.Linear(hidden_size, hidden_size),
                                  nn.ReLU(),
                                  MuSigmaLayer(hidden_size, action_size),
                                  )
    critic_network = nn.Sequential(
                                   nn.Linear(state_size, hidden_size),
                                   nn.ReLU(),
                                   nn.Linear(hidden_size, hidden_size),
                                   nn.ReLU(),
                                   nn.Linear(hidden_size, 1),
                                   )
    
    actor_model_1 = NormalPolicy(actor_network_1)
    actor_model_2 = NormalPolicy(actor_network_2)
    
    agent_1 = Agent(policy_model=actor_model_1)
    agent_2 = Agent(policy_model=actor_model_2)
    trainer = Trainer((agent_1, agent_2), value_model=critic_network)
    
    trainer.train(environment, 1000)
    trainer.to_pickle(weights_path)
    trainer.plot()
    plt.savefig(plot_path)
    plt.show()
    