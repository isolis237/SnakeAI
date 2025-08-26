# main.py
import random
from wall import SnakeWall
from snake import SnakeGame

import numpy as np

from NN.neuralnet import NeuralNetwork
import torch

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

def play_manual():
    g = SnakeGame(fps=10)
    g.init("Snake - Manual")
    g.reset()
    while g.running and not g.done:
        _, _, done, _ = g.step(action=None)
        if done: break
    g.close()

def play_agent():
    g = SnakeGame(grid_w=12, grid_h=12, fps=8)
    H,W, C = 12,12,3
    flat_dim = H*W*C

    g.init("Snake - Agent")
    g.reset()

    model = NeuralNetwork(
        input_dim=flat_dim,
        n_actions=3,                 # L / S / R
        hidden_sizes=(32, 32),  
        activation="relu",
        use_batchnorm=False,
        dropout_p=0.0,
        weight_init="kaiming",      
    ).to(device)
    model.eval()

     # 4) Dummy batch ON THE SAME DEVICE
    x = torch.randn(32, flat_dim, device=device)
    logits = model(x)                 # OK
    probs  = torch.softmax(logits, 1) # OK

    rng = random.Random(0)
    while g.running and not g.done:
        action = rng.choice([0, 1, 2])
        obs, reward, done, other = g.step(action)


        state = obs.reshape(-1).astype("float32")
        x1 = torch.from_numpy(state).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(x1)
            action = int(torch.argmax(logits, dim=1).item())

        _, _, done, _ = g.step(action)
        if done:
            break
        
        #print("head channel sample:\n", obs[:g.grid_w,:g.grid_h,1])
        #print("food channel sample:\n", obs[:g.grid_w,:g.grid_h,2])

        # body, head, food = obs[...,0], obs[...,1], obs[...,2]
        # merged = np.where(head==1, 2,       # code head as 2
        #         np.where(food==1, 3,      # code food as 3
        #         np.where(body==1, 1, 0)))
        # print(merged)

        #for row in g._obs_visual():
        #     print(row)
        # print()
        if done: break
    g.close()

def run_multi():
    # grid_w/h -> discrete positions(steps) exist horizontally/vertically in each game
    # cell -> how many pixels in each grid square / position

    # max ~ SnakeWall(rows=105, cols=210, grid_w=12, grid_h=12, cell= 1, fps=12, seed=0, restart_finished=True)
    # comfortable 
    wall = SnakeWall(rows=4, cols=5, grid_w=12, grid_h=12, cell=22, fps=12, seed=0, restart_finished=True)
    wall.run()

if __name__ == "__main__":
    #play_manual()
    play_agent()
    #run_multi()
