import gym
import torch
from gym.wrappers.atari_preprocessing import AtariPreprocessing
from gym.wrappers.frame_stack import  FrameStack
from helpers import A2C

def main():
    PATH = "./"
    LOAD = PATH + "checkpoint.pth"
    HISTORY = PATH + "history.pickle"
    GAMMA = .99
    LEARNING_RATE = .001
    ENTROPY_COEFICIENT = .0001
    BATCH = 128
    NUM_ENVS = 16
    MAX = 100000

    # cria environment
    env = gym.vector.make("BreakoutNoFrameskip-v4",
                        num_envs = NUM_ENVS,
                        asynchronous = False,
                        wrappers = [AtariPreprocessing, lambda env: FrameStack(env, 4)])

    # cria agente
    agent = A2C((3,210,160),
                env.single_action_space.n,
                LEARNING_RATE,
                GAMMA,
                ENTROPY_COEFICIENT,
                BATCH,
                NUM_ENVS,
                load = LOAD)

    # main loop
    counter = 0
    medias = 0
    r_history = 0
    episode = 0
    states = env.reset()
    while counter < MAX:
        counter += 1
        actions = agent.choose_action(states)
        next_states, rewards, dones, _ = env.step(actions)
        rewards += 0.1
        agent.memory.append((states, actions, rewards, next_states, dones))
        agent.train()
        states = next_states   

        r_history += rewards[0]
        if dones[0]:
            medias += r_history
            episode += 1
            c = counter/MAX
            print(f"(ep: {episode:03d} out of {int(episode/c)} | {int(r_history):03d} | {(c*100):.2f}%) mean {int(medias/episode)}")
            r_history = 0
        if episode%25 == 0:
            torch.save(agent.A2C.state_dict(), LOAD)

if __name__ == "__main__":
    main()
    