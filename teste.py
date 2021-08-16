import gym
from gym.wrappers.atari_preprocessing import AtariPreprocessing
from gym.wrappers.frame_stack import  FrameStack
from time import sleep, time
from helpers import A2C
   
if __name__ == "__main__":
    LOAD = "checkpoint.pth"
    env = gym.make("BreakoutNoFrameskip-v4")
    env = AtariPreprocessing(env)
    env = FrameStack(env,4)

    # os valores zerados não são importantes para a fase de testes.
    agent = A2C((3,210,160),
                env.action_space.n,
                0,
                0,
                0,
                0,
                0,
                load = LOAD)
    
    for i in range(10):
        score = 0
        done = False
        s = env.reset()
        acumulador = 0
        while not done:
            t0 = time()
            acumulador += 1
            action = agent.choose_action(s, test = True)
            s2, reward, done, _ = env.step(action)
            reward += .1
            score += reward
            s = s2
            env.render()
            dt = 1/40 - (time() - t0)
            sleep(dt if dt>0 else 0)
        print(acumulador)
        print(score)
        
            