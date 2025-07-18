import os
import gymnasium as gym
import numpy as np
import time
import tensorflow as tf
import argparse
from  tqdm import tqdm
from memory_vec import Memory
from model_vec import BaseModel, ModelActorCritic, DuelingDeepQNetwork
from agent_vec import Agent
from operator import add



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')



def disable_gpu(disable : bool = False) -> str:
    physical_gpu_devices = tf.config.list_physical_devices('GPU')
    print(f'physical_gpu_devices {physical_gpu_devices}')
    physical_cpu_devices = tf.config.list_physical_devices('CPU')
    print(f'physical_cpu_devices {physical_cpu_devices}')
    device = 'VECGPU'
    if disable:
        try:
            # Disable all GPUS
            tf.config.set_visible_devices([], 'GPU')
            visible_devices = tf.config.get_visible_devices()
            print(f'USING CPU {visible_devices}')
            for device in visible_devices:
                assert device.device_type != 'GPU'
            device = 'VECCPU'
        except:
            print('cannot disable gpu')
    else:
        print(f'USING GPU {physical_gpu_devices}')
    return device


def test(render : bool = False) -> int:
    t_start_eval = time.perf_counter()

    if render:
        env = gym.make(ENV_NAME, render_mode="human")
    else:
        env = gym.make(ENV_NAME)

    eval_net : BaseModel = DuelingDeepQNetwork(n_input=env.observation_space.shape[0],
                                               n_neurons=NEURONS,
                                               n_output=env.action_space.n,
                                               learning_rate=LEARNING_RATE,
                                               gamma=GAMMA,
                                               save_file=ENV_NAME,
                                               verbose=VERBOSE,
                                               device=device)
    
    eval_net.load()
    
    observation, info = env.reset()

    score = 0
    score_hist = []

    with tqdm(total=TEST_LENGTH, colour='green') as pbar:
        for _ in range(TEST_LENGTH):
            action = np.argmax(eval_net.predict(states=np.reshape(observation, (1, env.observation_space.shape[0]))))
            observation, reward, terminated, truncated, info = env.step(action)
            score += reward
            pbar.update(1)

            if terminated or truncated:
                score_hist.append(int(score))
                score = 0
                pbar.set_description(f'eval ({len(score_hist)} completed)')
                observation, info = env.reset()

    print(f'test scores {score_hist} (mean={int(np.mean(score_hist))})')

    env.close()

    t_end_eval = time.perf_counter()

    print(f'took {(t_end_eval - t_start_eval):.2f} secs to test')

    return int(np.mean(score_hist))


def learn(device : str) -> None:
    t_total_start = time.perf_counter()

    envs : gym.vector.VectorEnv = gym.make_vec(ENV_NAME, num_envs=PROCESSES, vectorization_mode="async" if ASYNC else "sync")

    mem : Memory = Memory(processes=PROCESSES,
                          max_memory=MAX_MEMORY,
                          state_shape=envs.single_observation_space.shape,
                          n_actions=envs.single_action_space.n)

    #nn : BaseModel = ModelActorCritic(n_input=envs.single_observation_space.shape[0],
    nn : BaseModel = DuelingDeepQNetwork(n_input=envs.single_observation_space.shape[0],
                                         n_output=envs.single_action_space.n,
                                         n_neurons=NEURONS,
                                         learning_rate=LEARNING_RATE,
                                         gamma=GAMMA,
                                         save_file=ENV_NAME,
                                         verbose=VERBOSE,
                                         device=device)
    nn.load()
    
    agent : Agent= Agent(envs=envs,
                         memory=mem,
                         model=nn,
                         n_samples=N_LEARN_SAMPLES,
                         eps=EPS,
                         eps_dec=EPS_DEC,
                         eps_min=EPS_MIN)
    
    for i in range(ITERS):

        print(f'\nITER {i}/{ITERS}')

        t_start = time.perf_counter()

        observations, infos = envs.reset()

        with tqdm(total=SIM_LENGTH, colour='cyan') as pbar:
            for e in range(SIM_LENGTH):
                old_observations = observations.copy()
                actions = agent.act(states=old_observations, learning=LEARNING)
                observations, rewards, terminations, truncations, infos = envs.step(actions)
                observations_ = observations.copy()
                dones_ = np.logical_or(terminations, truncations)

                try:
                    # Handle final observations if available (older gymnasium versions)
                    if 'final_observation' in infos and np.any([info is not None for info in infos.get('final_observation', [])]):
                        fo = [infos['final_observation'][i] if infos['final_observation'][i] is not None else observations[i] for i in range(PROCESSES)]
                        fo = np.array(fo)
                        observations_ = fo.copy()
                except Exception:
                    # For newer gymnasium versions, the final observation handling is automatic
                    pass

                mem.store(old_observations, actions, rewards, observations_, dones_)
                nn.learn(mem=mem, n_samples=N_LEARN_SAMPLES, offset=PROCESSES)
                if not(e % WEIGHTS_REFRESH_INTERVAL):
                    nn.refresh_weights()

                pbar.set_description(f'train (eps={agent.eps:.2f})')
                pbar.update(1)

        nn.refresh_weights()
        agent.model.save()

        t_end = time.perf_counter()

        print(f'collected memory entries: {mem.count}')
        print(f'took {(t_end - t_start):.2f} secs to train')

        train_eval(nn)

        t_total_end = time.perf_counter()
        t_total_dur = t_total_end - t_total_start
        h = int(t_total_dur // 3600)
        m = int(t_total_dur // 60 - h * 60)
        s = int(t_total_dur - h * 3600 - m * 60)

        print(f'total run time: {h}h {m}m {s}s')

    envs.close()


def train_eval(model : BaseModel) -> None:

    t_start = time.perf_counter()

    envs : gym.vector.VectorEnv = gym.make_vec(ENV_NAME, num_envs=PROCESSES, vectorization_mode="async" if ASYNC else "sync")

    observations, infos = envs.reset()
    scores = [0 for _ in range(PROCESSES)]
    score_hist = []

    with tqdm(total=EVAL_LENGTH, colour='green') as pbar:
        for _ in range(EVAL_LENGTH):
            old_observations = observations.copy()
            actions = np.argmax(model.predict(states=old_observations), axis=1)
            observations, rewards, terminations, truncations, infos = envs.step(actions)
            dones_ = np.logical_or(terminations, truncations)
            scores = list(map(add, scores, rewards))
            pbar.update(1)

            if np.any(dones_):
                for i, d in enumerate(dones_):
                    if d:
                        score_hist.append(int(scores[i]))
                        scores[i] = 0
                pbar.set_description(f'eval ({len(score_hist)} completed)')

    print(f'eval scores {score_hist} (mean={int(np.mean(score_hist))})')

    t_end = time.perf_counter()

    print(f'took {(t_end - t_start):.2f} secs to eval')



if __name__ == '__main__':

    LEARNING = False
    DISABLE_GPU = False
    ASYNC = True
    PROCESSES = 10

    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--enablegpu', action='store_true', help="enable GPU", default=False)
    parser.add_argument('-l', '--learn', action='store_true', help="learn the environment", default=False)
    parser.add_argument('-s', '--synchronous', action='store_true', help="disable async environments", default=False)
    parser.add_argument('-p', '--processcount', type=int, help="paralell process count", default=12)
    args = parser.parse_args()
    if args.enablegpu:
        DISABLE_GPU = False
    if args.learn:
        LEARNING = True
    if args.synchronous:
        ASYNC = False
    if args.processcount:
        PROCESSES = args.processcount

    ENV_NAME = 'LunarLander-v3' 
    #ENV_NAME = 'CartPole-v1'
    #ENV_NAME = 'MountainCar-v0'
    ASYNC = True
    ITERS = 100
    MAX_MEMORY = 1_000_000
    SIM_LENGTH = 10_000
    if DISABLE_GPU:
        NEURONS = [512, 256]
    else:
        NEURONS = [512, 256]
    LEARNING_RATE = 3e-4
    GAMMA = 0.99
    VERBOSE = 0
    N_LEARN_SAMPLES = 64
    EPS = 1
    EPS_DEC = 1-10e-4
    EPS_MIN = 0.1
    WEIGHTS_REFRESH_INTERVAL = 100
    EVAL_LENGTH = 2_000
    TEST_LENGTH = 10_000

    device = disable_gpu(DISABLE_GPU)

    if LEARNING:
        learn(device=device)

    if not LEARNING:

        _ = test(render=not LEARNING)
