import numpy as np
import gymnasium as gym
from observation_tools import ObservationTools
from memory import Memory
from driver import Driver


ENV_NAME = 'CarRacing-v3'
DISABLE_FITTING = False # disable fitting for testing the model
SHOW_GAME = False # show the game window
SHOW_RAYS = False # show the rays in different window

DATA_COLLECTION_EPISODES = 25 # episodes without learning, just collectiing data
LAUNCH_CONTROL_STEPS = 85 # steps after launch with full gas
CUT_Y = 84 # cuts the bottom of the image from this row
RAY_ORIGIN_X = 70 # distance rays origin on axis x
RAY_ORIGIN_Y = 48 # distance rays origin on axis y
NORMALISTION_FACTOR = 70. # used for measured ray distance normalisation

MEMORY_SIZE = 10 # number of stored steps in memory
MEMORY_SKIP_FRAMES = 40 # frames that wont be stored when an episode starts
SCORE_LOSS_LIMIT = 1.5 # maximum allowed score loss in episode, to speed up restart on grass, on stopped car
EPISODES = 3 # of episodes to simulate


env = gym.make(ENV_NAME, continuous=False, render_mode='human')
ot = ObservationTools()
mem = Memory(mem_size=MEMORY_SIZE, input_shape=ot.observation.shape, n_actions=env.action_space.n)
dr = Driver(environment=env, launch_control=LAUNCH_CONTROL_STEPS, n_actions=env.action_space.n, data_collection_episodes=DATA_COLLECTION_EPISODES)
best_episode_score = 0
score_history = np.zeros(EPISODES)

for e in range(EPISODES):
    observation, info = env.reset()
    mem.episode_index = 0
    step = 0
    terminated = False
    truncated = False
    score = 0
    max_score = 0

    while not(terminated or truncated) and (score + SCORE_LOSS_LIMIT > max_score):
        action = dr.choose_action(step=step,
                                  episode=e,
                                  state=mem.get_prediction_state())
        observation, reward, terminated, truncated, info = env.step(action)
        score += reward
        if score > max_score:
            max_score = score
        ot.create_grayscale_image(observation=observation, cutY=CUT_Y)
        ot.calculate_rays(ray_origin_x=RAY_ORIGIN_X, ray_origin_y=RAY_ORIGIN_Y, normalisation_factor=NORMALISTION_FACTOR)
        if SHOW_RAYS:
            ot.visualize_grayscale_image()
        if step > MEMORY_SKIP_FRAMES:
            mem.store(state=ot.old_observation, 
                      episode=e,
                      action=action,
                      new_state=ot.observation,
                      terminated=(terminated or truncated or not(score + SCORE_LOSS_LIMIT > max_score)))
        step += 1
    
    if score < 900:
        deleted = mem.delete_episode_data(e)

    if best_episode_score < score:
        best_episode_score = score
    score_history[e] = score

    print(f'e={e} s={step} score={int(score)}', end = ' ')
    print(f'mem={min(mem.counter, mem.mem_size)} validmem={mem.get_valid_training_data_count()}', end=' ')
    print(f'obs_tot=[{ot.total_normalized_distance:.2f}, {ot.total_measured_distance:.2f}] obs_max=[{ot.longest_normalized_distance:.2f}, {ot.longest_measured_distance:.2f}]', end=' ')
    print(f'action_stat={dr.action_stat_percentage}', end=' ')
    print(f'best_e_score={int(best_episode_score)} avg_score={score_history[score_history>0].mean():.2f} score_per_step={(score/step):.2f}')

env.close()
