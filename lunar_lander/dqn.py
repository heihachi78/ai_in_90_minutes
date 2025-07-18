import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import random
from collections import deque
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import pickle

# --- 1. DQN Agent class (optimized) ---
class DQNAgent:
    def __init__(
        self, state_size, action_size, lr=0.001, tau=0.05,
        epsilon_decay=0.995, epsilon_min=0.01, buffer_size=20000
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = lr
        self.tau = tau
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model(hard=True)

    def _build_model(self):
        # Slightly deeper network for improved learning
        model = tf.keras.Sequential([
            layers.Dense(256, input_dim=self.state_size, activation='relu'),
            layers.Dense(256, activation='relu'),
            layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self, hard=False):
        # Polyak averaging (soft) update, or hard copy if specified
        if hard:
            self.target_model.set_weights(self.model.get_weights())
        else:
            model_weights = np.array(self.model.get_weights(), dtype=object)
            target_weights = np.array(self.target_model.get_weights(), dtype=object)
            new_weights = self.tau * model_weights + (1 - self.tau) * target_weights
            self.target_model.set_weights(new_weights)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])

    def replay(self, batch_size=64):
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        # Use np.vstack for efficiency
        states = np.vstack([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.vstack([e[3] for e in batch])
        dones = np.array([e[4] for e in batch]).astype(int)

        # Q-value predictions
        target_q = self.model.predict(states, verbose=0)
        target_next_q = self.target_model.predict(next_states, verbose=0)
        # Bellman update
        targets = rewards + (1 - dones) * 0.99 * np.amax(target_next_q, axis=1)
        target_q[range(batch_size), actions] = targets

        # Train main network
        self.model.fit(states, target_q, epochs=1, verbose=0)

        # Soft update target network
        self.update_target_model(hard=False)

        # Decay epsilon per replay (step)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)

    def save_model(self, filepath):
        """Save the model weights and agent parameters"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model weights
        self.model.save_weights(f"{filepath}_main.weights.h5")
        self.target_model.save_weights(f"{filepath}_target.weights.h5")
        
        # Save agent parameters
        agent_params = {
            'epsilon': self.epsilon,
            'state_size': self.state_size,
            'action_size': self.action_size,
            'learning_rate': self.learning_rate,
            'tau': self.tau,
            'epsilon_decay': self.epsilon_decay,
            'epsilon_min': self.epsilon_min
        }
        
        with open(f"{filepath}_params.pkl", 'wb') as f:
            pickle.dump(agent_params, f)
        
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """Load the model weights and agent parameters"""
        try:
            # Load model weights
            self.model.load_weights(f"{filepath}_main.weights.h5")
            self.target_model.load_weights(f"{filepath}_target.weights.h5")
            
            # Load agent parameters
            with open(f"{filepath}_params.pkl", 'rb') as f:
                agent_params = pickle.load(f)
            
            self.epsilon = agent_params['epsilon']
            self.state_size = agent_params['state_size']
            self.action_size = agent_params['action_size']
            self.learning_rate = agent_params['learning_rate']
            self.tau = agent_params['tau']
            self.epsilon_decay = agent_params['epsilon_decay']
            self.epsilon_min = agent_params['epsilon_min']
            
            print(f"Model loaded from {filepath}")
            print(f"Loaded epsilon: {self.epsilon:.4f}")
            return True
        except FileNotFoundError:
            print(f"No saved model found at {filepath}")
            return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

# --- 2. Training function (optimized) ---
def train_dqn(
    episodes=500,
    replay_updates_per_episode=4,
    batch_size=64,
    tau=0.05,
    save_path="models/dqn_lunar_lander",
    load_model=False
):
    env = gym.make('LunarLander-v3')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(
        state_size, action_size, lr=0.001, tau=tau,
        epsilon_decay=0.995, epsilon_min=0.01, buffer_size=20000
    )
    
    # Try to load pre-trained model if requested
    if load_model:
        if agent.load_model(save_path):
            print("Continuing training from loaded model...")
        else:
            print("No saved model found. Starting fresh training...")
    
    scores = []

    print("Training DQN on LunarLander (optimized)...")
    progress_bar = tqdm(range(episodes), desc="Training DQN", unit="episode")

    for e in progress_bar:
        state, _ = env.reset()
        state = np.reshape(state, [1, state_size])
        total_reward = 0

        for t in range(1000):
            action = agent.act(state)
            next_state, reward, done, truncated, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done or truncated:
                break

        scores.append(total_reward)

        # Multiple replay steps per episode (can increase further for faster learning)
        if len(agent.memory) > batch_size:
            for _ in range(replay_updates_per_episode):
                agent.replay(batch_size)

        # Hard update target model every 20 episodes (redundant, but safe)
        if e % 20 == 0:
            agent.update_target_model(hard=True)

        # Update progress bar
        if len(scores) >= 50:
            avg_score = np.mean(scores[-50:])
            progress_bar.set_description(f"Training DQN - Avg Score: {avg_score:.1f}, ε: {agent.epsilon:.3f}")
        else:
            progress_bar.set_description(f"Training DQN - Score: {total_reward:.1f}, ε: {agent.epsilon:.3f}")

        # Print progress and run test every 50 episodes
        if e % 10 == 0:
            avg_score = np.mean(scores[-50:]) if len(scores) >= 50 else np.mean(scores)
            progress_bar.write(f"Episode {e}, Average Score: {avg_score:.2f}, Epsilon: {agent.epsilon:.3f}")
            if e > 0:
                progress_bar.write(f"--- Testing at Episode {e} ---")
                test_avg = test_dqn(agent, episodes=3, show_details=False)
                progress_bar.write(f"Test Average Score: {test_avg:.2f}")
                progress_bar.write("--- Resuming Training ---")

        # Save model every 25 episodes
        if e % 25 == 0 and e > 0:
            agent.save_model(save_path)
            progress_bar.write(f"Model saved at episode {e}")

    # Save final model
    agent.save_model(save_path)
    print(f"Final model saved at episode {episodes}")

    return agent, scores

# --- 3. Test function ---
def test_dqn(agent, episodes=5, show_details=True):
    env = gym.make('LunarLander-v3', render_mode='human')
    state_size = env.observation_space.shape[0]
    original_epsilon = agent.epsilon
    agent.epsilon = 0  # No exploration

    if show_details:
        print(f"\nTesting trained agent for {episodes} episodes...")

    test_scores = []

    for e in range(episodes):
        state, _ = env.reset()
        state = np.reshape(state, [1, state_size])
        total_reward = 0
        steps = 0

        for _ in range(1000):
            action = agent.act(state)
            next_state, reward, done, truncated, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            state = next_state
            total_reward += reward
            steps += 1
            if done or truncated:
                break

        test_scores.append(total_reward)
        if show_details:
            print(f"Test Episode {e+1}: Score = {total_reward:.2f}, Steps = {steps}")
            if total_reward > 200:
                print("🎉 Successful landing!")
            elif total_reward > 0:
                print("✅ Decent landing")
            else:
                print("💥 Crashed")

    env.close()
    agent.epsilon = original_epsilon
    return np.mean(test_scores)

# --- 4. Test only (load pre-trained model and test) ---
def test_pretrained_model(model_path="models/dqn_lunar_lander", episodes=5):
    """Load a pre-trained model and test it without any training"""
    env = gym.make('LunarLander-v3')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # Create agent
    agent = DQNAgent(state_size, action_size)
    
    # Load pre-trained model
    if not agent.load_model(model_path):
        print("Failed to load model. Please train a model first.")
        return None
    
    # Test the loaded model
    print(f"\nTesting pre-trained model for {episodes} episodes...")
    test_avg = test_dqn(agent, episodes=episodes, show_details=True)
    print(f"\nFinal Average Score: {test_avg:.2f}")
    
    return test_avg

# --- 5. Main execution ---
if __name__ == "__main__":
    """
    Usage options:
    1. Train from scratch: Set MODE = "train_fresh"
    2. Continue training from saved model: Set MODE = "train_continue"  
    3. Test only without training: Set MODE = "test_only"
    
    The model will be saved every 25 episodes and at the end of training.
    Saved files:
    - models/dqn_lunar_lander_main.weights.h5 (main network weights)
    - models/dqn_lunar_lander_target.weights.h5 (target network weights)  
    - models/dqn_lunar_lander_params.pkl (agent parameters including epsilon)
    """
    
    # Choose the mode of operation
    MODE = "train_continue"  # Options: "train_fresh", "train_continue", "test_only"
    MODEL_PATH = "models/dqn_lunar_lander"
    
    if MODE == "test_only":
        # Option 1: Test only without any training
        print("Testing pre-trained model without training...")
        test_pretrained_model(MODEL_PATH, episodes=5)
        
    elif MODE == "train_continue":
        # Option 2: Load pre-trained model and continue training
        print("Loading pre-trained model and continuing training...")
        agent, scores = train_dqn(
            episodes=100,  # Fewer episodes since we're continuing
            replay_updates_per_episode=4, 
            batch_size=64, 
            tau=0.05,
            save_path=MODEL_PATH,
            load_model=True
        )
        
        # Plot training progress
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(scores)
        plt.title('Training Scores (Continued)')
        plt.xlabel('Episode')
        plt.ylabel('Score')
        plt.grid(True)
        plt.subplot(1, 2, 2)
        window = 50
        if len(scores) >= window:
            rolling_mean = np.convolve(scores, np.ones(window)/window, mode='valid')
            plt.plot(rolling_mean)
            plt.title(f'Rolling Mean ({window} episodes)')
            plt.xlabel('Episode')
            plt.ylabel('Score')
            plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Test the trained agent
        print("\n" + "="*50)
        print("Testing the trained agent...")
        print("="*50)
        test_dqn(agent, episodes=5, show_details=True)
        
    elif MODE == "train_fresh":
        # Option 3: Train from scratch
        print("Training from scratch...")
        agent, scores = train_dqn(
            episodes=400, 
            replay_updates_per_episode=4, 
            batch_size=64, 
            tau=0.05,
            save_path=MODEL_PATH,
            load_model=False
        )
        
        # Plot training progress
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(scores)
        plt.title('Training Scores (Fresh)')
        plt.xlabel('Episode')
        plt.ylabel('Score')
        plt.grid(True)
        plt.subplot(1, 2, 2)
        window = 50
        if len(scores) >= window:
            rolling_mean = np.convolve(scores, np.ones(window)/window, mode='valid')
            plt.plot(rolling_mean)
            plt.title(f'Rolling Mean ({window} episodes)')
            plt.xlabel('Episode')
            plt.ylabel('Score')
            plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Test the trained agent
        print("\n" + "="*50)
        print("Testing the trained agent...")
        print("="*50)
        test_dqn(agent, episodes=5, show_details=True)
        
    else:
        print(f"Invalid MODE: {MODE}. Please choose 'train_fresh', 'train_continue', or 'test_only'.")
