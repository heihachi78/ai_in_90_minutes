import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time

class FrozenLakeQLearning:
    def __init__(self, environment_name="FrozenLake-v1", is_slippery=False, render=False):
        """
        FrozenLake Q-Learning ágens inicializálása
        
        Args:
            environment_name: A környezet neve
            is_slippery: Ha True, akkor a jég csúszós (sztochasztikus környezet)
        """
        self.is_slippery = is_slippery
        # Környezet létrehozása
        if render:
            self.env = gym.make(environment_name, is_slippery=self.is_slippery, render_mode="human")
        else:
            self.env = gym.make(environment_name, is_slippery=self.is_slippery, render_mode="grayscale")

        # Állapot- és akciótér mérete
        self.state_size = self.env.observation_space.n  # 16 (4x4 rács)
        self.action_size = self.env.action_space.n      # 4 (fel, le, balra, jobbra)
        
        # Q-tábla inicializálása nullákkal
        self.q_table = np.zeros((self.state_size, self.action_size))
        
        # Hiperparaméterek - optimalizált a konzisztens tanulásért
        self.learning_rate = 0.1    # Alpha - alacsonyabb tanulási ráta
        self.discount_factor = 0.99 # Gamma - magasabb leszámítolási tényező
        self.epsilon = 1.0          # Exploráció valószínűsége
        self.epsilon_min = 0.1      # Magasabb minimum exploráció
        self.epsilon_decay = 0.9995 # Lassabb exploráció csökkenés
        
        # Statisztikák követése
        self.rewards_per_episode = []
        self.epsilon_history = []
        
    def choose_action(self, state):
        """
        Epsilon-greedy stratégia: exploráció vs exploitáció
        
        Args:
            state: Jelenlegi állapot
            
        Returns:
            action: Választott akció (0-3)
        """
        if np.random.random() < self.epsilon:
            # Exploráció: véletlenszerű akció
            return self.env.action_space.sample()
        else:
            # Exploitáció: legjobb ismert akció
            return np.argmax(self.q_table[state])
    
    def update_q_table(self, state, action, reward, next_state, done):
        """
        Q-tábla frissítése a Bellman-egyenlet alapján
        
        Q(s,a) = Q(s,a) + α * [r + γ * max(Q(s',a')) - Q(s,a)]
        """
        # Jelenlegi Q-érték
        current_q = self.q_table[state, action]
        
        if done:
            # Ha az epizód véget ért, nincs következő állapot
            target = reward
        else:
            # Bellman-egyenlet: jutalom + leszámítolt jövőbeli maximum Q-érték
            target = reward + self.discount_factor * np.max(self.q_table[next_state])
        
        # Q-érték frissítése
        self.q_table[state, action] = current_q + self.learning_rate * (target - current_q)
    
    def decay_epsilon(self):
        """Exploráció csökkentése az idő múlásával"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def train(self, episodes=1000):
        """
        Ágens tanítása Q-Learning algoritmussal
        
        Args:
            episodes: Tanítási epizódok száma
        """
        print(f"Tanítás kezdése {episodes} epizóddal...")
        print(f"Környezet: {self.state_size} állapot, {self.action_size} akció")
        print(f"Hiperparaméterek: lr={self.learning_rate}, γ={self.discount_factor}, ε_min={self.epsilon_min}")
        print("-" * 50)
        
        for episode in range(episodes):
            # Epizód inicializálása
            state, _ = self.env.reset()
            total_reward = 0
            steps = 0
            done = False
            max_steps = 100  # Végtelen ciklusok elkerülése
            
            while not done and steps < max_steps:
                # Akció választása
                action = self.choose_action(state)
                
                # Lépés a környezetben
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                # Q-tábla frissítése
                self.update_q_table(state, action, reward, next_state, done)
                
                # Állapot frissítése
                state = next_state
                total_reward += reward
                steps += 1
            
            # Exploráció csökkentése
            self.decay_epsilon()
            
            # Statisztikák mentése
            self.rewards_per_episode.append(total_reward)
            self.epsilon_history.append(self.epsilon)
            
            # Haladás kiírása
            if (episode + 1) % 200 == 0:
                avg_reward = np.mean(self.rewards_per_episode[-200:])
                success_rate = np.sum([r > 0 for r in self.rewards_per_episode[-200:]]) / 200 * 100
                print(f"Epizód {episode + 1:4d} | "
                      f"Átlag jutalom: {avg_reward:.3f} | "
                      f"Sikerességi arány: {success_rate:.1f}% | "
                      f"Epsilon: {self.epsilon:.3f}")
        
        self.env.close()
        print("\nTanítás befejezve!")
        
        # Végső statisztikák
        final_success_rate = np.sum([r > 0 for r in self.rewards_per_episode[-100:]]) / 100 * 100
        print(f"Végső sikerességi arány (utolsó 100 epizód): {final_success_rate:.1f}%")
    
    def test(self, episodes=10, render=True, slowdown=0.25):
        """
        Tanított ágens tesztelése
        
        Args:
            episodes: Teszt epizódok száma
            render: Vizualizáció megjelenítése
        """
        if render:
            test_env = gym.make("FrozenLake-v1", is_slippery=self.is_slippery, render_mode="human")
        else:
            test_env = gym.make("FrozenLake-v1", is_slippery=self.is_slippery)
        
        print(f"\nTesztelés {episodes} epizóddal (exploráció kikapcsolva)...")
        successes = 0
        total_rewards = []
        
        for episode in range(episodes):
            state, _ = test_env.reset()
            total_reward = 0
            steps = 0
            done = False
            
            print(f"\nEpizód {episode + 1}:")
            
            while not done and steps < 100:  # Max 100 lépés
                # Mindig a legjobb akciót választjuk (epsilon = 0)
                action = np.argmax(self.q_table[state])
                next_state, reward, terminated, truncated, _ = test_env.step(action)
                done = terminated or truncated
                
                state = next_state
                total_reward += reward
                steps += 1
                
                if render:
                    time.sleep(slowdown)  # Lassabb megjelenítés
            
            total_rewards.append(total_reward)
            if total_reward > 0:
                successes += 1
                print(f"✅ Siker! {steps} lépésben elérte a célt")
            else:
                print(f"❌ Kudarc {steps} lépés után")
        
        test_env.close()
        
        success_rate = successes / episodes * 100
        avg_reward = np.mean(total_rewards)
        
        print(f"\n📊 Teszt eredmények:")
        print(f"Sikerességi arány: {success_rate:.1f}% ({successes}/{episodes})")
        print(f"Átlagos jutalom: {avg_reward:.3f}")
        
        return success_rate, avg_reward
    
    def plot_training_progress(self):
        """Tanítási haladás vizualizálása"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Jutalmak alakulása
        episodes = range(len(self.rewards_per_episode))
        ax1.plot(episodes, self.rewards_per_episode, alpha=0.6, color='blue')
        
        # Simított görbe (50 epizódos átlag)
        if len(self.rewards_per_episode) >= 50:
            smoothed = [np.mean(self.rewards_per_episode[max(0, i-49):i+1]) 
                       for i in range(len(self.rewards_per_episode))]
            ax1.plot(episodes, smoothed, color='red', linewidth=2, label='50-epizód átlag')
            ax1.legend()
        
        ax1.set_title('Jutalmak alakulása')
        ax1.set_xlabel('Epizód')
        ax1.set_ylabel('Jutalom')
        ax1.grid(True, alpha=0.3)
        
        # Epsilon alakulása (exploráció)
        ax2.plot(episodes, self.epsilon_history, color='green')
        ax2.set_title('Exploráció (epsilon) csökkenése')
        ax2.set_xlabel('Epizód')
        ax2.set_ylabel('Epsilon érték')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def print_q_table(self):
        """Q-tábla kiírása szépen formázva"""
        print("\n📋 Végső Q-tábla:")
        print("   Akciók:  Bal    Le   Jobb   Fel")
        print("-" * 40)
        for state in range(self.state_size):
            values = self.q_table[state]
            print(f"Állapot {state:2d}: {values[0]:6.3f} {values[1]:6.3f} {values[2]:6.3f} {values[3]:6.3f}")
    
    def print_policy(self):
        """Optimális stratégia kiírása"""
        action_symbols = ['←', '↓', '→', '↑']
        print("\n🎯 Tanult stratégia (optimális akciók):")
        
        policy = np.argmax(self.q_table, axis=1)
        
        # 4x4 rácsként megjelenítés
        for row in range(4):
            row_str = ""
            for col in range(4):
                state = row * 4 + col
                if state in [5, 7, 11, 12]:  # Lyukak (általában)
                    row_str += " H "
                elif state == 15:  # Cél
                    row_str += " G "
                elif state == 0:   # Start
                    row_str += " S "
                else:
                    row_str += f" {action_symbols[policy[state]]} "
            print(row_str)

# Használat példa
if __name__ == "__main__":
    print("🎮 FrozenLake Q-Learning megoldó")
    print("=" * 50)
    print("Cél: Eljutni az S-től a G-ig a lyukak (H) elkerülésével")
    print("Akciók: 0=Bal, 1=Le, 2=Jobb, 3=Fel")
    print()
    
    # Konzisztencia teszt - több futtatás összehasonlítása
    print("🔬 KONZISZTENCIA TESZT:")
    print("Több független tanítás összehasonlítása...")
    print("-" * 50)
    
    success_rates = []
    
    for run in range(3):
        print(f"\n--- {run + 1}. futtatás ---")
        
        # Ágens létrehozása minden futtatáshoz új
        agent = FrozenLakeQLearning(is_slippery=False, render=False)
        
        # Tanítás
        agent.train(episodes=6000)  # Több epizód a jobb tanuláshoz
        
        # Tesztelés
        success_rate, avg_reward = agent.test(episodes=20, render=True, slowdown=0.0)
        success_rates.append(success_rate)
        
        print(f"Sikerességi arány: {success_rate:.1f}%")
        
        # Csak az első futtatásnál jelenítjük meg a részletes eredményeket
        if run == 0:
            agent.plot_training_progress()  # Grafikon megjelenítése
            agent.print_policy()
            agent.print_q_table()
    
    # Összesített eredmények
    print(f"\n📊 KONZISZTENCIA EREDMÉNYEK:")
    print(f"Sikerességi arányok: {[f'{rate:.1f}%' for rate in success_rates]}")
    print(f"Átlagos sikerességi arány: {np.mean(success_rates):.1f}%")
    print(f"Szórás: {np.std(success_rates):.1f}%")
    print(f"Min-Max: {np.min(success_rates):.1f}% - {np.max(success_rates):.1f}%")
    
    if np.std(success_rates) < 15:
        print("\n✅ KONZISZTENS tanulás - az optimalizált hiperparaméterek működnek!")
    else:
        print("\n⚠️  VÁLTOZÓ eredmények - a sparse reward még mindig problémát okoz.")
        
    print(f"\n💡 MAGYARÁZAT:")
    print(f"A módosított hiperparaméterek (lr={agent.learning_rate}, ε_min={agent.epsilon_min}, ε_decay={agent.epsilon_decay})")
    print(f"segítenek a konzisztensebb tanulásban, de a FrozenLake sparse reward struktura")
    print(f"miatt még mindig lehet némi variancia az eredményekben.")