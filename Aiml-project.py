import numpy as np
import random

class QLearningAgent:
    def __init__(self, n_states, n_actions, learning_rate=0.1, discount_factor=0.95, epsilon=1.0):
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        # Initialize Q-table with zeros
        self.q_table = np.zeros((n_states, n_actions))

    def choose_action(self, state):
        """Explore vs Exploit: Epsilon-Greedy Policy"""
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.n_actions - 1) # Explore
        return np.argmax(self.q_table[state]) # Exploit

    def learn(self, state, action, reward, next_state, done):
        """Update Q-values using the Bellman Equation"""
        old_value = self.q_table[state, action]
        next_max = np.max(self.q_table[next_state])
        
        # Q-learning formula
        new_value = (1 - self.lr) * old_value + self.lr * (reward + self.gamma * next_max)
        self.q_table[state, action] = new_value

        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save_model(self, filename="q_table.npy"):
        np.save(filename, self.q_table)
        print(f"Model saved to {filename}")

# --- Example Usage ---
if __name__ == "__main__":
    # Define a simple 5-state environment
    # States: 0, 1, 2, 3, [4 is Goal]
    agent = QLearningAgent(n_states=5, n_actions=2) # 0: Left, 1: Right

    for episode in range(100):
        state = 0
        total_reward = 0
        done = False
        
        while not done:
            action = agent.choose_action(state)
            
            # Simple Environment Logic
            next_state = state + 1 if action == 1 else max(0, state - 1)
            reward = 10 if next_state == 4 else -1
            done = True if next_state == 4 else False
            
            agent.learn(state, action, reward, next_state, done)
            state = next_state
            
    agent.save_model()
