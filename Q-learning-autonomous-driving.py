import numpy as np

# Define environment parameters
lanes = [0, 1, 2]  # Left, Center, Right
actions = [-1, 0, 1]  # Move Left, Stay, Move Right
learning_rate = 0.1
discount_factor = 0.9
epsilon = 0.1
num_episodes = 1000

# Initialize Q-table
q_table = np.zeros((len(lanes), len(actions)))

# Reward function
def get_reward(position):
    return 1 if position == 1 else -1  # Reward staying in the center lane

# Q-Learning training loop
for episode in range(num_episodes):
    position = 1  # Start in the center lane

    while True:
        # Choose action using epsilon-greedy policy
        if np.random.rand() < epsilon:
            action_index = np.random.randint(len(actions))  # Explore
        else:
            action_index = np.argmax(q_table[position])  # Exploit

        action = actions[action_index]
        new_position = np.clip(position + action, 0, len(lanes) - 1)

        # Get reward and update Q-table
        reward = get_reward(new_position)
        next_max = np.max(q_table[new_position])
        q_table[position, action_index] += learning_rate * (reward + discount_factor * next_max - q_table[position, action_index])

        position = new_position
        if position == 1:  # Stop episode if back in center lane
            break

print("Q-table after training for lane following:")
print(q_table)
