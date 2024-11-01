import numpy as np

# Define environment parameters
states = ["basic_inquiry", "technical_issue", "complaint"]  # Types of customer queries
actions = ["provide_info", "escalate", "offer_apology"]  # Actions to respond to queries
learning_rate = 0.1
discount_factor = 0.9
epsilon = 0.1
num_episodes = 1000

# Initialize Q-table
q_table = np.zeros((len(states), len(actions)))

# Reward function
def get_reward(state, action):
    if state == "basic_inquiry" and action == "provide_info":
        return 10
    elif state == "technical_issue" and action == "escalate":
        return 8
    elif state == "complaint" and action == "offer_apology":
        return 5
    else:
        return -5  # Penalty for incorrect responses

# Q-Learning training loop
for episode in range(num_episodes):
    state_index = np.random.randint(len(states))  # Random initial query type

    while True:
        # Choose action using epsilon-greedy policy
        if np.random.rand() < epsilon:
            action_index = np.random.randint(len(actions))  # Explore
        else:
            action_index = np.argmax(q_table[state_index])  # Exploit

        action = actions[action_index]
        reward = get_reward(states[state_index], action)
        
        # Simulate new state transition (randomized for simplicity)
        new_state_index = np.random.randint(len(states))
        
        # Q-Learning update
        next_max = np.max(q_table[new_state_index])
        q_table[state_index, action_index] += learning_rate * (reward + discount_factor * next_max - q_table[state_index, action_index])

        state_index = new_state_index
        if reward > 0:  # End episode if the response is correct
            break

print("Q-table after training for customer support:")
print(q_table)
