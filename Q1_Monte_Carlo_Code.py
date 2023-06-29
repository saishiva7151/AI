def simulate_game_show_episode(mdp):
    state = mdp.start_state
    episode = []
    while state != mdp.end_state:
        action = np.random.choice([0, 1])
        reward = mdp.get_rewards(state, action, state + action)
        episode.append((state, action, reward))
        state += action
    episode.append((state, None, 0))
    return episode

def monte_carlo_evaluation(mdp, num_episodes):
    # Initialize the value function and count for each state
    V = np.zeros(mdp.num_states)
    N = np.zeros(mdp.num_states)

    # Simulate the specified number of episodes
    for i in range(num_episodes):
        episode = simulate_game_show_episode(mdp)
        G = 0
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            G = mdp.gamma * G + reward
            if action is not None:
                N[state] += 1
                alpha = 1 / N[state]
                V[state] += alpha * (G - V[state])

    return V
  
# Define the rewards and probabilities for the game show
rewards = [100,500,1000,5000,10000,50000,100000,500000,1000000,5000000]
probabilities = [0.99,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]

# Create the game show MDP
mdp = GameShowMDP(rewards, probabilities)

# Run Monte Carlo evaluation for 100 episodes
V = monte_carlo_evaluation(mdp, 100)

# Print the value function for each state
for s in range(mdp.num_states):
    print(f"V({s}) = {V[s]}")
