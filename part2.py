# part2 PA2
# authors: Luis Franco and Diego de Leon
# IDs: 80677301(LUIS) and 80754838(DIEGO)

import random

# Defining MDP

# Possible states
states = ["RU8p", "TU10p", "RU10p", "RD10p", "RU8a", "RD8a", "TU10a", "RU10a", "RD10a", "TD10a", "11aFin"]

# Possible actions from each state.
actions = {
    "RU8p": ["P", "R", "S"],
    "TU10p": ["P", "R"],
    "RU10p": ["R", "P", "S"],
    "RD10p": ["R", "P"],
    "RU8a": ["P", "R", "S"],
    "RD8a": ["R", "P"],
    "TU10a": ["P", "R", "S"],
    "RU10a": ["P", "R", "S"],
    "RD10a": ["P", "R", "S"],
    "TD10a": ["P", "R", "S"],
    "11aFin": []
}

# Transitions. syntax: [(probability, next state, reward), ...]
P = {
    "RU8p": {
        "P": [(1.0, "TU10p", 2)],
        "R": [(1.0, "RU10p", 0)],
        "S": [(1.0, "RD10p", -1)]
    },
    "TU10p": {
        "P": [(1.0, "RU10a", 2)],
        "R": [(1.0, "RU8a", 0)]
    },
    "RU10p": {
        "R": [(1.0, "RU8a", 0)],
        "P": [(0.5, "RU8a", 2), (0.5, "RU10a", 2)],
        "S": [(1.0, "RD8a", -1)]
    },
    "RD10p": {
        "R": [(1.0, "RD8a", 0)],
        "P": [(0.5, "RD8a", 2), (0.5, "RD10a", 2)]
    },
    "RU8a": {
        "P": [(1.0, "TU10a", 2)],
        "R": [(1.0, "RU10a", 0)],
        "S": [(1.0, "RD10a", -1)]
    },
    "RD8a": {
        "R": [(1.0, "RD10a", 0)],
        "P": [(1.0, "TD10a", 2)]
    },
    "TU10a": {
        "P": [(1.0, "11aFin", -1)],
        "R": [(1.0, "11aFin", -1)],
        "S": [(1.0, "11aFin", -1)]
    },
    "RU10a": {
        "P": [(1.0, "11aFin", 0)],
        "R": [(1.0, "11aFin", 0)],
        "S": [(1.0, "11aFin", 0)]
    },
    "RD10a": {
        "P": [(1.0, "11aFin", 4)],
        "R": [(1.0, "11aFin", 4)],
        "S": [(1.0, "11aFin", 4)]
    },
    "TD10a": {
        "P": [(1.0, "11aFin", 3)],
        "R": [(1.0, "11aFin", 3)],
        "S": [(1.0, "11aFin", 3)]
    },
    "11aFin": {}
}

# threshold
threshold = 0.001

# learning rate
alpha = 0.1

#discount factor
lambdaDR = 0.99

# epsilon greedy 80% best, 20% random
epsilon = 0.2

start_state = "RU8p"

# init Q(s,a) values to 0
Q = {}
for s in states:
    Q[s] = {}
    for a in actions[s]:
        Q[s][a] = 0.0

# pick an outcome for next state P[s][a] from probablity map
def simulate_transition(state, action, P):
    # Get all possible outcomes
    outcomes = P[state][action]
    # rand between 0-1
    r = random.random()
    cumulative = 0.0

    for prob, next_state, reward in outcomes:
        cumulative += prob
        if r<= cumulative:
            return next_state, reward
        
# Epsilon gredy action choice
def choose_action(state, actions, Q, epsilon):
    available_actions = actions[state]

    #terminal
    if not available_actions:
        return None
    
    # get num between 0 and 1, 20% of time do random action
    if random.random() < epsilon:
        return random.choice(available_actions)
    else: # else get max q from the equation
        return max(Q[state], key=Q[state].get)

def q_learning(states, actions, P, Q, alpha, lambdaDR, epsilon, threshold):
    episode = 0

    while True:
        max_change = 0
        episode += 1
        print(f"\n*** Episode {episode} ***")
        
        #start each episode at start state
        state = start_state
        while state != "11aFin":
            #choose action using epsilon greedy
            action = choose_action(state,actions,Q,epsilon)

        #simulate transition
            next_state, reward = simulate_transition(state, action, P)

        #get max Q value of next state (0 if terminal)
        if actions[next_state]:
            max_q_next = max(Q[next_state].values())
        else:
            max_q_next = 0.0

        #store old Q value for printing and delta
        old_q =  Q[state][action]
        #Q-learning update
        Q[state][action] = old_q + alpha *(reward + lambdaDR * max_q_next - old_q)

        print(f" State: {state}, Action: {action} ")
        print(f" Prev Q: {old_q:.4f} | New Q: {Q[state][action]:.4f}")
        print(f"  Reward: {reward} | Max Q(next): {max_q_next:.4f}")

        #track biggest Q-value change this episode
        max_change = max(max_change, abs(Q[state][action] - old_q))
        state = next_state

    if max_change < threshold
        print(f"\n*** Converged after {episode} episodes ***")
        break
return Q

#Run alg
Q = q_learning(states, actions, P, Q, alpha, lambdaDR, epsilon, threshold)

print(f"*** Final Q-Values ***")
for s in states:
    for a in action[s]:
        print(f" Q({s},{a}) = {Q[s][a]:.4f}")

print("\n=== Optimal Policy ===")
for s in states:
    if actions[s]:
        best = max(Q[s], key=Q[s].get)
        print(f"  {s}: {best}")
    else:
        print(f"  {s}: Terminal")
        
        
