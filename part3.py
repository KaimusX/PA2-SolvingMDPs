# part3 PA2
# authors: Luis Franco and Diego de Leon
# IDs: 80677301(LUIS) and 80754838(DIEGO)

import random

# Defining MDP TODO

# Possible states 
# Terminal state
terminal_state = "TERMINAL"

# Initial state:
# (c1, k1, c2, k2, c3, k3)
# c1 = binary, state 1 compromised?
# k1 = binary, state 1 ME vulnerability known?

start_state = (0, 0, 0, 0, 0, 0) # no known no compromised

# Generate state space
states = []
for c1 in [0, 1]:
    for k1 in [0, 1]:
        for c2 in [0, 1]:
            for k2 in [0, 1]:
                for c3 in [0, 1]:
                    for k3 in [0, 1]:
                        states.append((c1, k1, c2, k2, c3, k3))

states.append(terminal_state)



# Possible actions from each state.
# for each host we only allow sc bf or me if host is not compromised ot prevent wasting actions

actions = {}

for state in states:
    # Term state
    if state == terminal_state:
        actions[state] = []
        continue
    c1, k1, c2, k2, c3, k3 = state
    
    available = []

    available.append("END")

    #host 1 actions
    if c1 == 0:
        available.extend(["SC1", "BF1", "ME1"])

    # host 2 actions
    if c2 == 0:
        available.extend(["SC2", "BF2", "ME2"])

    # host 3 actions
    if c3 == 0:
        available.extend(["SC3", "BF3", "ME3"])

    actions[state] = available

# Transitions. syntax: [(probability, next state, reward), ...] TODO
#host info
hosts = {
    1: {"value": 20, "penalty": -10, "p_BF": 0.90, "p_ME": 0.70},
    2: {"value": 30, "penalty": -15, "p_BF": 0.25, "p_ME": 0.50},
    3: {"value": 45, "penalty": -25, "p_BF": 0.10, "p_ME": 0.30}
}
#Action Costs
cost = {"SC": 5, "BF": 10, "ME": 5}

#Build Transitions
P = {}
for state in states:
    #if terminal state, no transition
    if state == terminal_state:
        P[state] = {}
        continue

    c1, k1, c2, k2, c3, k3 = state
    #group each host's (compromised,known) status together for easy access
    host_vals = [(c1, k1), (c2, k2), (c3, k3)]
    P[state] = {}

    # END action: goes to terminal state w/no reward
    P[state]["END"] = [(1.0, terminal_state, 0)]

    #get each host's status (compromised,known) and information(value, penalty, and attack prob.)
    for i in range(1, 4):
        c, k = host_vals[i-1]
        h = hosts[i]
        if c == 1:
            continue
            
        #returns a new state with host i's updated c and k
        def next_s(state, i, new_c, new_k):
            s = list(state)
            s[(i-1)*2] = new_c
            s[(i-1)*2 + 1] = new_k
            return tuple(s)

        # SC action: sets k = 1 and costs 5
        P[state][f"SC{i}"] = [
            (1.0, next_s(state, i, 0, 1), -cost["SC"])
        ]

        # BF action: succeeds with p_BF, costs 10
        #success: host is compromised, get value - cost
        #failure: terminal
        p_bf = h["p_BF"]
        P[state][f"BF{i}"] = [
            (p_bf,   next_s(state, i, 1, k), h["value"] - cost["BF"]),
            (1-p_bf, terminal_state, h["penalty"] - cost["BF"]) #
        ]
        # ME action: always available with probablity p_ME.
        # Scan does not enable ME, but instead provides information (k bit) which
        # Could influence learned policy
        p_me = h["p_ME"]
        P[state][f"ME{i}"] = [
            (p_me,   next_s(state, i, 1, k), h["value"] - cost["ME"]),
            (1-p_me, terminal_state, h["penalty"] - cost["ME"])
        ]
# threshold
threshold = 0.001

# learning rate
alpha = 0.1

#discount factor
lambdaDR = 0.99

# epsilon greedy 80% best, 20% random
epsilon = 0.2

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
    
    #fallback for fps
    return outcomes[-1][1], outcomes[-1][2]
        
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

# Visit counter to prevent premature convergence
visits = {}
for s in states:
    visits[s] = {}
    for a in actions[s]:
        visits[s][a] = 0

# Check for minimum visits
def enough_visits(visits, actions, min_visits=3):
    for s in actions:
        for a in actions[s]:
            if visits[s][a] < min_visits:
                return False
    return True

def q_learning(states, actions, P, Q, alpha, lambdaDR, epsilon, threshold):
    episode = 0

    # to prevent premature convergence
    stable_count = 0
    required_stable_episodes = 3
    min_episodes = 500
    
    while True:
        max_change = 0
        episode += 1
        print(f"\n*** Episode {episode} ***")
        #start each episode at start state
        state = start_state
        #choose action using epsilon greedy
        
        while state != terminal_state:
            action = choose_action(state, actions, Q, epsilon)
            #simulate transition
            next_state, reward = simulate_transition(state, action, P)

            #get max Q-value of next state(o if terminal)
            if actions[next_state]:
                max_q_next = max(Q[next_state].values())
            else:
                max_q_next = 0.0
                
            #store old Q-value for printing and delta
            old_q = Q[state][action]
            #Q-learning update
            Q[state][action] = old_q + alpha * (reward + lambdaDR * max_q_next - old_q)
            visits[state][action] += 1 #increment visit

            print(f"  State: {state}, Action: {action}")
            print(f"  Prev Q: {old_q:.4f} | New Q: {Q[state][action]:.4f}")
            print(f"  Reward: {reward} | Max Q(next): {max_q_next:.4f}")

            #track biggest Q-value change this episode
            max_change = max(max_change, abs(Q[state][action] - old_q))
            state = next_state

        # if small updates, is stable
        if max_change < threshold:
            stable_count += 1
        else:
            stable_count = 0

        #only check convergence after min_episodes
        if episode >= min_episodes and stable_count >= required_stable_episodes:
            print(f"\n*** Converged after {episode} episodes ***")
            break

    return Q

# Run it
Q = q_learning(states, actions, P, Q, alpha, lambdaDR, epsilon, threshold)

print("\n*** Final Q-Values ***")
for s in states:
    for a in actions[s]:
        print(f"  Q({s},{a}) = {Q[s][a]:.4f}")

print("\n*** Optimal Policy ***")
for s in states:
    if actions[s]:
        best = max(Q[s], key=Q[s].get)
        print(f"  {s}: {best}")
    else:
        print(f"  {s}: Terminal")
        
