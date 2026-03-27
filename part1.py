# part1 PA2
# authors: Luis Franco and Diego de Leon
# IDs: 80677301(LUIS) and 80754838(DIEGO)

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

# Initialize value estimates to 0
V = {s: 0.0 for s in states}

# discount factor
gamma = 0.99

# max change of value in any state for a single iteration threshold
threshold = 0.001

# policy table storing best action for each state
policy = {s: None for s in states}

# Bellman equation, without the max over actions
def value_of_action(state, action, V, P, gamma):
    total = 0.0
    for prob, next_state, reward in P[state][action]:
        total += prob * (reward + gamma * V[next_state])
    return total

# Call value_of_action for every action in a state
# Take the max across all actions
def value_iteration(states, actions, P, V, gamma, threshold):
    iteration = 0

    while True:
        delta = 0.0
        iteration += 1
        print(f"\n*** Iteration: {iteration} ***")

        # Copy so all updates use old V values
        V_new = V.copy()

        # Process each state
        for s in states:
            # Skip terminal state
            if not actions[s]:
                continue

            # Store previous value to compare
            prev_v = V[s]

            # Compute value of each action in the state
            action_values = {}
            for a in actions[s]:
                action_values[a] = value_of_action(s, a, V, P, gamma)

            # Choose action with largest value
            best_action = max(action_values, key=action_values.get)
            best_val = action_values[best_action]

            # Store best action in policy
            policy[s] = best_action

            # Prints
            print(f"\nState: {s}")
            print(f"Previous V: {prev_v:.4f} | New V: {best_val:.4f}")
            for a, v in action_values.items():
                print(f"Estimated value of action {a} in state {s} = {v:.4f}")
            print(f"Best Action: {best_action}")

            delta = max(delta, abs(best_val - prev_v))

            # Store new value in new value table
            V_new[s] = best_val

        # Replace old values with new
        V = V_new

        print(f"\nMaximum change this iteration: {delta:.6f}")

        # Stop if values are no longer changing enough to matter
        if delta < threshold:
            print(f"\n*** Converged after {iteration} iterations ***")
            break

    return V, policy

# call function
V, policy = value_iteration(states, actions, P, V, gamma, threshold)

print("\nFinal Values:")
for s in states:
    print(f"{s}: {V[s]:.4f}")

print("\nOptimal Policy:")
for s in states:
    print(f"{s}: {policy[s]}")