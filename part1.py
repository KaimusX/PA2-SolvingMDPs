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

# Transitions. syntax: [(probablitty, next state, reward)...]
P = {
    #level 0 states
    "RU8p": {
        "P": [(1.0, "TU10p", 2)],
        "R": [(1.0, "RU10p", 0)],
        "S": [(1.0, "RD10p", -1)]
    },
    # level 1 states
    "TU10p": {
        "P": [(1.0, "RU10a", 2)],
        "R": [(1.0, "RU8a", 0)]
    },
    "RU10p": {
        "R": [(1.0, "RU8a", 0)],
        "P": [(0.5, "RU8a",2), (0.5, "RU10a", 2)],
        "S": [(1.0, "RD8a", -1)]
    },
    "RD10p": {
        "R": [(1.0, "RD8a", 0)],
        "P": [(0.5, "RD8a", 2), (0.5, "RD10a", 2)]
    },
    # level 2 states
    "RU8a": {
        "P":[(1.0,"TU10a",2)],
        "R":[(1.0,"RU10a",0)],
        "S":[(1.0,"RD10a",-1)]
    },
    "RD8a":{
        "R":[(1.0,"RD10a",0)],
        "P":[(1.0,"TD10a",2)]
    },
    # level 3 states
    "TU10a":{
        "P": [(1.0,"11aFin",-1)],
        "R": [(1.0,"11aFin",-1)],
        "S": [(1.0,"11aFin",-1)]
    },
    "RU10a":{
        "P":[(1.0,"11aFin",0)],
        "R":[(1.0,"11aFin",0)],
        "S":[(1.0,"11aFin",0)]
    },  
    "RD10a":{
        "P":[(1.0,"11aFin",4)],
        "R":[(1.0,"11aFin",4)],
        "S":[(1.0,"11aFin",4)]

    },
    "TD10a":{
        "P":[(1.0,"11aFin",3)],
        "R":[(1.0,"11aFin",3)],
        "S":[(1.0,"11aFin",3)]
    },
    # level 4 state, terminal
    "11aFin": {}
}

# Initialize value estimates to zilch 0
V = {s: 0.0 for s in states}

# discount factor
gamma = 0.99

# max change of value in any state for a single iteration threshold
threshold = 0.001

# policy table storing best action for each state
policy = {s: None for s in states}

# Bellman equation, NOTE: does not have max over actions yet.
def value_of_action(state, action, V, P, gamma):
    total = 0.0
    for prob, next_state, reward in P[state][action]:
        total += prob * (reward + gamma * V[next_state])
    return total

#Call value_of_action for every action in a state
#Takes the max across all actions
def value_iteration(states, actions, P, V, gamma, threshold):
    iteration = 0

    while True:
        delta = 0
        iteration += 1
        #TODO print iteration header

        for s in states:
            if not actions[s]:
                continue
        prev_v = V[s]
        action_values = {}

        for a in actions[s]:
            action_values[a] = value_of_action(s, a, V, P, gamma)

        best_action = max(action_values, key=action_values.get)
        best_val = action_values[best_action]

        #TODO: add print statements

        delta = max(delta, abs(best_val - prev_v))
            V[s] = best_val
        if delta < threshold:
            print(f"\n*** Converged after {iteration} iterations ***")
            break
        return V
