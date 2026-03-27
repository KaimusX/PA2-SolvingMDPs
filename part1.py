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
    
    # level 3 states NOTE: for 'any' we should write out each state instead.

    # level 4 state, terminal
    "11aFin": {}
}

# Initialize value estimates to zilch 0
V = {s: 0.0 for s in states}

# discount factor
gamma = 0.99

# Bellman equation, NOTE: does not have max over actions yet.
def value_of_action(state, action, V, P, gamma):
    total = 0.0
    for prob, next_state, reward in P[state][action]:
        total += prob * (reward + gamma * V[next_state])
    return total