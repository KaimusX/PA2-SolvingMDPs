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



#Part 1: Value Iteration
def value_iteration(states, actions, transition_model, reward_function, gamma, epsilon):
        V = {s: 0 for s in states} ##every state has est. value of 0
        
        while True:
            delta = 0
            for s in states:
                v = V[s]
                #Bellman Equation applied to every state
                #for each action a, computes expected value from
                #taking that action in state s. Picks max Value
                V[s] = max(sum(transition_model(s,a,s_next) * 
                    (reward_function(s,a,s_next)+ gamma *
                        V[s_next])for s_next in states) for a in actions)
                delta = max(delta, abs(v - V[s])) #keeps track of biggest change seen this iteration
                if delta < epsilon:
                    break
                #Conversion:loop stops once no state's value changes by more than epsilon
                #For current state (s), find best possible action
            policy = {}
            for s in states:
                policy[s] = max(actions,
                                    key = lambda a: sum(
                                        transition_model(s,a,s_next *
                                            reward_function(s,a,s_next) + gamma *
                                            V[s_next])
                                    for s_next in states))
                return policy, V
