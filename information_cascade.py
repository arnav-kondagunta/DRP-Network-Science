import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

'''
This program simulates a sequential simply binary information cascade. 
We assume each individual chooses to accept or reject an option, which has an unknown binary state: good or bad. 
Before deciding, each individual receives a private, imperfect signal about the state, and also observes the actions of everyone who has decided before them.
'''

def simulate_information_cascade(n, p, q, seed=None):
    '''
    Simulates a sequential Bayesian information cascade model
    
    Input:
        n: number of individuals in simulation
        p: probability of good state
        q: probability of signal matching the state, represents signal accuracy (must be greater than 0.5)
        seed: optional if you want to set a random seed
    '''
    
    if q < 0.5:
        raise ValueError("q must be greater than or equal to 0.5")

    #check if user passed a seed value
    if seed is not None:
        np.random.seed(seed)

    #simulate each individual receiving a signal
    states = ["good", "bad"]
    true_state = np.random.choice(states, p=[p,1-p])
    signals = ["high", "low"]
    probabilities = [q, 1-q] if true_state == "good" else [1-q, q]

    #calculate probability using Bayes rule
    def posterior(a, b):
        numerator = p*q**a*(1-q)**b
        denominator = numerator + (1-p)*(1-q)**a*q**b
        return numerator / denominator

    #uses inference probability and private signal to identify individual's action
    def resolve_action(inference, signal):
        if inference > 0.5:
            return "accept"
        elif inference < 0.5:
            return "reject"
        else:
            return "accept" if signal == "high" else "reject"

    results = pd.DataFrame(columns=["signal_type","action","matches_true_state","inference_prob","is_in_cascade","cascade_direction"])
    num_accept = 0
    num_reject = 0
    step = -1

    for i in range(n):
        draw = np.random.choice(signals, p=probabilities)
        results.at[i, "signal_type"] = draw

        #cascade exists if public info alone already decides the action regardless of private signal
        inference_if_high = posterior(num_accept + 1, num_reject)
        inference_if_low = posterior(num_accept, num_reject + 1)
        action_if_high = resolve_action(inference_if_high, "high")
        action_if_low = resolve_action(inference_if_low, "low")
        in_cascade = action_if_high == action_if_low

        results.at[i, "is_in_cascade"] = in_cascade
        results.at[i, "cascade_direction"] = action_if_high if in_cascade else None

        bayesian_inference = inference_if_high if draw == "high" else inference_if_low
        results.at[i, "inference_prob"] = bayesian_inference
        action = resolve_action(bayesian_inference, draw)
        results.at[i, "action"] = action

        #once in a cascade, actions no longer reveal private signals public counts must freeze
        if not in_cascade:
            step = i
            if action == "accept":
                num_accept += 1
            else:
                num_reject += 1

    results["matches_true_state"] = (
        ((true_state == "good") & (results["action"] == "accept")) |
        ((true_state == "bad") & (results["action"] == "reject"))
    )

    cascade_start = step + 1 if in_cascade == True else None

    return true_state, cascade_start, results


#monte carlo simulation to pull data from N trials 
n = 50
p = 0.5
q = 0.7
N = 1000
correct_cascades = 0
incorrect_cascades = 0
cascade_start_steps = []

for x in range(N):
    state, cascade_start, results = simulate_information_cascade(n, p, q)
    if results.loc[results.index[-1], "is_in_cascade"]:
        cascade_start_steps.append(cascade_start)
        if results.loc[results.index[-1], "matches_true_state"]:
            correct_cascades += 1
        else:
            incorrect_cascades += 1

cascade_prob = (correct_cascades + incorrect_cascades) / N
correct_prob = correct_cascades / (correct_cascades + incorrect_cascades) if cascade_prob > 0 else 0
avg_cascade_start = np.mean(cascade_start_steps) if cascade_prob > 0 else 0

print(f"Probability of a cascade occurring is {cascade_prob:.2%}")
print(f"Proabibilty that a cascade is correct is {correct_prob:.2%}")
print(f"Average step when the cascade began is {avg_cascade_start}")

#graph the distribution of cascade start steps
if cascade_start_steps:
    bins = np.arange(min(cascade_start_steps), max(cascade_start_steps) + 2) - 0.5
    plt.hist(cascade_start_steps, bins=bins, edgecolor="black")
    plt.xlabel("Number of individuals who decided before the cascade began")
    plt.ylabel("Frequency")
    plt.title(f"Distribution of Cascade Start Steps (n={n}, p={p}, q={q}, N={N})")
    plt.show()
else:
    print("No cascades occurred across the trials, so there is nothing to plot.")



    


