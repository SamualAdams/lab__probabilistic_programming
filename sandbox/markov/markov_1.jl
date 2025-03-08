using QuantEcon
P = [0.9 0.1; 0.4 0.6]  # Transition probabilities
mc = MarkovChain(P, ["Unemployed", "Employed"])
simulate(mc, 5, init=1)  # Start unemployed