import numpy as np
import constants
import math
import plotly.express as px
import pandas as pd


# learning rate
def learning_rate(episode):
    return 10000000 / (10000000 + episode)                          # leaning factor for tab. q learn


# utility function
def utility_function(cost):
    return np.exp(constants.alpha_utility * cost)                   # utility function see shen et al.


# CVaR risk
def cvar_risk(sorted_costs):
    index = int((1-constants.alpha_cvar)*len(sorted_costs))
    highest_costs = sorted_costs[index:]
    cvar = sum(highest_costs) / len(highest_costs)
    return cvar


# compute running mean
def running_mean(episode_no, m, new_cost):
    return (episode_no / (episode_no + 1)) * m + (1 / (episode_no + 1)) * new_cost


# compute running variance
def running_var(n, old_m, m, var, new_cost):

    # see wikipedia:
    # https://de.wikipedia.org/wiki/Stichprobenvarianz_(Sch%C3%A4tzfunktion)#Berechnung_f%C3%BCr_auflaufende_Messwerte
    old_s = (n-2) / (n-1) * var
    new_var = n / (n-1) * ((n-1) / n * (old_s + old_m**2) + new_cost**2 / n - m**2)
    return new_var


def risk_measure_absolute(costs):

    # Stone's risk measure using k = 2, Y_0 = 1 and A = 1
    # see pedersen and satchell 1998

    risk = 0
    for c in costs:
        if c > 1:
            risk += 1/len(costs) * (c - 1)**2

    return math.sqrt(risk)


def risk_measure_expectation(costs):

    # semi standard deviation
    # see pedersen and satchell 1998

    mean = sum(costs) / len(costs)

    risk = 0
    for c in costs:
        if c > mean:
            risk += 1 / len(costs) * (c - mean) ** 2

    return math.sqrt(risk)


# plot a moving average of data over episodes
def plot_moving_avg(data, title, episodes, strategy_type):
    # calculate moving average
    averages = []
    length = constants.moving_average_length
    for i in range(len(data) - length + 1):
        average = sum(data[i:i + length]) / length
        averages.append(average)

    df = pd.DataFrame({'episode': range(episodes - length + 1), title: averages})

    if strategy_type == "utility_function":
        risk_factor = constants.alpha_utility
    elif strategy_type == "cvar":
        risk_factor = constants.alpha_cvar
    elif strategy_type == "mean_variance":
        risk_factor = constants.mv_risk_factor
    else:
        risk_factor = "---"

    fig = px.line(df, x='episode', y=title, title='moving average: ' + title + ' over ' + str(length) + ' episodes.'
                                                  + ' strategy type: ' + str(strategy_type)
                                                  + '\t \t' + 'risk factor: ' + str(risk_factor)
                                                  + ', p: ' + str(constants.new_package_prob)
                                                  + ', lambda: ' + str(constants.send_prob)
                                                  + ', energy weight: ' + str(constants.energy_weight))
    fig.show()
