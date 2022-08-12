import numpy as np
import constants
import math
import plotly.express as px
import pandas as pd


# learning rate
def learning_rate(episode):
    return 10000000 / (1000000000 + episode)  # leaning factor for tab. q learn around 0.01


# utility function
def utility_function(cost, risk_factor):
    cap = 500
    arg = max(risk_factor * cost, -cap)
    arg = min(arg, cap)
    return np.exp(arg)  # utility function see shen et al.


# CVaR risk
def cvar_risk(sorted_costs, risk_factor):
    index = int((1 - risk_factor) * len(sorted_costs))
    highest_costs = sorted_costs[index:]
    if len(highest_costs) > 0:
        cvar = sum(highest_costs) / len(highest_costs)
    else:
        cvar = 0
    return cvar


# compute running mean
def running_mean(episode_no, m, new_cost):
    return (episode_no / (episode_no + 1)) * m + (1 / (episode_no + 1)) * new_cost


# compute running variance
def running_var(n, old_m, m, var, new_cost):
    # see wikipedia:
    # https://de.wikipedia.org/wiki/Stichprobenvarianz_(Sch%C3%A4tzfunktion)#Berechnung_f%C3%BCr_auflaufende_Messwerte
    old_s = (n - 2) / (n - 1) * var
    new_var = n / (n - 1) * ((n - 1) / n * (old_s + old_m ** 2) + new_cost ** 2 / n - m ** 2)
    return new_var


def stone_measure(costs):
    # Stone's risk measure using k = 2, Y_0 = 1 and A = 1
    # see pedersen and satchell 1998

    risk = 0
    for c in costs:
        if c > 1:
            risk += 1 / len(costs) * (c - 1) ** 2

    return math.sqrt(risk)


def fishburn_measure(costs, target):
    # Fishburn's alpha-t measure with alpha = 2 and variable target
    # see pedersen and satchell 1998

    risk = 0
    for c in costs:
        if c > target:
            risk += 1 / len(costs) * (c - target) ** 2

    return math.sqrt(risk)


def semi_std_dev(costs):
    # semi standard deviation
    # see pedersen and satchell 1998

    mean = sum(costs) / len(costs)

    risk = 0
    for c in costs:
        if c > mean:
            risk += 1 / len(costs) * (c - mean) ** 2

    return math.sqrt(risk)


def bar_chart(data, title, rev):

    # sort dict by title entry
    # sort colors accordingly
    colors = px.colors.qualitative.Plotly
    data = dict(zip(data['strategy'], zip(data[title], colors)))
    data = dict(sorted(data.items(), key=lambda x: x[1][0], reverse=rev))
    colors = list(zip(*data.values()))[1]
    data = {'strategy': list(data), title: list(zip(*data.values()))[0]}

    df = pd.DataFrame(data)
    fig = px.bar(df, x='strategy', y=title, color='strategy', color_discrete_sequence=colors, text=title,
                 title=title + '\t \t \t \t'
                             + 'p: ' + str(constants.new_package_prob)
                             + ', lambda: ' + str(constants.send_prob)
                             + ', energy weight: ' + str(constants.energy_weight)
                             + '\t \t \t \t'  # + 'train eps: ' + str(constants.train_episodes) + ', '
                             + ' test eps: ' + str(constants.test_episodes))
    fig.update_layout(font=dict(size=24))
    fig.update_traces(textfont_size=30)
    # fig.show()
    fig.update_layout(autosize=False, width=1904, height=931)
    fig.write_image("results/rn" + title + ".png")


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def line_plot(data):

    df = pd.DataFrame(dict(eps=range(len(data)), data=data))
    fig = px.line(df, x="eps", y="data")
    fig.show()
