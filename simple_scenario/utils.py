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


def running_cvar_risk(sorted_costs, quantile, cost, cvar_sum):
    index = int((1 - quantile) * len(sorted_costs))
    index_old = int((1-quantile) * (len(sorted_costs) - 1))
    leng = len(sorted_costs) - index
    if leng > 0:
        if cost < sorted_costs[index]:
            if index == index_old:
                cvar_sum += sorted_costs[index]
        else:
            if index == index_old:
                cvar_sum += cost
            else:
                cvar_sum += cost - sorted_costs[index-1]
        cvar = cvar_sum / leng
    else:
        cvar = 0

    return cvar, cvar_sum


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


def bar_chart(data, title, rev, experiment):

    # sort dict by title entry
    # sort colors accordingly
    colors = px.colors.qualitative.Plotly
    data = dict(zip(data['strategy'], zip(data[title], colors)))
    data = dict(sorted(data.items(), key=lambda x: x[1][0], reverse=rev))
    colors = list(zip(*data.values()))[1]
    data = {'strategy': list(data), title: list(zip(*data.values()))[0]}

    df = pd.DataFrame(data)
    fig = px.bar(df, x='strategy', y=title, color='strategy', color_discrete_sequence=colors, text=title,
                 title=title + '\t '
                             + 'p: ' + str(constants.new_package_prob)
                             + ', lambda: ' + str(constants.send_prob)
                             + ', energy weight: ' + str(constants.energy_weight)
                             + '\t '  # + 'train eps: ' + str(constants.train_episodes) + ', '
                             + ' test eps: ' + str(constants.test_episodes)
                             + ', experiment: ' + experiment)
    fig.update_layout(font=dict(size=24))
    fig.update_traces(textfont_size=30)
    # fig.show()
    fig.update_layout(autosize=False, width=1904, height=931)
    fig.write_image("results/" + experiment + "_" + title + ".png")


def bar_chart_manual():

    # colors = ['#000033', '#000066', '#000099', '#0000CC', '#0000FF', '#3366FF']
    non_learning_data = \
        {'strategy': ['always', 'random', 'send_once', 'threshold', 'basic_monte_carlo', 'optimal_threshold'],
            'average cost':         [5.1112, 4.7214, 3.7197, 3.6609, 3.5267, 3.5083],
            'semi std deviation':   [1.2526, 1.6188, 1.1012, 1.2663, 1.1831, 1.1605],
            'risky states':         [0.0023, 0.0166, 0.0044, 0.0045, 0.0025, 0.0028],
            'Fishburn measure':     [1.9806, 1.8340, 0.9981, 1.1357, 0.9957, 0.9650]}

    # colors = ['#000033', '#CC0099', '#3366FF']
    value_iteration_data = \
        {'strategy': ['always', 'value_iteration', 'optimal_threshold'],
            'average cost':         [4.6830, 3.5245, 3.5053]}

    # colors = ['#000033', '#660000', '#990000', '#CC0000', '#FF0000', '#3366FF']
    risk_neutral_data = \
        {'strategy':
         ['random', 'REINFORCE_action_prob', 'network_Q', 'REINFORCE_sigmoid', 'tabular_Q', 'optimal_threshold'],
         'average cost':            [4.7318, 4.6548, 3.7266, 3.5351, 3.5324, 3.5097],
         'semi std deviation':      [1.6280, 1.4678, 1.3106, 1.1615, 1.2170, 1.1580],
         'risky states':            [0.0171, 0.0080, 0.0053, 0.0027, 0.0034, 0.0028],
         'Fishburn measure':        [1.9947, 1.7999, 1.2040, 0.9753, 1.0366, 0.9627]}

    # colors = ['#000033', '#003300', '#006600', '#FF0000', '#009900', '#00CC00', '#00FF00', '#33FF66', '#3366FF']
    tabular5_data = \
        {'strategy': ['random', 'cvar_tabular', 'mean_variance_tabular', 'tabular_Q', 'semi_std_deviation_tabular',
                      'fishburn_tabular', 'risk_states_tabular', 'utility_function_tabular', 'optimal_threshold'],
         'average cost':            [4.7256, 3.7604, 3.6964, 3.6581, 3.6513, 3.6490, 3.6299, 3.5771, 3.5124],
         'semi std deviation':      [1.6225, 1.4912, 1.3446, 1.3818, 1.3529, 1.3419, 1.3247, 1.2471, 1.1670],
         'risky states':            [0.0168, 0.0087, 0.0046, 0.0061, 0.0051, 0.0045, 0.0038, 0.0039, 0.0030],
         'Fishburn measure':        [1.9862, 1.4028, 1.2291, 1.2537, 1.2219, 1.2096, 1.1844, 1.0832, 0.9737]}

    # colors = ['#000033', '#003300', '#006600', '#FF0000', '#009900', '#00CC00', '#00FF00', '#33FF66', '#3366FF']
    tabular6_data = \
        {'strategy': ['random', 'cvar_tabular', 'mean_variance_tabular', 'tabular_Q', 'semi_std_deviation_tabular',
                      'fishburn_tabular', 'risk_states_tabular', 'utility_function_tabular', 'optimal_threshold'],
         'average cost':            [4.7258, 3.6667, 3.6115, 3.5281, 3.5499, 3.5314, 3.5232, 3.5655, 3.5072],
         'semi std deviation':      [1.6154, 1.2947, 1.2382, 1.2053, 1.2115, 1.2077, 1.1966, 1.2872, 1.1577],
         'risky states':            [0.0164, 0.0045, 0.0035, 0.0032, 0.0031, 0.0032, 0.0027, 0.0048, 0.0027],
         'Fishburn measure':        [1.9803, 1.1679, 1.0876, 1.0223, 1.0368, 1.0262, 1.0112, 1.1241, 0.9617]}

    colors = ['#003300', '#000033', '#33FF66', '#006600', '#009900', '#00FF00', '#00CC00', '#990000', '#3366FF']
    network_data = \
        {'strategy': ['cvar_network', 'random', 'utility_function_network', 'mean_variance_network',
                      'semi_std_deviation_network', 'risk_states_network', 'fishburn_network', 'network_Q',
                      'optimal_threshold'],
         'average cost':            [4.8537, 4.7220, 3.9723, 3.7814, 3.7543, 3.7384, 3.7293, 3.7152, 3.5079],
         'semi std deviation':      [1.3698, 1.6204, 1.3498, 1.2831, 1.3182, 1.3044, 1.3109, 1.3009, 1.1536],
         'risky states':            [0.0243, 0.0167, 0.0125, 0.0053, 0.0082, 0.0044, 0.0054, 0.0051, 0.0025],
         'Fishburn measure':        [2.3350, 1.9824, 1.3923, 1.2096, 1.2361, 1.2038, 1.2076, 1.1896, 0.9569]}

    experiment = 'network'
    data = network_data
    title = 'Fishburn measure'
    data = dict(zip(data['strategy'], zip(data[title], colors)))
    data = dict(sorted(data.items(), key=lambda x: x[1][0], reverse=False))
    colors = list(zip(*data.values()))[1]
    data = {'strategy': list(data), title: list(zip(*data.values()))[0]}
    df = pd.DataFrame(data)
    fig = px.bar(df, x='strategy', y=title, color='strategy', color_discrete_sequence=colors, text=title,
                 title=title + '\t '
                             + 'p: ' + str(constants.new_package_prob)
                             + ', lambda: ' + str(constants.send_prob)
                             + ', energy weight: ' + str(constants.energy_weight)
                             + '\t '  # + 'train eps: ' + str(constants.train_episodes) + ', '
                                      #  + ' test eps: ' + str(constants.test_episodes) + ', '
                             + 'experiment: ' + experiment)
    fig.update_layout(font=dict(size=24))
    fig.update_traces(textfont_size=30)
    fig.show()
    fig.update_layout(autosize=False, width=1904, height=931)
    fig.write_image("results/" + experiment + "_" + title + ".png")


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def line_plot(data):

    df = pd.DataFrame(dict(eps=range(len(data)), data=data))
    fig = px.line(df, x="eps", y="data")
    fig.show()
