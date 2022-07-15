# model constants
rho_d = 0.2  # power constraint
pilot_sequence_constant = 1  # square of abs of phi^H * phi
tau_p = 20  # pilot sequence length
K = 5  # number of user equipments
M = 15  # number of access points
rho_u = 1  # uplink normalized transition power
map_size = 1000  # size of the map in meters
f = 1.9e9  # transmission frequency
c = 299792458  # speed of light
alpha = 3  # pathloss exponent

# ai constants
train_episodes = 1000
test_episodes = 1000
epsilon_0 = 0.9
decay = 0.999
