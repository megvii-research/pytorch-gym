"""
Here are some default network structures as well as
some lists containing the names of different agents and environments.
"""

net_topology_pol_vec = [
    {'kind': 'dense', 'units': 64},
    {'kind': 'Tanh'},
    {'kind': 'dense', 'units': 64},
    {'kind': 'Tanh'},
    {'kind': 'dense', 'units': 64},
    {'kind': 'Tanh'},
]

net_topology_v_vec = [
    {'kind': 'dense', 'units': 64},
    {'kind': 'Tanh'},
    {'kind': 'dense', 'units': 64},
    {'kind': 'Tanh'},
    {'kind': 'dense', 'units': 64},
    {'kind': 'Tanh'},
]
