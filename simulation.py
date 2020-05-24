import numpy
import networkx as nx

# Number of patients present in each exchange at the start of the process
START_SIZE = 10

# Odds a patient will pass after each period
DEPARTURE_CHANCE = 0.25

id_iterator = 0


def add_patients(exchange, num_patients):
    """
    Adds numpatients patients to exchange
    """
    # Ensures all nodes have unique id
    global id_iterator

    for i in range(num_patients):
        # Creates a node with a personalized probability prob
        # Immediately increments iterator to ensure id is not reused
        new_id = id_iterator
        id_iterator = id_iterator + 1

        # Create node for new patient
        exchange.add_node(new_id, prob = numpy.random.uniform())

        # For each existing node, test whether they match
        # Test for each node up to the newest
        for patient in exchange.nodes():

            # Test whether there's a match, and if so create an edge
            match_chance = exchange[patient]['prob'] *  exchange[new_id]['prob']
            if numpy.random.uniform() < match_chance:
                exchange.add_edge(patient, new_id)


def pass_time(exchange):
    """
    Whenever time passes, some patients are lost
    """
    for patient in exchange.nodes():
        if numpy.random.uniform() < DEPARTURE_CHANCE:
            exchange.remove_node(patient)


def match(exchange):
    """
    Given an exchange, finds the maximal matching and removes
    all matched elements
    """

    # Finds the maximal matching of the exchange
    max_match = nx.algorithms.matching.maximal_matching(exchange)

    # Remove all matched patients
    for pair in max_match:
        for patient in pair:
            exchange.remove_node(patient)


def main():
    weekly = nx.Graph()
    monthly = nx.Graph()

    add_patients(weekly, START_SIZE)
    add_patients(monthly, START_SIZE)

    match(weekly)
    pass_time(weekly)


main()
