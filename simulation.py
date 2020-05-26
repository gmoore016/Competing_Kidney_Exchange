import numpy
import networkx as nx

# Number of patients present in each exchange at the start of the process
START_SIZE = 100

# Odds a patient will pass after each period
DEPARTURE_CHANCE = 0.05

# How much more frequently the fast exchange matches
FREQ = 20

# How many new patients exchange receives each period
INFLOW = 10

# How many periods to run
TIME_LEN = 50

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
        new_prob = numpy.random.uniform()
        exchange.add_node(new_id, prob=new_prob)

        # For each existing node, test whether they match
        # Test for each node up to the newest
        for patient in exchange.nodes(data=True):
            # Skip if you're looking at the new node
            if patient[0] == new_id:
                continue

            # Get probability from existing node
            old_prob = patient[1]['prob']

            # Test whether there's a match, and if so create an edge
            match_chance = old_prob * new_prob
            if numpy.random.uniform() < match_chance:
                exchange.add_edge(patient[0], new_id)

    return exchange.nodes()


def pass_time(exchange):
    """
    Whenever time passes, some patients are lost
    """
    expirations = 0

    for patient in list(exchange.nodes()):
        if numpy.random.uniform() < DEPARTURE_CHANCE:
            exchange.remove_node(patient)
            expirations = expirations + 1

    return expirations


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

    return 2 * len(max_match)


def main():
    # Initiates graphs for each exchange
    weekly = nx.Graph()
    monthly = nx.Graph()

    print("Weekly matches: ")
    expirations = 0
    matches = 0
    add_patients(weekly, START_SIZE)
    for i in range(TIME_LEN):
        # Add patients
        add_patients(weekly, INFLOW)
        # Match patients if scheduled
        matches = match(weekly) + matches
        # Atrophy patients
        expirations = pass_time(weekly) + expirations
    print(matches)
    print(expirations)

    print("Monthly matches: ")
    expirations = 0
    matches = 0
    add_patients(monthly, START_SIZE)
    for i in range(TIME_LEN):
        # Add patients
        add_patients(monthly, INFLOW)

        # Match patients if scheduled
        if not i % FREQ:
            matches = match(monthly) + matches

        # Atrophy patients
        expirations = pass_time(monthly) + expirations
    print(matches)
    print(expirations)


main()
