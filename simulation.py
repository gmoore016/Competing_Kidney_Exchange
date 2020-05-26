import numpy
import json
import random
import networkx as nx

with open("parameters.json") as param_file:
    parameters = json.load(param_file)

# Number of patients present in each exchange at the start of the process
START_SIZE = parameters["START_SIZE"]

# Odds a patient will pass after each period
EX_RATE = parameters["EX_RATE"]

# How much more frequently the fast exchange matches
FREQ = parameters["FREQ"]

# How many new patients exchange receives each period
INFLOW = parameters["INFLOW"]

# How many periods to run
TIME_LEN = parameters["TIME_LEN"]

id_iterator = 0


class Tracker:
    def __init__(self):
        # For tracking economy's performance
        self.matches = []
        self.expiries = []
        self.sizes = []
        self.ages = []
        self.probs = []

    def add_matches(self, matches):
        self.matches.append(matches)

    def add_expiries(self, expiries):
        self.expiries.append(expiries)

    def add_size(self, sizes):
        self.sizes.append(sizes)

    def add_age(self, age):
        self.ages.append(age)

    def add_prob(self, difficulty):
        self.probs.append(difficulty)

    def get_matches(self):
        return self.matches

    def get_expiries(self):
        return self.expiries

    def get_sizes(self):
        return self.sizes

    def get_ages(self):
        return self.ages

    def get_probs(self):
        return self.probs


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
        exchange.add_node(new_id, prob=new_prob, age=0)

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


def pass_time(exchange, expiry_rate, tracker):
    """
    Whenever time passes, some patients are lost
    """
    expirations = 0
    matches = 0

    node_ages = nx.get_node_attributes(exchange, "age")
    node_probs = nx.get_node_attributes(exchange, "prob")

    for patient in list(exchange.nodes()):
        # Possible node will have been removed by neighbor
        if patient not in exchange.nodes():
            continue

        if numpy.random.uniform() < expiry_rate:
            # Node has become critical
            # Choose a random neighbor
            neighbors = list(exchange.neighbors(patient))

            # If they have at least one feasible match, match them
            if neighbors:
                # Remove neighbor from graph after matching
                match = random.choice(neighbors)
                tracker.add_age(node_ages[match])
                tracker.add_prob(node_probs[match])
                exchange.remove_node(match)
                matches = matches + 2

            # If they have no neighbors, they expire
            else:
                expirations = expirations + 1

            # Either way, remove the patient
            tracker.add_age(node_ages[patient])
            exchange.remove_node(patient)

    # Age all remaining patients
    for patient in exchange.nodes():
        new_age = node_ages[patient] + 1
        nx.classes.function.set_node_attributes(exchange, new_age, "age")

    # Track data
    tracker.add_expiries(expirations)
    tracker.add_matches(matches)
    tracker.add_size(exchange.order())


def main():
    # Initiates graphs for each exchange
    weekly = nx.Graph()
    monthly = nx.Graph()

    # Initiates trackers for stats
    weekly_stats = Tracker()
    monthly_stats = Tracker()

    add_patients(weekly, START_SIZE)
    print("Weekly:")
    weekly_sequence = []
    for i in range(FREQ * TIME_LEN):
        # Add patients
        add_patients(weekly, INFLOW)

        # Patients become critical
        pass_time(weekly, EX_RATE, weekly_stats)
        weekly_sequence.append(weekly.order())

    print("Matches: " + str(sum(weekly_stats.get_matches())))
    print("Expiries: " + str(sum(weekly_stats.get_expiries())))
    print("Pool Size: " + str(sum(weekly_stats.get_sizes())//len(weekly_stats.get_sizes())))
    print("Average Matched Age: " + str(sum(weekly_stats.get_ages())/len(weekly_stats.get_ages())))

    remaining_ages = nx.get_node_attributes(weekly, 'age')
    total = 0
    for node in remaining_ages:
        total = total + remaining_ages[node]
    print("Average Unmatched Age: " + str(total/len(remaining_ages)))


    print("Monthly:")
    monthly_sequence = []
    add_patients(monthly, START_SIZE)
    modified_expiry = 1 - (1 - EX_RATE) ** FREQ
    for i in range(TIME_LEN):
        # Add patients
        add_patients(monthly, FREQ * INFLOW)

        # Atrophy patients
        pass_time(monthly, modified_expiry, monthly_stats)
        monthly_sequence.append(monthly.order())

    print("Matches: " + str(sum(monthly_stats.get_matches())))
    print("Expiries: " + str(sum(monthly_stats.get_expiries())))
    print("Pool Size: " + str(sum(monthly_stats.get_sizes())/len(monthly_stats.get_sizes())))
    print("Average Age: " + str(FREQ * sum(weekly_stats.get_ages())/len(weekly_stats.get_ages())))

    remaining_ages = nx.get_node_attributes(monthly_stats, 'age')
    total = 0
    for node in remaining_ages:
        total = total + remaining_ages[node]
    print("Average Unmatched Age: " + str(total/len(remaining_ages)))

main()
