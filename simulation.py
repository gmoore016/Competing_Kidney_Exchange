import random
import networkx as nx

# Sets random seed
random.seed(16)

# Tracker to maintain unique ids
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
        new_prob = random.random()
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
            if random.random() < match_chance:
                # Starts with weight of zero since not currently
                # connected to critical node
                exchange.add_edge(patient[0], new_id, weight=0)

    return exchange.nodes()


def run_match(exchange, critical_patients, tracker):
    # For each critical patient...
    for patient in critical_patients:
        # Get list of edges connected to node...
        edges = exchange.edges(patient, data=True)
        for edge in edges:
            patient_1 = edge[0]
            patient_2 = edge[1]

            # Boost weight by one since at least one connected node is critical
            exchange[patient_1][patient_2]['weight'] = exchange[patient_1][patient_2]['weight'] + 1

    # Find the maximal matching now that we've weighted the edges
    max_match = nx.algorithms.max_weight_matching(exchange)

    # Record how many matches we made
    matches = 2 * len(max_match)

    # Get node attribute data
    # (There's definitely a better way to do this,
    #  but I can't get it to work)
    node_ages = nx.get_node_attributes(exchange, "age")
    node_probs = nx.get_node_attributes(exchange, "prob")

    # Now remove all matched patients
    for edge in max_match:
        for patient in edge:
            # Remove it from the unmatched critical patients
            # if it was a critical patient
            if patient in critical_patients:
                critical_patients.remove(patient)

            tracker.add_age(node_ages[patient])
            tracker.add_prob(node_probs[patient])

            # Remove the node
            exchange.remove_node(patient)

    # Any unmatched critical patients expire
    expiries = len(critical_patients)
    for patient in critical_patients:
        exchange.remove_node(patient)

    # Save data to tracker
    tracker.add_expiries(expiries)
    tracker.add_matches(matches)
    tracker.add_size(exchange.order())


def run_sim(start_size, inflow, expiry_rate, frequency):
    # Initialize objects
    ex = nx.Graph()
    stats = Tracker()

    # Add starting pool
    add_patients(ex, start_size)

    # Generate set of critical patients
    critical_patients = set()

    # Run the simulation
    # Currently runs for a year
    for i in range(1, 51):
        # If inflow is the total number of patients,
        # inflow//frequency is the number of patients
        # that arrive each period
        add_patients(ex, inflow)

        # Each patient in the pool has a chance of becoming critical
        for patient in list(ex.nodes()):
            if random.random() < expiry_rate:
                # Node has become critical
                critical_patients.add(patient)

        # If a patient expires at
        # rate expiry_rate in a day, they expire
        # at rate expiry_rate * 365 in a year
        if not i % frequency:
            # Matched patients are removed
            run_match(ex, critical_patients, stats)

            # Unmatched critical patients expire
            critical_patients = set()

        # Age all remaining patients
        age_dict = {}
        for patient in ex.nodes():
            new_age = nx.classes.function.get_node_attributes(ex, "age")[patient] + 1
            age_dict[patient] = new_age
        nx.classes.function.set_node_attributes(ex, age_dict, "age")

    print("Frequency==" + str(frequency))

    print("Matches: " + str(sum(stats.get_matches())))
    print("Expiries: " + str(sum(stats.get_expiries())))
    print("Pool Size: " + str(ex.order()))
    print("Average Matched Age: " + str(sum(stats.get_ages()) / len(stats.get_ages())))

    # TODO: Expired age

    remaining_ages = nx.get_node_attributes(ex, 'age')
    if not remaining_ages:
        print("No pending patients!")

    else:
        total = 0
        for node in remaining_ages:
            total = total + remaining_ages[node]
        print("Average Pending Age: " + str(total / len(remaining_ages)))


