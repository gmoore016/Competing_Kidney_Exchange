import random
import networkx as nx
from tabulate import tabulate
import statistics
from multiprocessing import Pool

# Sets random seed
random.seed(16)

# Tracker to maintain unique ids
id_iterator = 0

# How many periods to run simulation
# Must be greater than max(frequencies)
# All elements of frequencies should be
# (approximately) factors in order to avoid
# having leftovers.
#
# Has to be +1 so modular arithmetic works
RUN_LEN = 351

# How many times to run each parameterization
SAMPLE_SIZE = 10

# How large should the pool be at the start
START_SIZE = 0


class Exchange:
    def __init__(self, start_size, inflow, expiry_rate, frequency):
        # For tracking economy status
        self.patients = nx.Graph()
        self.critical_patients = set()

        # Record parameters of simulation
        self.start_size = start_size
        self.inflow = inflow
        self.expiry_rate = expiry_rate
        self.frequency = frequency

        # For tracking economy's performance
        self.matches = []
        self.expiries = []
        self.sizes = []
        self.ages = []
        self.probs = []

    def add_patients(self, num_patients):
        # Ensures all nodes have unique id
        global id_iterator

        for i in range(num_patients):
            # Creates a node with a personalized probability prob
            # Immediately increments iterator to ensure id is not reused
            new_id = id_iterator
            id_iterator = id_iterator + 1

            # Create node for new patient
            new_prob = random.random()
            self.patients.add_node(new_id, prob=new_prob, age=0)

            # For each existing node, test whether they match
            # Test for each node up to the newest
            for patient in self.patients.nodes(data=True):
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
                    self.patients.add_edge(patient[0], new_id, weight=0)

        return self.patients.nodes()

    def activate_critical(self):
        for patient in list(self.get_patients()):
            if random.random() < self.expiry_rate:
                # Node has become critical
                self.critical_patients.add(patient)

    def run_match(self):
        critical_edges = set()

        # For each critical patient...
        for patient in self.critical_patients:
            # Get list of edges connected to node...
            edges = self.patients.edges(patient, data=True)
            for edge in edges:
                # Add to set of edges we care about
                critical_edges.add((edge[0], edge[1]))

                patient_1 = edge[0]
                patient_2 = edge[1]

                # Boost weight by one since at least one connected node is critical
                self.patients[patient_1][patient_2]['weight'] = self.patients[patient_1][patient_2]['weight'] + 1

        # Generate graph composed only of critical pieces
        # If we don't do this, zero-value edges may be part of
        # maximal matching
        critical_subgraph = self.patients.edge_subgraph(critical_edges)

        # Find the maximal matching now that we've weighted the edges
        max_match = nx.algorithms.max_weight_matching(critical_subgraph)

        # Record how many matches we made
        matches = 2 * len(max_match)

        # Get node attribute data
        # (There's definitely a better way to do this,
        #  but I can't get it to work)
        node_ages = nx.get_node_attributes(self.patients, "age")
        node_probs = nx.get_node_attributes(self.patients, "prob")

        # Now remove all matched patients
        for edge in max_match:
            for patient in edge:
                # Remove it from the unmatched critical patients
                # if it was a critical patient
                if patient in self.critical_patients:
                    self.critical_patients.remove(patient)

                self.add_age(node_ages[patient])
                self.add_prob(node_probs[patient])

                # Remove the node
                self.patients.remove_node(patient)

        # Any unmatched critical patients expire
        expiries = len(self.critical_patients)
        for patient in list(self.critical_patients):
            self.patients.remove_node(patient)
            self.critical_patients.remove(patient)

        # Save data to tracker
        self.add_expiries(expiries)
        self.add_matches(matches)
        self.add_size(self.patients.order())

    def age_patients(self):
        # Age all remaining patients
        age_dict = {}
        for patient in self.get_patients():
            new_age = nx.classes.function.get_node_attributes(self.patients, "age")[patient] + 1
            age_dict[patient] = new_age
        nx.classes.function.set_node_attributes(self.patients, age_dict, "age")

    def dump_garbage(self):
        # When we return the exchange, we don't need all the bulky
        # graph data anymore
        self.patients = 0
        self.critical_patients = 0

    # Myriad getters/setters for tracking
    def get_patients(self):
        return self.patients.nodes()

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

    def get_start_size(self):
        return self.start_size

    def get_inflow(self):
        return self.inflow

    def get_expiry_rate(self):
        return self.expiry_rate

    def get_frequency(self):
        return self.frequency

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


class Simulation:
    def __init__(self, parameterization, results):
        self.start_size = parameterization[0]
        self.inflow = parameterization[1]
        self.expiry_rate = parameterization[2]
        self.frequency = parameterization[3]

        self.results = results

        # Concats ages
        self.ages = []
        for result in results:
            self.ages = self.ages + result.get_ages()

        # Concats probs
        self.probs = []
        for result in results:
            self.probs = self.probs + result.get_probs()

    def get_inflow(self):
        return self.inflow

    def get_expiry_rate(self):
        return self.expiry_rate

    def get_frequency(self):
        return self.frequency

    def get_avg_matches(self):
        return statistics.mean([sum(result.get_matches()) for result in self.results])

    def get_sd_matches(self):
        return statistics.stdev([sum(result.get_matches()) for result in self.results])

    def get_avg_age(self):
        return statistics.mean(self.ages)

    def get_sd_age(self):
        return statistics.stdev(self.ages)

    def get_avg_prob(self):
        return statistics.mean(self.probs)

    def get_sd_prob(self):
        return statistics.stdev(self.probs)


def take_sample(parameterization):
    # Unpack the parameterization
    # Note sample_size is only there so we can use
    # the same tuple as passed to run_sim
    start_size, inflow, expiry_rate, frequency, sample_size = parameterization

    # Initialize exchange
    ex = Exchange(start_size, inflow, expiry_rate, frequency)

    # Add starting pool
    ex.add_patients(start_size)

    # Run the simulation
    # Currently runs for a year
    for i in range(1, RUN_LEN):
        # If inflow is the total number of patients,
        # inflow//frequency is the number of patients
        # that arrive each period
        ex.add_patients(inflow)

        # Each patient in the pool has a chance of becoming critical
        ex.activate_critical()

        # If a patient expires at
        # rate expiry_rate in a day, they expire
        # at rate expiry_rate * 365 in a year
        if not i % frequency:
            # Matched patients are removed
            ex.run_match()

        # Age all remaining patients
        ex.age_patients()

    # Dump elements we don't need anymore
    ex.dump_garbage()

    return ex


def print_table(values, sds, inflows, exp_rates, frequencies):
    for exp_rate in exp_rates:
        table = []
        for inflow in inflows:
            table.append([inflow] + [str(values[freq][inflow][exp_rate]) for freq in frequencies])
            table.append(["(SD)"] + ["(" + str(sds[freq][inflow][exp_rate]) + ")" for freq in frequencies])

        output = tabulate(table, headers=["Inflow\\Frequency"] + [str(freq) for freq in frequencies])
        print("Expiry rate==" + str(exp_rate))
        print(output)


def table_dict(frequencies, inflows):
    """
    Used to store results of simulations in a way
    that is easy to convert to tabular form
    """
    freq_tables = {}
    for freq in frequencies:
        freq_tables[freq] = {}
        for inflow in inflows:
            freq_tables[freq][inflow] = {}
    return freq_tables


def run_sim(parameterization):
    print("Running parameterization: " + str(parameterization), flush=True)
    sample_size = parameterization[4]
    with Pool() as pool:
        results = pool.map(take_sample, [parameterization] * sample_size)
    return Simulation(parameterization, results)


def main():
    # How many samples of each parameterization do we want?
    sample_size = SAMPLE_SIZE

    # How many times more slowly does the "slow" match run?
    frequencies = [
        1,   # Daily
        7,   # Weekly
        30,  # Monthly
        87,  # Quarterly
        350  # Yearly
    ]

    # What is your chance of criticality each period?
    exp_rates = [
        .1,  # Low
        .5,  # Med
        .9   # High
    ]

    # How many new patients arrive each period?
    inflows = [
        10,  # Slow
        50,  # Med
        100  # Fast
    ]

    # The list of possible combinations
    # Will contain tuples of the form (start_size, inflow_exp_rate, freq)
    # which will get passed as the arguments of run_sim
    parameterizations = []

    # Generate a tuple for each combination of the above
    for freq in frequencies:
        for exp_rate in exp_rates:
            for inflow in inflows:
                parameterizations.append((START_SIZE, inflow, exp_rate, freq, sample_size))

    # Run the simulations
    simulations = []
    for parameterization in parameterizations:
        simulations.append(run_sim(parameterization))

    # Pulls and saves the results of each simulation to the dict match_tables, then
    # outputs it in tabular format

    # Print match count table
    match_values = table_dict(frequencies, inflows)
    match_sds = table_dict(frequencies, inflows)
    for sim in simulations:
        match_values[sim.get_frequency()][sim.get_inflow()][sim.get_expiry_rate()] = sim.get_avg_matches()
        match_sds[sim.get_frequency()][sim.get_inflow()][sim.get_expiry_rate()] = sim.get_sd_matches()
    print("Matches:")
    print_table(match_values, match_sds, inflows, exp_rates, frequencies)

    # Print average matched age table
    age_values = table_dict(frequencies, inflows)
    age_sds = table_dict(frequencies, inflows)
    for sim in simulations:
        age_values[sim.get_frequency()][sim.get_inflow()][sim.get_expiry_rate()] = sim.get_avg_age()
        age_sds[sim.get_frequency()][sim.get_inflow()][sim.get_expiry_rate()] = sim.get_sd_age()
    print("Ages:")
    print_table(age_values, age_sds, inflows, exp_rates, frequencies)

    # Print average matched prob table
    prob_values = table_dict(frequencies, inflows)
    prob_sds = table_dict(frequencies, inflows)
    for sim in simulations:
        prob_values[sim.get_frequency()][sim.get_inflow()][sim.get_expiry_rate()] = sim.get_avg_prob()
        prob_sds[sim.get_frequency()][sim.get_inflow()][sim.get_expiry_rate()] = sim.get_sd_prob()
    print("Probs:")
    print_table(prob_values, prob_sds, inflows, exp_rates, frequencies)


if __name__ == "__main__":
    main()
