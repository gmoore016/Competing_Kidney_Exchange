import numpy

# Number of patients present in each exchange at the start of the process
START_SIZE = 10

# Odds a patient will pass after each period
DEPARTURE_CHANCE = 0.25


class Patient:
    """
    Represents a single patient. Has some probability likelihood p, and
    some unique id id.
    """
    def __init__(self, id):
        self.prob = numpy.random.uniform()
        self.id = id

    def get_prob(self):
        return self.prob

    def get_id(self):
        return self.id


class Exchange:
    """
    Represents an independently running kidney exchange. Contains several Patients,
    as well as a graph representing which patients are compatible with which others.
    """
    def __init__(self):
        """
        Creates an empty exchange with no patients and no edges
        """
        # A set of the patient objects present in the hospital
        self.patients = set()

        # A dictionary representing edges. If an edge exists from
        # n1 to n2, edges[n1] == n2 and edges[n2] == n1
        self.edges = {}

    def get_patients(self):
        """
        Retrieves the set of patients in the hospital
        """
        return self.patients

    def get_edges(self):
        return self.edges

    def add_patients(self, n):
        """
        Adds n patients to the exchange, as well as calculating
        edges
        """
        for i in range(n):
            # Adds a patient with id equal to the number of existing patients
            # plus their position within the new patients
            new_patient = Patient(len(self.patients) + 1)

            # Generates an empty set for storing edges
            self.edges[new_patient.get_id()] = set()

            # For each current patient...
            for patient in self.patients:
                # ...test if there is a match
                match_chance = patient.get_prob() * new_patient.get_prob()
                # If so, add an edge for the pair of patients
                if numpy.random.uniform() < match_chance:
                    self.add_edge(new_patient.get_id(), patient.get_id())

            # Finally, add the new patient to the set of patients
            self.patients.add(new_patient)

    def add_edge(self, n1, n2):
        """
        Add an edge to edges from n1 to n2
        """
        self.edges[n1].add(n2)
        self.edges[n2].add(n1)

    def match(self):
        """
        Matches and removes the maximum number of possible patients
        given the edges provided
        """
        # Find set of matches
        matches = self.max_matches()

        # Remove matched patients
        for match in matches:
            self.remove_patient(match[0])
            self.remove_patient(match[1])

    def max_matches(self):
        """
        Maximal matching algorithm taken from Dartmouth CS:
        https://www.cs.dartmouth.edu/~ac/Teach/CS105-Winter05/Notes/kavathekar-scribe.pdf
        """
        print("Hello!")
        return []

    def pass_time(self):
        """
        For each node, calculate chance of departure.
        If departed, remove patient and all relevant edges
        """
        for patient in list(self.patients):
            # If the node is lost
            if numpy.random.uniform() < DEPARTURE_CHANCE:
                self.remove_patient(patient)

    def remove_patient(self, patient):
        """
        Given a patient, removes it from the patients list
        and all relevant edges from the edge list
        """
        # Removes patient from set of patients
        self.patients.remove(patient)

        # Fetches ID of removed patient
        removed_id = patient.get_id()

        # Fetches all nodes connected to removed patient
        destinations = self.edges[removed_id]

        # Removes edges in one direction
        for dest in destinations:
            self.edges[dest].remove(removed_id)

        # Removes edges in other direction
        del self.edges[removed_id]


def main():
    weekly = Exchange()
    monthly = Exchange()

    weekly.add_patients(START_SIZE)
    monthly.add_patients(START_SIZE)

    print([pat.get_id() for pat in weekly.get_patients()])
    print([pat.get_prob() for pat in weekly.get_patients()])
    print(weekly.get_edges())


main()
