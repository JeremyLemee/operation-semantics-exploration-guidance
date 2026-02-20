class Sensor:
    def __init__(self):
        self.gathering = False
        self.percepts = []
        self.waiting_percepts = []

    def add_percept(self, percept):
        if not self.gathering:
            self.percepts.append(percept)
        else:
            self.waiting_percepts.append(percept)

    def gather(self):
        self.gathering = True
        prompt = "".join(self.percepts)
        self.percepts = []
        self.gathering = False
        self.percepts = self.waiting_percepts
        self.waiting_percepts = []
        return prompt
