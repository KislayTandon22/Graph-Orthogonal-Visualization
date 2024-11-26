class Edge:
    def __init__(self, start, end, weight=1, direction=None):
        self.start = start
        self.end = end
        self.weight = weight
        self.length = weight
        self.direction = direction