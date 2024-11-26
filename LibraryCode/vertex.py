class Vertex:
    def __init__(self, label, angle_type='F'):
        self.label = label
        self.angle_type = angle_type
        self.edges = {}