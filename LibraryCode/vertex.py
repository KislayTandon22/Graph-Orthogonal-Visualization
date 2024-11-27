class Vertex:
    def __init__(self, label, angle_type='F'):
        self.label = label
        self.angle_type = angle_type
        self.edges = {}
        self.coords = None  # Add coords attribute to store coordinates

    def set_coords(self, x, y):
        self.coords = (x, y)