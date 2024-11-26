import matplotlib.pyplot as plt
import networkx as nx
from edges import Edge
from vertex import Vertex
class WeightedGraph:
    def __init__(self):
        self.vertices = {}
        self.first_edge_points = None
        
    def add_vertex(self, key, angle_type='F'):
        if key not in self.vertices:
            self.vertices[key] = Vertex(key, angle_type)

    def add_edge(self, start, end, weight=1, direction=None):
        if start in self.vertices and end in self.vertices:
            edge = Edge(start, end, weight, direction)
            self.vertices[start].edges[end] = edge
            if self.first_edge_points is None:
                self.first_edge_points = (start, end)

    def update_vertex_angle_type(self, vertex_key, new_angle_type):
        if vertex_key in self.vertices:
            self.vertices[vertex_key].angle_type = new_angle_type

    def update_edge(self, start, end, new_length=None, new_direction=None):
        if start in self.vertices and end in self.vertices[start].edges:
            edge = self.vertices[start].edges[end]
            if new_length is not None:
                edge.length = new_length
            if new_direction is not None:
                edge.direction = new_direction

    def change_edge_length(self, start, end, new_length):
        self.update_edge(start, end, new_length=new_length)

    def remove_vertex(self, key):
        if key in self.vertices:
            # Check if the vertex to be removed is the first vertex
            if self.first_edge_points and self.first_edge_points[0] == key:
                # Find the first existing vertex in the graph (not removed)
                remaining_vertices = [v for v in self.vertices if v != key]
                if remaining_vertices:
                    # Update first_edge_points to point to the first remaining vertex
                    self.first_edge_points = (remaining_vertices[0], self.first_edge_points[1])

            # Remove the vertex and its edges
            del self.vertices[key]
            for vertex in self.vertices.values():
                # Remove the removed vertex from other vertices' edges
                if key in vertex.edges:
                    del vertex.edges[key]
            
            # Ensure the first_edge_points points to a valid remaining edge if the second vertex is removed
            if self.first_edge_points and self.first_edge_points[1] == key:
                # Find the next valid vertex to update the second part of first_edge_points
                remaining_vertices = [v for v in self.vertices if v != self.first_edge_points[0]]
                if remaining_vertices:
                    self.first_edge_points = (self.first_edge_points[0], remaining_vertices[0])



    def get_edge(self, start, end):
        if start in self.vertices and end in self.vertices[start].edges:
            return self.vertices[start].edges[end]
        return None

    def calculate_positions(self, start_key=None):
        if start_key is None and self.first_edge_points:
            start_key = self.first_edge_points[0]
        pos = {start_key: (0, 0)}

        def place_vertex(vertex_key, x, y):
            pos[vertex_key] = (x, y)
            for end_key, edge in self.vertices[vertex_key].edges.items():
                if end_key in self.vertices and end_key not in pos:
                    if edge.direction == 'E':
                        place_vertex(end_key, x + max(edge.length, edge.weight), y)
                    elif edge.direction == 'S':
                        place_vertex(end_key, x, y - max(edge.length, edge.weight))
                    elif edge.direction == 'W':
                        place_vertex(end_key, x - max(edge.length, edge.weight), y)
                    elif edge.direction == 'N':
                        place_vertex(end_key, x, y + max(edge.length, edge.weight))
            if len(pos) == len(self.vertices):
                x_last, y_last = pos[vertex_key]
                if vertex_key != start_key:
                    for end_key, edge in self.vertices[vertex_key].edges.items():
                        if end_key in pos:
                            if edge.direction in ['N', 'S'] and x_last != pos[end_key][0]:
                                pos[vertex_key] = (pos[end_key][0], y_last)
                            elif edge.direction in ['E', 'W'] and y_last != pos[end_key][1]:
                                pos[vertex_key] = (x_last, pos[end_key][1])

        if start_key in self.vertices:
            place_vertex(start_key, 0, 0)
        return pos


    def draw_edges(self, pos):
        for vertex_key in self.vertices:
            if vertex_key not in pos:
                continue
            for end_key, edge in self.vertices[vertex_key].edges.items():
                if end_key in pos:
                    x_start, y_start = pos[vertex_key]
                    x_end, y_end = pos[end_key]
                    plt.plot([x_start, x_end], [y_start, y_end], 'k-')

    def print_graph(self):
        for key, vertex in self.vertices.items():
            edges_info = ', '.join([f"{end} (weight: {edge.weight}, {edge.direction})" for end, edge in vertex.edges.items()])
            print(f"Vertex: {key}, Angle Type: {vertex.angle_type}, Edges: {edges_info if edges_info else 'No edges'}")

    def draw_graph(self):
        pos = self.calculate_positions()
        x_values = [pos[key][0] for key in pos]
        y_values = [pos[key][1] for key in pos]
        min_x, max_x = min(x_values), max(x_values)
        min_y, max_y = min(y_values), max(y_values)
        mid_x = (min_x + max_x) / 2
        mid_y = (min_y + max_y) / 2
        width = max_x - min_x + 2
        height = max_y - min_y + 2

        plt.figure(figsize=(8, 8))
        self.draw_edges(pos)

        for key, vertex in self.vertices.items():
            if key in pos:
                plt.scatter(*pos[key], s=700, c='skyblue')
                plt.text(pos[key][0], pos[key][1], f"{vertex.label} ({vertex.angle_type})", fontsize=12, ha='center', va='center')

        plt.grid(True)
        plt.xlim(mid_x - width / 2, mid_x + width / 2)
        plt.ylim(mid_y - height / 2, mid_y + height / 2)
        plt.xticks(range(int(mid_x - width / 2), int(mid_x + width / 2) + 1))
        plt.yticks(range(int(mid_y - height / 2), int(mid_y + height / 2) + 1))
        plt.gca().set_aspect('equal', adjustable='box')

        for vertex_key in self.vertices:
            if vertex_key in pos:
                for end_key in self.vertices[vertex_key].edges:
                    if end_key in pos:
                        edge_length = self.vertices[vertex_key].edges[end_key].length
                        plt.text((pos[vertex_key][0] + pos[end_key][0]) / 2, (pos[vertex_key][1] + pos[end_key][1]) / 2, str(edge_length), fontsize=10, ha='center')

        plt.show()

    def detect_rotation_direction(self):
        rotations = {'Clockwise': 0, 'Anticlockwise': 0}
        direction_map = {'N': 0, 'E': 1, 'S': 2, 'W': 3}
        clockwise_rules = {'C': [2, 0, 1, 3], 'R': [1, 3, 2, 0]}  # New direction indices
        anticlockwise_rules = {'C': [1, 3, 2, 0], 'R': [2, 0, 1, 3]}

        for vertex_key, vertex in self.vertices.items():
            incoming_edges = []
            outgoing_edges = []

            for other_vertex, other_edges in self.vertices.items():
                for end, edge in other_edges.edges.items():
                    if end == vertex_key:
                        incoming_edges.append(edge)
                    if other_vertex == vertex_key:
                        outgoing_edges.append(edge)

            for in_edge in incoming_edges:
                for out_edge in outgoing_edges:
                    if in_edge.direction and out_edge.direction:
                        in_dir = direction_map[in_edge.direction]
                        out_dir = direction_map[out_edge.direction]

                        if vertex.angle_type in clockwise_rules:
                            rule = clockwise_rules if vertex.angle_type == 'R' else anticlockwise_rules
                            if out_dir == rule[vertex.angle_type][in_dir]:
                                rotations['Clockwise'] += 1
                            else:
                                rotations['Anticlockwise'] += 1

        return 'Clockwise' if rotations['Clockwise'] > rotations['Anticlockwise'] else 'Anticlockwise'


    def get_graph_string(self):
        visited = set()
        path = []
        
        def dfs(vertex_key):
            if vertex_key in visited:
                return
            visited.add(vertex_key)
            path.append(self.vertices[vertex_key].angle_type)
            for end_key in self.vertices[vertex_key].edges:
                if end_key not in visited:
                    dfs(end_key)
        
        if self.first_edge_points:
            dfs(self.first_edge_points[0])
        elif self.vertices:
            dfs(next(iter(self.vertices)))
            
        return ''.join(path)

    def calculate_rotation_value(self):
        graph_string = self.get_graph_string()
        rotation_value = 0
        for char in graph_string:
            if char == 'R':
                rotation_value += 1
            elif char == 'C':
                rotation_value -= 1
        return rotation_value
    def create_graph_with_rotation(self, vertices_info, rotation='Clockwise', default_direction='N'):
        def generate_label(index):
            def int_to_base26(n):
                result = ""
                while n >= 0:
                    result = chr(65 + (n % 26)) + result
                    n = n // 26 - 1
                return result
            return int_to_base26(index)

        if isinstance(vertices_info, str):
            vertices_info = [(generate_label(i), char if char in ['R', 'C'] else 'F', 1) 
                            for i, char in enumerate(vertices_info)]

        clockwise_sequence = ['N', 'E', 'S', 'W']
        anticlockwise_sequence = ['N', 'W', 'S', 'E']
        rotation_sequence = clockwise_sequence if rotation == 'Clockwise' else anticlockwise_sequence

        # Add all vertices
        for vertex_info in vertices_info:
            label, angle_type = vertex_info[:2]
            self.add_vertex(label, angle_type)

        if len(vertices_info) > 0:
            # Add explicit edge between the first and second points
            first_vertex = vertices_info[0]
            second_vertex = vertices_info[1] if len(vertices_info) > 1 else None
            self.add_edge(first_vertex[0], second_vertex[0], weight=1, direction=default_direction)

        # Add remaining edges and determine directions
        previous_direction = default_direction
        for i in range(len(vertices_info) - 1):
            current_vertex = vertices_info[i]
            next_vertex = vertices_info[i + 1]
            current_label = current_vertex[0]
            current_angle_type = current_vertex[1]

            # Determine direction based on angle type
            if current_angle_type == 'F':
                edge_direction = previous_direction
            else:
                rotation_index = rotation_sequence.index(previous_direction)
                if current_angle_type == 'R':
                    edge_direction = rotation_sequence[(rotation_index + 1) % 4]
                elif current_angle_type == 'C':
                    edge_direction = rotation_sequence[(rotation_index - 1) % 4]

            # Add edge and update direction
            self.add_edge(current_label, next_vertex[0], weight=1, direction=edge_direction)
            previous_direction = edge_direction

        # Connect the last vertex back to the first
        if len(vertices_info) > 1:
            last_vertex = vertices_info[-1]
            first_vertex = vertices_info[0]
            last_angle_type = last_vertex[1]
            if last_angle_type == 'F':
                edge_direction = previous_direction
            else:
                rotation_index = rotation_sequence.index(previous_direction)
                if last_angle_type == 'R':
                    edge_direction = rotation_sequence[(rotation_index + 1) % 4]
                elif last_angle_type == 'C':
                    edge_direction = rotation_sequence[(rotation_index - 1) % 4]
            
            self.add_edge(last_vertex[0], first_vertex[0], weight=1, direction=edge_direction)

        return self.detect_rotation_direction()
    def find_path(self, u, v):
        # Assuming you have a method to find the path from u to v (excluding v itself)
        # This will implement the path-finding logic between u and v
        path = []
        visited = set()
        stack = [u]

        while stack:
            current = stack.pop()
            if current == v:
                break
            if current not in visited:
                visited.add(current)
                path.append(current)
                for neighbor in self.vertices[current].edges:
                    if neighbor not in visited:
                        stack.append(neighbor)

        return path

    def rot(self, u, v):
        path = self.find_path(u, v)
        score = 0
        

        for vertex_key in path:
            vertex = self.vertices[vertex_key]
            degree = len(vertex.edges)  # Outgoing edges
            for other_vertex in self.vertices.values():
                for end_key in other_vertex.edges:
                    if end_key == vertex_key:
                        degree += 1  # Incoming edges
            
            
            if vertex.angle_type == 'C':
                score += 1
            elif vertex.angle_type == 'R' and degree==1:
                score -= 2
            elif vertex.angle_type == 'R' and degree!=1:
                score -= 1

        
        return score


    def find_kitty_corners_first(self):
        kitty_corners = set()

        # Find all reflex vertices
        reflex_vertices = [key for key, vertex in self.vertices.items() if vertex.angle_type == 'R']

        # Check all pairs of reflex vertices
        for i in range(len(reflex_vertices)):
            for j in range(i + 1, len(reflex_vertices)):
                u = reflex_vertices[i]
                v = reflex_vertices[j]

                # Check if u and v are kitty corners
                if self.rot(u, v) == 2 or self.rot(v, u) == 2:
                    kitty_corners.add(u)
                    kitty_corners.add(v)

        return kitty_corners
    


    
    

