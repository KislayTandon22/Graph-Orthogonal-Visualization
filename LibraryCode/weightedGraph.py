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
            if self.first_edge_points and self.first_edge_points[0] == key:
                remaining_vertices = [v for v in self.vertices if v != key]
                if remaining_vertices:
                    self.first_edge_points = (remaining_vertices[0], self.first_edge_points[1])

            del self.vertices[key]
            for vertex in self.vertices.values():
                if key in vertex.edges:
                    del vertex.edges[key]
            
            if self.first_edge_points and self.first_edge_points[1] == key:
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
            self.vertices[vertex_key].set_coords(x, y)  # Store coordinates in the vertex object
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
        clockwise_rules = {'C': [2, 0, 1, 3], 'R': [1, 3, 2, 0]}
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
        weights = []
        
        def dfs(vertex_key):
            if vertex_key in visited:
                return
            visited.add(vertex_key)
            angle_type = self.vertices[vertex_key].angle_type
            path.append(angle_type)
            
            for end_key, edge in self.vertices[vertex_key].edges.items():
                if end_key not in visited:
                    weights.append(str(edge.weight) if edge.weight > 1 else '')
                    dfs(end_key)
        
        if self.first_edge_points:
            dfs(self.first_edge_points[0])
        elif self.vertices:
            dfs(next(iter(self.vertices)))
        
        combined_path = []
        for i, (angle_type, weight) in enumerate(zip(path, weights + [''])):
            combined_path.append(angle_type)
            if weight:
                combined_path.append(weight)
        
        return ''.join(combined_path)

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
            parsed_vertices_info = []
            i = 0
            while i < len(vertices_info):
                angle_type = vertices_info[i]
                
                weight = 1
                if i + 1 < len(vertices_info) and vertices_info[i+1].isdigit():
                    j = i + 1
                    while j < len(vertices_info) and vertices_info[j].isdigit():
                        j += 1
                    weight = int(vertices_info[i+1:j])
                    i = j
                else:
                    i += 1
                
                label = generate_label(len(parsed_vertices_info))
                parsed_vertices_info.append((label, angle_type, weight))
            
            vertices_info = parsed_vertices_info

        clockwise_sequence = ['N', 'E', 'S', 'W']
        anticlockwise_sequence = ['N', 'W', 'S', 'E']
        rotation_sequence = clockwise_sequence if rotation == 'Clockwise' else anticlockwise_sequence

        for vertex_info in vertices_info:
            label, angle_type = vertex_info[:2]
            self.add_vertex(label, angle_type)

        if len(vertices_info) > 0:
            first_vertex = vertices_info[0]
            second_vertex = vertices_info[1] if len(vertices_info) > 1 else None
            self.add_edge(first_vertex[0], second_vertex[0], weight=first_vertex[2], direction=default_direction)

        previous_direction = default_direction
        for i in range(len(vertices_info) - 1):
            current_vertex = vertices_info[i]
            next_vertex = vertices_info[i + 1]
            current_label = current_vertex[0]
            current_angle_type = current_vertex[1]

            if current_angle_type == 'F':
                edge_direction = previous_direction
            else:
                rotation_index = rotation_sequence.index(previous_direction)
                if current_angle_type == 'R':
                    edge_direction = rotation_sequence[(rotation_index + 1) % 4]
                elif current_angle_type == 'C':
                    edge_direction = rotation_sequence[(rotation_index - 1) % 4]

            self.add_edge(current_label, next_vertex[0], weight=next_vertex[2], direction=edge_direction)
            previous_direction = edge_direction

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
            
            self.add_edge(last_vertex[0], first_vertex[0], weight=first_vertex[2], direction=edge_direction)

        return self.detect_rotation_direction()

    def find_path(self, u, v):
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
            degree = len(vertex.edges)
            for other_vertex in self.vertices.values():
                for end_key in other_vertex.edges:
                    if end_key == vertex_key:
                        degree += 1
            
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
    


def reduction_1(wg):
    paths = list_all_paths_with_F_angle(wg)

    for path in paths:
        if len(path) < 3:
            continue
        
        start = path[0]
        end = path[-1]
        
        total_weight = sum(wg.vertices[path[i]].edges[path[i + 1]].weight for i in range(len(path) - 1))
        
        direction = wg.vertices[path[0]].edges[path[1]].direction if path[1] in wg.vertices[path[0]].edges else None
        wg.add_edge(start, end, total_weight, direction)

        for vertex in path[1:-1]:
            wg.remove_vertex(vertex)

def list_all_paths_with_F_angle(wg):
    def find_paths(start, end, path=None, paths=None):
        if path is None:
            path = []
        if paths is None:
            paths = []
        path = path + [start]
        if start == end:
            if len(path) >= 3:
                if all(wg.vertices[v].angle_type == 'F' for v in path[1:-1]):
                    paths.append(path)
            return paths
        for neighbor in wg.vertices[start].edges.keys():
            if neighbor not in path:
                find_paths(neighbor, end, path, paths)
        return paths

    all_paths = []
    for start in wg.vertices.keys():
        if wg.vertices[start].angle_type != 'F':
            for end in wg.vertices.keys():
                if start != end and wg.vertices[end].angle_type != 'F':
                    paths = find_paths(start, end)
                    all_paths.extend(paths)
    return all_paths

def find_paths_with_rcr(graph):
    def dfs(current_vertex, path):
        if len(path) == 5:
            if (len(path) == 5 and
                graph.vertices[path[1]].angle_type == 'R' and
                graph.vertices[path[2]].angle_type == 'C' and
                graph.vertices[path[3]].angle_type == 'R'):
                paths.append(path[:])
            return
        
        if current_vertex in graph.vertices:
            for neighbor in graph.vertices[current_vertex].edges:
                path.append(neighbor)
                dfs(neighbor, path)
                path.pop()

    paths = []
    for vertex in graph.vertices:
        dfs(vertex, [vertex])
    return paths

def rename_middle_and_update_neighbors(graph, path):
    
    middle_index = 2  
    middle_vertex = path[middle_index]
    new_label = f"{middle_vertex}'"
    graph.update_vertex_angle_type(path[middle_index - 1], 'F')
    graph.update_vertex_angle_type(path[middle_index], 'R')
    graph.update_vertex_angle_type(path[middle_index + 1], 'F')
    graph.vertices[middle_vertex].label = new_label

def reduction_2(graph):
    paths = find_paths_with_rcr(graph)

    
    for path in paths:
        if (
            graph.vertices[path[1]].angle_type == 'R' and
            graph.vertices[path[2]].angle_type == 'C' and
            graph.vertices[path[3]].angle_type == 'R'
        ):
            rename_middle_and_update_neighbors(graph, path)
            middle_index = 2
            u2 = path[middle_index - 1]
            u3 = path[middle_index]
            u4 = path[middle_index + 1]
            edge_u2_u3 = graph.get_edge(u2, u3)
            edge_u3_u4 = graph.get_edge(u3, u4)
            
            # Remove the edges between u2 -> u3 and u3 -> u4
            del graph.vertices[u2].edges[u3]
            del graph.vertices[u3].edges[u4]

            

            # Add the new edges
            graph.add_edge(
                u2, u3,
                weight=edge_u3_u4.weight,
                direction=edge_u3_u4.direction
            )
            graph.add_edge(
                u3, u4,
                weight=edge_u2_u3.weight,
                direction=edge_u2_u3.direction
            )

    
def find_paths_with_rccc(graph):
    def dfs(current_vertex, path):
        if len(path) == 7:
            if (len(path) == 7 and
                graph.vertices[path[2]].angle_type == 'R' and
                graph.vertices[path[3]].angle_type == 'C' and
                graph.vertices[path[4]].angle_type == 'C' and
                graph.vertices[path[5]].angle_type == 'C'):
                paths.append(path[:])
            return
        
        if current_vertex in graph.vertices:
            for neighbor in graph.vertices[current_vertex].edges:
                path.append(neighbor)
                dfs(neighbor, path)
                path.pop()

    paths = []
    for vertex in graph.vertices:
        dfs(vertex, [vertex])
    return paths

def reduction_3(graph):
    paths = find_paths_with_rccc(graph)
    for path in paths:
        if (all(graph.vertices[path[i]].edges.get(path[i + 1]) is not None for i in range(len(path) - 1)) and
            graph.vertices[path[2]].angle_type == 'R' and
            graph.vertices[path[3]].angle_type == 'C' and
            graph.vertices[path[4]].angle_type == 'C' and
            graph.vertices[path[5]].angle_type == 'C'):
            
            direction_u3_u4 = graph.get_edge(path[3], path[4]).direction
            weight_u3_u4 = graph.get_edge(path[3], path[4]).weight
            direction_u4_u5 = graph.get_edge(path[4], path[5]).direction
            weight_u4_u5 = graph.get_edge(path[4], path[5]).weight
            weight_u2_u3 = graph.get_edge(path[2], path[3]).weight

            graph.vertices[path[2]].angle_type = 'F'

            graph.remove_vertex(path[3])
            graph.remove_vertex(path[4])
            new_label = path[4] + "'"
            graph.add_vertex(new_label, 'C')
            graph.add_edge(new_label, path[5], max(1, weight_u4_u5 - weight_u2_u3), direction_u4_u5)
            graph.add_edge(path[2], new_label, weight_u3_u4, direction_u3_u4)
            graph.get_edge(path[0], path[1]).weight = max(weight_u2_u3, graph.get_edge(path[0], path[1]).weight)

def find_paths_with_rccr(graph):
    def dfs(current_vertex, path):
        if len(path) == 7:
            if (len(path) == 7 and
                graph.vertices[path[2]].angle_type == 'R' and
                graph.vertices[path[3]].angle_type == 'C' and
                graph.vertices[path[4]].angle_type == 'C' and
                graph.vertices[path[5]].angle_type == 'R'):
                paths.append(path[:])
            return
        
        if current_vertex in graph.vertices:
            for neighbor in graph.vertices[current_vertex].edges:
                path.append(neighbor)
                dfs(neighbor, path)
                path.pop()

    paths = []
    for vertex in graph.vertices:
        dfs(vertex, [vertex])
    return paths

def reduction_4(graph):
    paths = find_paths_with_rccr(graph)
    for path in paths:
        if (all(graph.vertices[path[i]].edges.get(path[i + 1]) is not None for i in range(len(path) - 1)) and
            graph.vertices[path[2]].angle_type == 'R' and
            graph.vertices[path[3]].angle_type == 'C' and
            graph.vertices[path[4]].angle_type == 'C' and
            graph.vertices[path[5]].angle_type == 'R'):
            
            direction_u3_u4 = graph.get_edge(path[4], path[5]).direction
            weight_u3_u4 = graph.get_edge(path[4], path[5]).weight
            weight_u2_u3 = graph.get_edge(path[3], path[4]).weight

            graph.remove_vertex(path[4])
            graph.remove_vertex(path[5])

            graph.add_edge(path[3], path[6], weight_u3_u4, direction_u3_u4)
            graph.get_edge(path[0], path[1]).weight = max(graph.get_edge(path[0], path[1]).weight, weight_u2_u3)
            graph.update_vertex_angle_type(path[3], 'F')
            graph.update_vertex_angle_type(path[6], 'F')

def find_paths_with_crc(graph):
    def dfs(current_vertex, path):
        if len(path) == 5:
            paths.append(path[:])
            return

        if current_vertex in graph.vertices:
            for neighbor in graph.vertices[current_vertex].edges:
                path.append(neighbor)
                dfs(neighbor, path)
                path.pop()

    paths = []
    for vertex in graph.vertices:
        dfs(vertex, [vertex])
    return paths



def reduction_5(graph):
    paths = find_paths_with_crc(graph)
    
    for path in paths:
        if (all(node in graph.vertices for node in path) and
            graph.vertices[path[1]].angle_type == 'C' and
            graph.vertices[path[2]].angle_type == 'R' and
            graph.vertices[path[3]].angle_type == 'C'):

            u2 = path[1]
            u3 = path[2]
            u4 = path[3]
            weight_u2_u3=graph.vertices[u2].edges[u3].weight
            direction_u2_u3=graph.vertices[u2].edges[u3].direction
            weight_u3_u4=graph.vertices[u3].edges[u4].weight
            direction_u3_u4=graph.vertices[u3].edges[u4].direction
            # Create a new vertex u3' and add it to the graph
            u3_prime = f"{u3}'"
            graph.add_vertex(u3_prime, 'C')  # Assuming 'C' is the appropriate angle type for u3'
            graph.remove_vertex(u3)
            # Update angle types
            graph.update_vertex_angle_type(u2, 'F')
            
            graph.update_vertex_angle_type(u4, 'F')
            
            
            # Add new edges with the new vertex u3'
            graph.add_edge(u2, u3_prime, weight=weight_u3_u4, direction=direction_u3_u4)
            graph.add_edge(u3_prime, u4, weight=weight_u2_u3, direction=direction_u2_u3)

           

def find_paths_with_cccr(graph):
        def dfs(current_vertex, path):
            if len(path) == 6:
                if (len(path) == 6 and
                    graph.vertices[path[1]].angle_type == 'C' and
                    graph.vertices[path[2]].angle_type == 'C' and
                    graph.vertices[path[3]].angle_type == 'C' and
                    graph.vertices[path[4]].angle_type == 'R'):
                    paths.append(path[:])
                return

            if current_vertex in graph.vertices:
                for neighbor in graph.vertices[current_vertex].edges:
                    path.append(neighbor)
                    dfs(neighbor, path)
                    path.pop()

        paths = []
        for vertex in graph.vertices:
            dfs(vertex, [vertex])
        return paths
def reduction_6(graph):
    

    paths = find_paths_with_cccr(graph)
    for path in paths:
        if (all(graph.vertices[path[i]].edges.get(path[i + 1]) is not None for i in range(len(path) - 1)) and
            graph.vertices[path[1]].angle_type == 'C' and
            graph.vertices[path[2]].angle_type == 'C' and
            graph.vertices[path[3]].angle_type == 'C' and
            graph.vertices[path[4]].angle_type == 'R'):
            
            direction_u3_u4 = graph.get_edge(path[2], path[3]).direction
            weight_u3_u4 = graph.get_edge(path[2], path[3]).weight
            direction_u4_u5 = graph.get_edge(path[3], path[4]).direction
            weight_u4_u5 = graph.get_edge(path[3], path[4]).weight
            weight_u2_u3 = graph.get_edge(path[1], path[2]).weight

            graph.vertices[path[1]].angle_type = 'C'
            graph.vertices[path[4]].angle_type = 'F'
            new_label = path[2] + "'"
            graph.remove_vertex(path[2])
            graph.remove_vertex(path[3])
            
            graph.add_vertex(new_label, 'C')
            graph.add_edge(path[1], new_label, max(weight_u2_u3-weight_u4_u5,1), direction_u3_u4)
            graph.add_edge(new_label, path[4], weight_u3_u4, direction_u3_u4)
            graph.get_edge(path[4],path[5]).weight = max(graph.get_edge(path[0],path[1]).weight,weight_u3_u4,)

def find_paths_with_crrr(graph):
    def dfs(current_vertex, path):
        if len(path) == 7:
            if (len(path) == 7 and
                graph.vertices[path[2]].angle_type == 'C' and
                graph.vertices[path[3]].angle_type == 'R' and
                graph.vertices[path[4]].angle_type == 'R' and
                graph.vertices[path[5]].angle_type == 'R'):
                paths.append(path[:])
            return
        
        if current_vertex in graph.vertices:
            for neighbor in graph.vertices[current_vertex].edges:
                path.append(neighbor)
                dfs(neighbor, path)
                path.pop()

    paths = []
    for vertex in graph.vertices:
        dfs(vertex, [vertex])
    return paths

def reduction_7(graph):
    paths = find_paths_with_crrr(graph)
    
    for path in paths:
        if (all(graph.vertices[path[i]].edges.get(path[i + 1]) is not None for i in range(len(path) - 1)) and
            graph.vertices[path[2]].angle_type == 'C' and
            graph.vertices[path[3]].angle_type == 'R' and
            graph.vertices[path[4]].angle_type == 'R' and
            graph.vertices[path[5]].angle_type == 'R'):
            
            direction_u3_u4 = graph.get_edge(path[3], path[4]).direction
            weight_u3_u4 = graph.get_edge(path[3], path[4]).weight
            direction_u4_u5 = graph.get_edge(path[4], path[5]).direction
            weight_u4_u5 = graph.get_edge(path[4], path[5]).weight
            weight_u2_u3 = graph.get_edge(path[2], path[3]).weight

            graph.vertices[path[2]].angle_type = 'F'

            graph.remove_vertex(path[3])
            graph.remove_vertex(path[4])
            new_label = path[3] + "'"
            graph.add_vertex(new_label, 'r')
            graph.add_edge(new_label, path[5], max(1, weight_u4_u5 - weight_u2_u3), direction_u4_u5)
            graph.add_edge(path[2], new_label, weight_u3_u4, direction_u3_u4)
            graph.get_edge(path[0], path[1]).weight = max(weight_u2_u3, graph.get_edge(path[0], path[1]).weight)

def find_paths_with_rrrc(graph):
        def dfs(current_vertex, path):
            if len(path) == 6:
                if (len(path) == 6 and
                    graph.vertices[path[1]].angle_type == 'R' and
                    graph.vertices[path[2]].angle_type == 'R' and
                    graph.vertices[path[3]].angle_type == 'R' and
                    graph.vertices[path[4]].angle_type == 'C'):
                    paths.append(path[:])
                return

            if current_vertex in graph.vertices:
                for neighbor in graph.vertices[current_vertex].edges:
                    path.append(neighbor)
                    dfs(neighbor, path)
                    path.pop()

        paths = []
        for vertex in graph.vertices:
            dfs(vertex, [vertex])
        return paths
def reduction_8(graph):
    

    paths = find_paths_with_rrrc(graph)
    for path in paths:
        if (all(graph.vertices[path[i]].edges.get(path[i + 1]) is not None for i in range(len(path) - 1)) and
            graph.vertices[path[1]].angle_type == 'R' and
            graph.vertices[path[2]].angle_type == 'R' and
            graph.vertices[path[3]].angle_type == 'R' and
            graph.vertices[path[4]].angle_type == 'C'):
            
            direction_u3_u4 = graph.get_edge(path[2], path[3]).direction
            weight_u3_u4 = graph.get_edge(path[2], path[3]).weight
            direction_u4_u5 = graph.get_edge(path[3], path[4]).direction
            weight_u4_u5 = graph.get_edge(path[3], path[4]).weight
            weight_u2_u3 = graph.get_edge(path[1], path[2]).weight

            graph.vertices[path[1]].angle_type = 'C'
            graph.vertices[path[4]].angle_type = 'F'
            new_label = path[3] + "'"
            graph.remove_vertex(path[2])
            graph.remove_vertex(path[3])
            
            graph.add_vertex(new_label, 'R')
            graph.add_edge(path[1], new_label, max(weight_u2_u3-weight_u4_u5,1), direction_u3_u4)
            graph.add_edge(new_label, path[4], weight_u3_u4, direction_u3_u4)
            graph.get_edge(path[4],path[5]).weight = max(graph.get_edge(path[0],path[1]).weight,weight_u3_u4,)

def find_paths_with_crrc(graph):
    def dfs(current_vertex, path):
        if len(path) == 7:
            if (len(path) == 7 and
                graph.vertices[path[2]].angle_type == 'C' and
                graph.vertices[path[3]].angle_type == 'R' and
                graph.vertices[path[4]].angle_type == 'R' and
                graph.vertices[path[5]].angle_type == 'C'):
                paths.append(path[:])
            return
        
        if current_vertex in graph.vertices:
            for neighbor in graph.vertices[current_vertex].edges:
                path.append(neighbor)
                dfs(neighbor, path)
                path.pop()

    paths = []
    for vertex in graph.vertices:
        dfs(vertex, [vertex])
    return paths

def reduction_9(graph):
    paths = find_paths_with_crrc(graph)
    print(paths)
    for path in paths:
        
        if (all(graph.vertices[path[i]].edges.get(path[i + 1]) is not None for i in range(len(path) - 1)) and
            graph.vertices[path[2]].angle_type == 'C' and
            graph.vertices[path[3]].angle_type == 'R' and
            graph.vertices[path[4]].angle_type == 'R' and
            graph.vertices[path[5]].angle_type == 'C'):
            
            direction_u2_u3 = graph.get_edge(path[2], path[3]).direction
            weight_u1_u2 = graph.get_edge(path[1], path[2]).weight
            weight_u2_u3 = graph.get_edge(path[2], path[3]).weight

            graph.remove_vertex(path[2])
            graph.remove_vertex(path[3])
            

            graph.add_edge(path[1], path[4], weight_u2_u3, direction_u2_u3)
            graph.get_edge(path[0], path[1]).weight = max(graph.get_edge(path[0], path[1]).weight, weight_u1_u2)
            graph.update_vertex_angle_type(path[1], 'F')
            graph.update_vertex_angle_type(path[4], 'F')

