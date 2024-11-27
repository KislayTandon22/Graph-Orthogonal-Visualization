import streamlit as st
import matplotlib.pyplot as plt
import io
import base64
from weightedGraph import WeightedGraph
from edges import Edge
from weightedGraph import (
    reduction_1, reduction_2, reduction_3, reduction_4,
    reduction_5, reduction_6, reduction_7, reduction_8, reduction_9
)

class AutoGraphBuilder:
    def __init__(self):
        self.graph = WeightedGraph()
        self.rotation = 'Anticlockwise'
        self.last_direction = None  # Tracks the last edge direction
        self.is_closed = False      # Tracks if the graph is closed

    def create_graph_from_string(self, graph_string):
        self.graph = WeightedGraph()
        try:
            detected_rotation = self.graph.create_graph_with_rotation(graph_string, rotation=self.rotation)
            self.rotation = detected_rotation
            self.is_closed = True
            return True
        except Exception as e:
            st.error(f"Error creating graph: {e}")
            return False

    def generate_graph_image(self):
        plt.figure(figsize=(10, 10))
        plt.clf()
        try:
            self.graph.draw_graph()
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            plot_url = base64.b64encode(buf.getbuffer()).decode('ascii')
            plt.close()
            return plot_url
        except Exception as e:
            plt.close()
            return None


def main():
    st.set_page_config(layout="wide")
    st.title('Directional Graph Builder and Reducer')

    if 'graph_builder' not in st.session_state:
        st.session_state.graph_builder = AutoGraphBuilder()

    layout = st.columns([1.5, 2, 1.5])  # Left, Middle, Right

    with layout[0]:
        st.header('Input and Operations')
        tabs = st.radio("Choose Input Type", ["String Input", "Draw"], index=0)

        if tabs == "String Input":
            graph_string = st.text_input('Enter graph pattern (R: Reflex, C: Convex, F: Fixed)', placeholder='e.g., RCRF')
            rotation = st.radio('Select Rotation Direction', ['Anticlockwise', 'Clockwise'], index=0)
            if st.button('Create Graph'):
                if graph_string:
                    st.session_state.graph_builder.rotation = rotation
                    if st.session_state.graph_builder.create_graph_from_string(graph_string):
                        st.success(f'Graph created from pattern: {graph_string} (Rotation: {rotation})')
                else:
                    st.warning('Please enter a valid pattern using R, C, and F')

        elif tabs == "Draw":
            edge_length = st.number_input('Edge Length', min_value=0.1, value=1.0, step=0.1)

            directions = st.columns(4)
            if directions[0].button('North (N)'):
                add_edge_to_graph('N', edge_length)
            if directions[1].button('South (S)'):
                add_edge_to_graph('S', edge_length)
            if directions[2].button('East (E)'):
                add_edge_to_graph('E', edge_length)
            if directions[3].button('West (W)'):
                add_edge_to_graph('W', edge_length)

            if st.button("Close Graph"):
                close_graph(st.session_state.graph_builder.graph)
                # Set the flag indicating that the graph is closed
                st.session_state.graph_builder.is_closed = True
                st.success("Graph has been closed successfully.")

        st.header('Reduction Operations')
        columns_per_row = 3
        reductions = [
            ('Reduction 1', reduction_1),
            ('Reduction 2', reduction_2),
            ('Reduction 3', reduction_3),
            ('Reduction 4', reduction_4),
            ('Reduction 5', reduction_5),
            ('Reduction 6', reduction_6),
            ('Reduction 7', reduction_7),
            ('Reduction 8', reduction_8),
            ('Reduction 9', reduction_9),
        ]
        for i in range(0, len(reductions), columns_per_row):
            cols = st.columns(columns_per_row)
            for col, (name, func) in zip(cols, reductions[i:i + columns_per_row]):
                if col.button(name):
                    try:
                        func(st.session_state.graph_builder.graph)
                        st.success(f'{name} applied successfully.')
                    except Exception as e:
                        st.error(f'Error applying {name}: {e}')

    with layout[1]:
        st.header('Graph Visualization')
        try:
            plot_url = st.session_state.graph_builder.generate_graph_image()
            if plot_url:
                st.image(base64.b64decode(plot_url), use_column_width=True)
            else:
                st.info("Create a graph to visualize")
        except Exception as e:
            st.error(f"Error generating graph: {e}")

    with layout[2]:
        st.header('Graph Information')
        if st.button('Detect Rotation'):
            try:
                rotation = st.session_state.graph_builder.graph.detect_rotation_direction()
                st.write(f"**Rotation Direction:** {rotation}")
            except Exception as e:
                st.error(f"Error detecting rotation: {e}")

        if st.button('Get Graph String'):
            try:
                graph_string = st.session_state.graph_builder.graph.get_graph_string()
                st.write(f"**Graph String:** {graph_string}")
            except Exception as e:
                st.error(f"Error getting graph string: {e}")

        if st.button('Find Kitty Corners'):
            try:
                kitty_corners = st.session_state.graph_builder.graph.find_kitty_corners_first()
                st.write(f"**Kitty Corners:** {kitty_corners}")
            except Exception as e:
                st.error(f"Error finding kitty corners: {e}")

        if st.button('Calculate Rotation Value'):
            try:
                rotation_value = st.session_state.graph_builder.graph.calculate_rotation_value()
                st.write(f"**Rotation Value:** {rotation_value}")
            except Exception as e:
                st.error(f"Error calculating rotation value: {e}")

        st.header('Graph Details')
        import sys
        from io import StringIO
        old_stdout = sys.stdout
        result = StringIO()
        sys.stdout = result
        try:
            st.session_state.graph_builder.graph.print_graph()
            sys.stdout = old_stdout
            graph_details = result.getvalue()
            if graph_details.strip():
                st.text(graph_details)
            else:
                st.info("No graph details available")
        except Exception:
            sys.stdout = old_stdout
            st.info("No graph details available")


def add_edge_to_graph(direction, edge_length):
    builder = st.session_state.graph_builder
    graph = builder.graph

    if len(graph.vertices) == 0:
        # Add first two vertices and initial edge
        graph.add_vertex('A', 'F')  # Default first point is 'Fixed' until closure
        graph.add_vertex('B', 'F')
        graph.add_edge('A', 'B', weight=edge_length, direction=direction)
        builder.last_direction = direction
        st.success("Initial edge added between A and B")
    else:
        last_vertex = max(graph.vertices.keys())
        next_vertex = chr(ord(last_vertex) + 1)
        angle = compute_angle(builder.last_direction, direction)
        graph.add_vertex(next_vertex, 'R' if angle == 270 else 'C')
        graph.add_edge(last_vertex, next_vertex, weight=edge_length, direction=direction)
        builder.last_direction = direction
        st.success(f"Edge added between {last_vertex} and {next_vertex} in direction {direction}")
    if builder.is_closed:
        close_graph(graph)

def compute_angle(prev_direction, curr_direction):
    # Map directions to angles
    direction_angles = {'N': 90, 'E': 0, 'S': 270, 'W': 180}
    prev_angle = direction_angles.get(prev_direction, 0)
    curr_angle = direction_angles.get(curr_direction, 0)
    angle_diff = (curr_angle - prev_angle) % 360
    return angle_diff if angle_diff != 0 else 360  # Reflexively handle zero-difference as full circle


def close_graph(graph):
    # Ensure that the graph has vertices
    if not graph.vertices:
        print("Error: Graph has no vertices.")
        return

    # Get the first and last vertex in the graph
    first_vertex = list(graph.vertices.keys())[0]
    last_vertex = list(graph.vertices.keys())[-1]
    
    # Access the coordinates of the first and last vertex
    first_vertex_coords = graph.vertices[first_vertex].coords
    last_vertex_coords = graph.vertices[last_vertex].coords

    # Check if the first and last vertex have the same x or y coordinate
    if first_vertex_coords[0] != last_vertex_coords[0] and first_vertex_coords[1] != last_vertex_coords[1]:
        print(f"Warning: First vertex ({first_vertex_coords}) and last vertex ({last_vertex_coords}) do not align on the same axis.")
        return

    # Calculate the absolute distance between the first and last vertex
    x_dist = abs(first_vertex_coords[0] - last_vertex_coords[0])
    y_dist = abs(first_vertex_coords[1] - last_vertex_coords[1])

    # Add a final edge that connects the last vertex to the first vertex
    if x_dist > 0:
        direction = 'E' if first_vertex_coords[0] < last_vertex_coords[0] else 'W'
    elif y_dist > 0:
        direction = 'N' if first_vertex_coords[1] < last_vertex_coords[1] else 'S'
    else:
        print("Error: No valid direction to connect the first and last vertices.")
        return

    # The edge weight will be the maximum distance along the x or y axis
    weight = max(x_dist, y_dist)

    # Add the edge connecting the last vertex to the first vertex
    graph.add_edge(last_vertex, first_vertex, weight=weight, direction=direction)

    # Optional: Update the angle types of the first and last vertices
    graph.update_vertex_angle_type(first_vertex, 'F')
    graph.update_vertex_angle_type(last_vertex, 'F')

    # Ensure the graph is closed by checking that all edges align properly (Optional)
    print(f"Graph closed with an edge from {last_vertex} to {first_vertex}.")
    print(f"Edge weight: {weight}, Direction: {direction}")





if __name__ == '__main__':
    main()
