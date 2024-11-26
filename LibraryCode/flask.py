import streamlit as st
import matplotlib.pyplot as plt
import io
import base64

# Import the WeightedGraph class
from weightedGraph import WeightedGraph

class AutoGraphBuilder:
    def __init__(self):
        self.graph = WeightedGraph()
        self.rotation = 'Anticlockwise'  # Default rotation changed to Anticlockwise
    
    def create_graph_from_string(self, graph_string):
        """
        Create graph directly using the graph's create_graph_with_rotation method
        
        Args:
        graph_string (str): String of vertex types (R, C, F)
        """
        # Reset the graph
        self.graph = WeightedGraph()
        
        # Detect and set rotation based on the graph string
        try:
            detected_rotation = self.graph.create_graph_with_rotation(graph_string, rotation=self.rotation)
            self.rotation = detected_rotation
            return True
        except Exception as e:
            st.error(f"Error creating graph: {e}")
            return False
    
    def generate_graph_image(self):
        """Generate a base64 encoded graph image."""
        plt.figure(figsize=(10, 10))
        plt.clf()  # Clear the current figure
        
        try:
            # Use the graph's draw method
            self.graph.draw_graph()
            
            # Save plot to buffer
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
    
    st.title('Directional Graph Builder from String')
    
    # Initialize or retrieve the graph from session state
    if 'graph_builder' not in st.session_state:
        st.session_state.graph_builder = AutoGraphBuilder()
    
    # Main content area
    col1, col2 = st.columns(2)
    
    with col1:
        st.header('Graph Input')
        
        # Text input for graph creation
        graph_string = st.text_input(
            'Enter graph pattern (R: Reflex, C: Convex, F: Fixed)', 
            placeholder='e.g., RCRF'
        )
        
        # Rotation selection with Anticlockwise as default
        rotation = st.radio(
            'Select Rotation Direction', 
            ['Anticlockwise', 'Clockwise'],
            index=0  # Set Anticlockwise as default
        )
        
        # Create Graph Button
        if st.button('Create Graph'):
            if graph_string and all(char in 'RCF' for char in graph_string):
                # Update rotation
                st.session_state.graph_builder.rotation = rotation
                
                # Create graph
                if st.session_state.graph_builder.create_graph_from_string(graph_string):
                    st.success(f'Created graph from pattern: {graph_string} (Rotation: {rotation})')
            else:
                st.warning('Please enter a valid pattern using R, C, and F')
        
        # Graph Analysis Section
        st.header('Graph Analysis')
        
        # Rotation Detection
        if st.button('Detect Rotation'):
            rotation = st.session_state.graph_builder.graph.detect_rotation_direction()
            st.write(f"**Rotation Direction:** {rotation}")
        
        # Graph String
        if st.button('Get Graph String'):
            graph_string = st.session_state.graph_builder.graph.get_graph_string()
            st.write(f"**Graph String:** {graph_string}")
        
        # Kitty Corners
        if st.button('Find Kitty Corners'):
            kitty_corners = st.session_state.graph_builder.graph.find_kitty_corners_first()
            st.write(f"**Kitty Corners:** {kitty_corners}")
        
        # Rotation Value
        if st.button('Calculate Rotation Value'):
            rotation_value = st.session_state.graph_builder.graph.calculate_rotation_value()
            st.write(f"**Rotation Value:** {rotation_value}")
    
    with col2:
        # Graph Visualization
        st.header('Graph Visualization')
        
        # Automatically generate and display graph
        try:
            # Generate graph image
            plot_url = st.session_state.graph_builder.generate_graph_image()
            
            if plot_url:
                # Display the graph image
                st.image(base64.b64decode(plot_url), use_column_width=True)
            else:
                st.info("Create a graph to visualize")
        except Exception as e:
            st.error(f"Error generating graph: {e}")
        
        # Graph Details Section
        st.header('Graph Details')
        
        # Capture and display graph details
        import sys
        from io import StringIO
        
        # Redirect stdout to capture print
        old_stdout = sys.stdout
        result = StringIO()
        sys.stdout = result
        
        # Call print_graph method
        st.session_state.graph_builder.graph.print_graph()
        
        # Restore stdout and get output
        sys.stdout = old_stdout
        graph_details = result.getvalue()
        
        # Display graph details
        st.text(graph_details) if graph_details.strip() else st.info("No graph details yet")

if __name__ == '__main__':
    main()