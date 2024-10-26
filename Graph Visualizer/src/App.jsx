// src/App.jsx

import React, { useState, useRef, useEffect } from "react";
import { Network } from "vis-network";
import "vis-network/styles/vis-network.css";
import { DataSet } from "vis-data";

const options = {
  nodes: {
    shape: "dot",
    size: 5,
    color: {
      background: "#0074D9",
      border: "#001f3f",
    },
    font: {
      size: 10,
      color: "#ffffff",
    },
  },
  edges: {
    color: { color: "#AAAAAA", highlight: "#FF4136", hover: "#FF4136" },
    width: 1,
  },
  physics: {
    enabled: true,
    barnesHut: {
      gravitationalConstant: -2000,
      centralGravity: 0.3,
      springLength: 10,
      springConstant: 0.04,
      damping: 0.09,
    },
  },
  interaction: {
    hover: true,
    tooltipDelay: 200,
    hideEdgesOnDrag: true,
  },
};

const App = () => {
  const containerRef = useRef(null);
  const [network, setNetwork] = useState(null);
  const [nodeId, setNodeId] = useState(1);
  const [fromNode, setFromNode] = useState("");
  const [toNode, setToNode] = useState("");
  const nodes = useRef(new DataSet());
  const edges = useRef(new DataSet());

  useEffect(() => {
    const data = {
      nodes: nodes.current,
      edges: edges.current,
    };
    const networkInstance = new Network(containerRef.current, data, options);
    setNetwork(networkInstance);
  }, []);

  const getRandomPosition = () => {
    // Get container dimensions to place nodes within visible bounds
    const container = containerRef.current;
    const width = container ? container.offsetWidth : 800;
    const height = container ? container.offsetHeight : 600;

    // Generate random x, y positions within container bounds
    return {
      x: Math.random() * width - width / 2, // Centering around 0
      y: Math.random() * height - height / 2,
    };
  };

  const addNode = () => {
    const { x, y } = getRandomPosition();
    nodes.current.add({ id: nodeId, label: `${nodeId}`, x, y });
    setNodeId(nodeId + 1);
  };

  const addEdge = () => {
    if (fromNode && toNode) {
      edges.current.add({ from: parseInt(fromNode), to: parseInt(toNode) });
      setFromNode("");
      setToNode("");
    }
  };

  const applyReductionStep = () => {
    console.log("Reduction step applied");
  };

  const zoomIn = () => {
    if (network) {
      const scale = network.getScale();
      network.moveTo({ scale: scale * 1.2 });
    }
  };

  const zoomOut = () => {
    if (network) {
      const scale = network.getScale();
      network.moveTo({ scale: scale * 0.8 });
    }
  };

  return (
    <div style={{ display: "flex", height: "100vh", fontFamily: "Arial, sans-serif" }}>
      <div
        style={{
          flex: 3,
          border: "1px solid lightgray",
          padding: "10px",
          backgroundColor: "#1E1E1E",
        }}
        ref={containerRef}
      />

      <div
        style={{
          flex: 1,
          padding: "10px",
          borderLeft: "1px solid #333",
          backgroundColor: "#333",
          color: "#FFF",
          display: "flex",
          flexDirection: "column",
          gap: "10px",
        }}
      >
        <h2 style={{ color: "#FFDC00" }}>Graph Controls</h2>

        <button
          onClick={addNode}
          style={{
            padding: "10px",
            backgroundColor: "#0074D9",
            color: "#FFF",
            border: "none",
            borderRadius: "5px",
            cursor: "pointer",
          }}
        >
          Add Node
        </button>

        <div>
          <h4>Add Edge</h4>
          <label style={{ display: "block", marginBottom: "5px" }}>
            From Node:
            <input
              type="number"
              value={fromNode}
              onChange={(e) => setFromNode(e.target.value)}
              style={{
                width: "50px",
                marginLeft: "5px",
                padding: "5px",
                borderRadius: "3px",
              }}
            />
          </label>
          <label style={{ display: "block", marginBottom: "5px" }}>
            To Node:
            <input
              type="number"
              value={toNode}
              onChange={(e) => setToNode(e.target.value)}
              style={{
                width: "50px",
                marginLeft: "5px",
                padding: "5px",
                borderRadius: "3px",
              }}
            />
          </label>
          <button
            onClick={addEdge}
            style={{
              padding: "8px",
              backgroundColor: "#2ECC40",
              color: "#FFF",
              border: "none",
              borderRadius: "5px",
              cursor: "pointer",
            }}
          >
            Add Edge
          </button>
        </div>

        <div>
          <h4>Reduction Algorithm</h4>
          <button
            onClick={applyReductionStep}
            style={{
              padding: "10px",
              backgroundColor: "#FF4136",
              color: "#FFF",
              border: "none",
              borderRadius: "5px",
              cursor: "pointer",
            }}
          >
            Apply Reduction Step
          </button>
          <p style={{ marginTop: "10px", color: "#DDD" }}>
            Log and steps of the reduction algorithm will appear here.
          </p>
        </div>

        <div>
          <h4>Zoom Controls</h4>
          <button
            onClick={zoomIn}
            style={{
              padding: "8px",
              backgroundColor: "#0074D9",
              color: "#FFF",
              border: "none",
              borderRadius: "5px",
              cursor: "pointer",
              marginRight: "5px",
            }}
          >
            Zoom In
          </button>
          <button
            onClick={zoomOut}
            style={{
              padding: "8px",
              backgroundColor: "#FF4136",
              color: "#FFF",
              border: "none",
              borderRadius: "5px",
              cursor: "pointer",
            }}
          >
            Zoom Out
          </button>
        </div>
      </div>
    </div>
  );
};

export default App;
