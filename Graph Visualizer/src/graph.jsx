import React, { useState, useEffect } from 'react';

const DRAWING_ALGORITHMS = {
  'force-directed': {
    name: 'Force-Directed',
    algorithms: ['BFS', 'DFS', 'Dijkstra', 'Prim', 'Kruskal']
  },
  'circular': {
    name: 'Circular',
    algorithms: ['BFS', 'DFS', 'Coloring']
  },
  'spectral': {
    name: 'Spectral',
    algorithms: ['Clustering', 'Community Detection']
  }
};

const GraphVisualizer = () => {
  // State management
  const [nodes, setNodes] = useState([]);
  const [edges, setEdges] = useState([]);
  const [drawingMethod, setDrawingMethod] = useState('force-directed');
  const [selectedAlgorithm, setSelectedAlgorithm] = useState('');
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [nodeForm, setNodeForm] = useState({ id: '', x: '', y: '' });
  const [edgeForm, setEdgeForm] = useState({ source: '', target: '', weight: '1' });
  const [output, setOutput] = useState('');

  // SVG settings
  const width = 800;
  const height = 600;

  // Handle node addition
  const handleAddNode = () => {
    const newNode = {
      id: nodes.length + 1,
      x: Math.random() * (width - 100) + 50,
      y: Math.random() * (height - 100) + 50
    };
    setNodes([...nodes, newNode]);
  };

  // Handle edge addition
  const handleAddEdge = (e) => {
    e.preventDefault();
    if (edgeForm.source && edgeForm.target) {
      const newEdge = {
        source: parseInt(edgeForm.source) - 1,
        target: parseInt(edgeForm.target) - 1,
        weight: parseFloat(edgeForm.weight)
      };
      setEdges([...edges, newEdge]);
      setEdgeForm({ source: '', target: '', weight: '1' });
    }
  };

  return (
    <div className="flex h-screen bg-gray-900 text-white">
      {/* Left Panel - Graph Display */}
      <div className="w-2/3 p-4 border-r border-gray-700">
        <div className="bg-gray-800 rounded-lg p-4 h-full">
          <svg 
            width={width} 
            height={height} 
            className="bg-gray-900 rounded-lg"
          >
            {/* Draw edges */}
            {edges.map((edge, i) => (
              <line
                key={`edge-${i}`}
                x1={nodes[edge.source]?.x}
                y1={nodes[edge.source]?.y}
                x2={nodes[edge.target]?.x}
                y2={nodes[edge.target]?.y}
                stroke="#4299e1"
                strokeWidth="2"
              />
            ))}
            {/* Draw nodes */}
            {nodes.map((node) => (
              <g key={`node-${node.id}`}>
                <circle
                  cx={node.x}
                  cy={node.y}
                  r="20"
                  fill="#4299e1"
                  stroke="#2b6cb0"
                  strokeWidth="2"
                />
                <text
                  x={node.x}
                  y={node.y}
                  textAnchor="middle"
                  dy=".3em"
                  fill="white"
                >
                  {node.id}
                </text>
              </g>
            ))}
          </svg>
        </div>
      </div>

      {/* Right Panel - Controls */}
      <div className="w-1/3 p-4 space-y-4">
        {/* Drawing Method Selection */}
        <div className="space-y-2">
          <label className="block text-sm font-medium">Drawing Technique</label>
          <select
            value={drawingMethod}
            onChange={(e) => {
              setDrawingMethod(e.target.value);
              setSelectedAlgorithm('');
            }}
            className="w-full bg-gray-800 border border-gray-700 rounded-md p-2"
          >
            {Object.entries(DRAWING_ALGORITHMS).map(([key, value]) => (
              <option key={key} value={key}>{value.name}</option>
            ))}
          </select>
        </div>

        {/* Algorithm Selection */}
        <div className="space-y-2">
          <label className="block text-sm font-medium">Algorithm</label>
          <select
            value={selectedAlgorithm}
            onChange={(e) => setSelectedAlgorithm(e.target.value)}
            className="w-full bg-gray-800 border border-gray-700 rounded-md p-2"
          >
            <option value="">Select Algorithm</option>
            {DRAWING_ALGORITHMS[drawingMethod].algorithms.map(algo => (
              <option key={algo} value={algo}>{algo}</option>
            ))}
          </select>
        </div>

        {/* Graph Controls */}
        <div className="space-y-4">
          <button
            onClick={handleAddNode}
            className="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded"
          >
            Add Node
          </button>

          {/* Edge Addition Form */}
          <form onSubmit={handleAddEdge} className="space-y-2">
            <div className="grid grid-cols-3 gap-2">
              <input
                type="number"
                placeholder="Source"
                value={edgeForm.source}
                onChange={(e) => setEdgeForm({...edgeForm, source: e.target.value})}
                className="bg-gray-800 border border-gray-700 rounded p-2"
                min="1"
              />
              <input
                type="number"
                placeholder="Target"
                value={edgeForm.target}
                onChange={(e) => setEdgeForm({...edgeForm, target: e.target.value})}
                className="bg-gray-800 border border-gray-700 rounded p-2"
                min="1"
              />
              <input
                type="number"
                placeholder="Weight"
                value={edgeForm.weight}
                onChange={(e) => setEdgeForm({...edgeForm, weight: e.target.value})}
                className="bg-gray-800 border border-gray-700 rounded p-2"
                step="0.1"
              />
            </div>
            <button
              type="submit"
              className="w-full bg-green-600 hover:bg-green-700 text-white font-bold py-2 px-4 rounded"
            >
              Add Edge
            </button>
          </form>

          {/* File Upload */}
          <div className="space-y-2">
            <label className="block text-sm font-medium">Upload Graph</label>
            <input
              type="file"
              accept=".json"
              className="w-full bg-gray-800 border border-gray-700 rounded p-2"
            />
          </div>
        </div>

        {/* Playback Controls */}
        <div className="space-y-4">
          <div className="flex justify-center space-x-4">
            <button className="p-2 bg-gray-800 rounded">⏮️</button>
            <button className="p-2 bg-gray-800 rounded">
              {isPlaying ? '⏸️' : '▶️'}
            </button>
            <button className="p-2 bg-gray-800 rounded">⏭️</button>
          </div>
          <input
            type="range"
            min="0"
            max="100"
            value={currentStep}
            onChange={(e) => setCurrentStep(parseInt(e.target.value))}
            className="w-full"
          />
        </div>

        {/* Output Display */}
        <div className="bg-gray-800 rounded-lg p-4 h-32 overflow-auto">
          <pre className="text-sm">
            {output || 'Algorithm output will appear here...'}
          </pre>
        </div>
      </div>
    </div>
  );
};

export default GraphVisualizer;