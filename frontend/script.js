document.addEventListener('DOMContentLoaded', () => {
    // DOM elements
    const fileInput = document.getElementById('file-input');
    const processBtn = document.getElementById('process-btn');
    const fileInfo = document.getElementById('file-info');
    const progressBar = document.getElementById('progress-bar');
    const progressText = document.getElementById('progress-text');
    const resultsSummary = document.getElementById('results-summary');
    const graphContainer = document.getElementById('graph-container');
    
    // State
    let selectedFile = null;
    let graph = null;

    // Event listeners
    fileInput.addEventListener('change', handleFileSelect);
    processBtn.addEventListener('click', processPcapFile);

    function handleFileSelect(event) {
        selectedFile = event.target.files[0];
        if (selectedFile) {
            fileInfo.textContent = `Selected: ${selectedFile.name} (${formatFileSize(selectedFile.size)})`;
            processBtn.disabled = false;
        } else {
            fileInfo.textContent = 'No file selected';
            processBtn.disabled = true;
        }
    }

    async function processPcapFile() {
        if (!selectedFile) {
            alert('Please select a file first');
            return;
        }

        try {
            updateProgress('Uploading file...', 10);
            processBtn.disabled = true;

            const formData = new FormData();
            formData.append('file', selectedFile);

            const response = await fetch('/api/analyze', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Analysis failed');
            }

            updateProgress('Processing results...', 90);
            const results = await response.json();
            
            updateProgress('Complete!', 100);
            showResults(results);
            initVisualization(results);

        } catch (error) {
            console.error('Error:', error);
            updateProgress(`Error: ${error.message}`, 100);
            alert(`Analysis failed: ${error.message}`);
        } finally {
            processBtn.disabled = false;
        }
    }

    function updateProgress(message, percent) {
        progressText.textContent = message;
        progressBar.value = percent;
    }

    function showResults(data) {
        resultsSummary.innerHTML = `
            <h3>Analysis Results</h3>
            <p><strong>Filename:</strong> ${data.meta.filename}</p>
            <p><strong>Processing Time:</strong> ${data.meta.processing_time.toFixed(2)}s</p>
            <p><strong>Device Used:</strong> ${data.meta.device}</p>
            <p><strong>Attacks Detected:</strong> ${data.meta.attack_count}</p>
        `;
    }

    function initVisualization(results) {
        // Clear previous graph if exists
        if (graph) {
            graphContainer.innerHTML = '';
            graph = null;
        }

        // Prepare graph data
        const nodes = results.nodes.map((node, i) => ({
            id: node,
            group: results.predictions[i],
            confidence: Math.max(...results.probabilities[i])
        }));

        const links = results.edges[0].map((srcIdx, i) => ({
            source: results.nodes[srcIdx],
            target: results.nodes[results.edges[1][i]],
            value: 1
        }));

        // Create new graph
        graph = ForceGraph3D()(graphContainer)
            .graphData({ nodes, links })
            .nodeLabel(node => `${node.id}\nConfidence: ${(node.confidence * 100).toFixed(1)}%`)
            .nodeColor(node => node.group ? '#ff3333' : '#00cc00')
            .nodeOpacity(0.9)
            .linkWidth(0.5)
            .linkDirectionalParticles(1)
            .linkDirectionalParticleWidth(2);
    }

    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2) + ' ' + sizes[i]);
    }
});

    const statusEl = document.getElementById('status');
    const tooltipEl = document.getElementById('tooltip');
    const container = document.getElementById('graph-container');
    let originalGraphData;

    // Main initialization function
    async function initializeVisualization() {
      try {
        statusEl.textContent = "Loading graph data...";
        
        // Load data
        const response = await fetch('../data/graph.json');
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        
        const data = await response.json();
        if (!data.nodes || !data.links) throw new Error("Invalid graph structure");
        
        // Store original data
        originalGraphData = JSON.parse(JSON.stringify(data));
        
        // Initialize graph
        initGraph(data);
      } catch (e) {
        statusEl.textContent = `Error: ${e.message}`;
        console.error("Initialization error:", e);
      }
    }

    // Graph initialization
    function initGraph(data) {
      // Clear previous graph if exists
      if (Graph) {
        container.innerHTML = '';
        Graph = null;
      }

      statusEl.textContent = "Rendering...";
      
      // Create new graph instance
      Graph = ForceGraph3D()(container)
        .graphData(data)
        .nodeLabel(node => `${node.id}\nType: ${node.group ? 'MALICIOUS' : 'Benign'}\nConfidence: ${(node.confidence * 100).toFixed(1)}%`)
        .nodeColor(node => {
          if (!node.group) return '#00cc00';
          if (node.confidence >= 0.8) return '#ff3333';
          if (node.confidence >= 0.6) return '#ff6666';
          return '#ff9999';
        })
        .nodeVal(node => node.group ? 2 + (node.confidence * 5) : 2)
        .nodeOpacity(0.9)
        .linkColor(() => 'rgba(255, 255, 255, 0.5)')
        .linkWidth(1)
        .onNodeHover(node => {
          container.style.cursor = node ? 'pointer' : null;
          if (node) {
            showTooltip(node);
            highlightConnections(node);
          } else {
            hideTooltip();
            resetConnections();
          }
        })
        .onNodeClick(zoomToNode)
        .onEngineStop(() => {
          if (!statusEl.textContent.includes('Done!')) {
            const maliciousCount = data.nodes.filter(n => n.group).length;
            statusEl.textContent = `Rendered ${data.nodes.length} nodes (${maliciousCount} malicious), ${data.links.length} links | Done!`;
          }
        });

      // Initial camera position
      Graph.cameraPosition({ z: 500 });

      // Set up event listeners
      setupEventListeners();
    }

    // Helper functions
    function showTooltip(node) {
      tooltipEl.style.display = 'block';
      tooltipEl.innerHTML = `
        <strong>${node.id}</strong><br>
        Type: ${node.group ? 'MALICIOUS' : 'Benign'}<br>
        Confidence: ${(node.confidence * 100).toFixed(1)}%
      `;
    }

    function hideTooltip() {
      tooltipEl.style.display = 'none';
    }

    function highlightConnections(node) {
      Graph.linkColor(link => 
        link.source === node.id || link.target === node.id 
          ? 'rgba(255, 255, 0, 0.8)' 
          : 'rgba(200, 200, 200, 0.2)'
      );
    }

    function resetConnections() {
      Graph.linkColor(() => 'rgba(200, 200, 200, 0.2)');
    }

    function zoomToNode(node) {
      Graph.cameraPosition(
        { x: node.x, y: node.y, z: 150 },
        node,
        1000
      );
    }

    function setupEventListeners() {
      // Mouse movement for tooltip
      container.addEventListener('mousemove', event => {
        if (tooltipEl.style.display === 'block') {
          tooltipEl.style.left = `${event.clientX + 15}px`;
          tooltipEl.style.top = `${event.clientY + 15}px`;
        }
      });

      // Window resize
      window.addEventListener('resize', () => {
        if (Graph) {
          Graph.width(container.offsetWidth);
          Graph.height(container.offsetHeight);
        }
      });
    }

    // Control functions
    function toggleVisibility(type) {
      Graph.nodeVisibility(node => 
        type === 'benign' ? !node.group : node.group
      );
    }

    function highlightNode() {
      const query = document.getElementById('search').value.trim();
      if (!query) return;
      
      Graph.nodeColor(node => 
        node.id.includes(query) 
          ? '#ffff00' 
          : !node.group 
            ? '#00cc00'
            : node.confidence >= 0.8 
              ? '#ff3333'
              : node.confidence >= 0.6 
                ? '#ff6666'
                : '#ff9999'
      );
    }

    function hardReset() {
      // Completely reinitialize the visualization
      initializeVisualization();
      document.getElementById('search').value = '';
    }

    function exportPNG() {
      Graph.screenshot();
    }