document.addEventListener('DOMContentLoaded', () => {
  // DOM elements
  const fileInput = document.getElementById('file-input');
  const processBtn = document.getElementById('process-btn');
  const fileInfo = document.getElementById('file-info');
  const progressBar = document.getElementById('progress-bar');
  const progressText = document.getElementById('progress-text');
  const resultsSummary = document.getElementById('results-summary');
  let graphContainer = document.getElementById('graph-container');
  
  // State
  let selectedFile = null;
  let graphInstance = null;
  let resizeObserver = null;

  // Event listeners
  fileInput.addEventListener('change', handleFileSelect);
  processBtn.addEventListener('click', processPcapFile);

  document.getElementById('max-packets').addEventListener('change', function() {
    const value = parseInt(this.value);
    if (value < 1000) {
        this.value = 1000;
        alert('Minimum packet limit is 1000');
    }
});

  // 1. File selection handler
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

  // 2. Process PCAP file
  async function processPcapFile() {
    if (!selectedFile) {
        alert('Please select a file first');
        return;
    }

    try {
        updateProgress('Uploading file...', 10);
        processBtn.disabled = true;
        clearPreviousGraph();

        const maxPackets = document.getElementById('max-packets').value;
        const formData = new FormData();
        formData.append('file', selectedFile);
        formData.append('max_packets', maxPackets);  // Add max_packets parameter

        const response = await fetch('/api/analyze', {
            method: 'POST',
            body: formData
        });

      if (!response.ok) throw new Error('Analysis failed');
      
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

  // 3. IMPROVED GRAPH CLEANUP (guaranteed no ghosting)
  function clearPreviousGraph() {
    // Stop any running animations
    if (graphInstance && graphInstance.pauseAnimation) {
      graphInstance.pauseAnimation();
    }

    // Dispose of Three.js resources properly
    if (graphInstance && graphInstance.renderer) {
      const renderer = graphInstance.renderer;
      renderer.dispose();
      renderer.forceContextLoss();
      
      // Remove canvas if it exists
      if (renderer.domElement && renderer.domElement.parentNode) {
        renderer.domElement.parentNode.removeChild(renderer.domElement);
      }
    }

    // Clear resize observer
    if (resizeObserver) {
      resizeObserver.disconnect();
      resizeObserver = null;
    }

    // Nuclear option: Replace the entire container
    const newContainer = document.createElement('div');
    newContainer.id = 'graph-container';
    newContainer.style.width = '100%';
    newContainer.style.height = '600px';
    graphContainer.replaceWith(newContainer);
    graphContainer = newContainer;

    // Force garbage collection (available in Chrome with --js-flags="--expose-gc")
    if (window.gc) window.gc();
  }

  // 4. GRAPH INITIALIZATION (with safety checks)
  function initVisualization(results) {
    // Prepare data
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

    // Create fresh container
    const graphDiv = document.createElement('div');
    graphDiv.style.width = '100%';
    graphDiv.style.height = '100%';
    graphDiv.style.position = 'absolute';
    graphContainer.appendChild(graphDiv);

    // Initialize graph with proper cleanup handlers
    graphInstance = ForceGraph3D()(graphDiv)
      .graphData({ nodes, links })
      .nodeLabel(node => `${node.id}\n${node.group ? 'ATTACK' : 'Normal'}\nConfidence: ${(node.confidence * 100).toFixed(1)}%`)
      .nodeColor(node => node.group ? '#ff3333' : '#00cc00')
      .onEngineStop(() => {
        // Reduce physics updates after initial layout
        graphInstance.d3VelocityDecay(0.1);
      });

    // Handle window resize
    resizeObserver = new ResizeObserver(() => {
      if (graphInstance) {
        graphInstance.width(graphContainer.offsetWidth)
                   .height(graphContainer.offsetHeight);
      }
    });
    resizeObserver.observe(graphContainer);
  }

  // Helper functions
  function updateProgress(message, percent) {
    progressText.textContent = message;
    progressBar.value = percent;
  }

  function showResults(data) {
    const maxPackets = document.getElementById('max-packets').value;
    resultsSummary.innerHTML = `
        <h3>Analysis Results</h3>
        <p><strong>Filename:</strong> ${data.meta.filename}</p>
        <p><strong>Processing Time:</strong> ${data.meta.processing_time.toFixed(2)}s</p>
        <p><strong>Device Used:</strong> ${data.meta.device}</p>
        <p><strong>Packet Limit:</strong> ${maxPackets}</p>
        <p><strong>Attacks Detected:</strong> ${data.meta.attack_count}</p>
    `;
}

  function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }
});