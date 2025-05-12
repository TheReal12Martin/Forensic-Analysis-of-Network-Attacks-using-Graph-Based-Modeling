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
  let currentXHR = null;
  let nodeData = null; // Store node data for popup

  // Create popup elements
  const popup = document.createElement('div');
  popup.id = 'node-popup';
  popup.style.position = 'absolute';
  popup.style.backgroundColor = 'white';
  popup.style.border = '1px solid #ccc';
  popup.style.borderRadius = '5px';
  popup.style.padding = '15px';
  popup.style.boxShadow = '0 2px 10px rgba(0,0,0,0.2)';
  popup.style.zIndex = '100';
  popup.style.display = 'none';
  popup.style.maxWidth = '300px';
  document.body.appendChild(popup);

  // Close popup when clicking anywhere
  document.addEventListener('click', (e) => {
    if (!popup.contains(e.target)) {
      popup.style.display = 'none';
    }
  });

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
    if (currentXHR) {
      currentXHR.abort();
      currentXHR = null;
    }

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
    // Clear previous results
    clearPreviousGraph();
    updateProgress('Initializing...', 0);
    
    try {
        // Validate file selection
        if (!selectedFile) {
            throw new Error('No file selected');
        }

        // Validate file size
        const maxSize = 10 * 1024 * 1024 * 1024; // 10GB
        if (selectedFile.size > maxSize) {
            if (!confirm(`This file is large (${formatFileSize(selectedFile.size)}). Processing may take time. Continue?`)) {
                return;
            }
        }

        // Prepare form data
        const formData = new FormData();
        const maxPackets = document.getElementById('max-packets').value;
        
        // Create blob with correct MIME type
        const fileBlob = new Blob([selectedFile], { 
            type: 'application/vnd.tcpdump.pcap' 
        });
        formData.append('file', fileBlob, selectedFile.name);
        formData.append('max_packets', maxPackets);

        // Configure request
        currentXHR = new XMLHttpRequest();
        currentXHR.open('POST', '/api/analyze', true);
        currentXHR.responseType = 'json';
        
        // Add debug headers
        currentXHR.setRequestHeader('X-Debug-Info', 'pcap-analysis');
        currentXHR.setRequestHeader('X-File-Size', selectedFile.size);

        // Upload progress tracking
        currentXHR.upload.onprogress = (e) => {
            if (e.lengthComputable) {
                const percent = Math.round((e.loaded / e.total) * 100);
                updateProgress(`Uploading... ${percent}%`, percent * 0.4);
            }
        };

        // Response handling
        currentXHR.onload = () => {
            if (currentXHR.status === 200) {
                try {
                    updateProgress('Processing results...', 60);
                    const results = currentXHR.response;
                    
                    // Validate response structure
                    if (!results || !results.nodes || !results.predictions) {
                        throw new Error('Invalid server response format');
                    }
                    
                    updateProgress('Rendering visualization...', 80);
                    showResults(results);
                    initVisualization(results);
                    updateProgress('Analysis complete!', 100);
                    
                } catch (e) {
                    handleError(new Error(`Result processing failed: ${e.message}`));
                }
            } else {
                let errorMsg = currentXHR.statusText;
                try {
                    const errorResponse = JSON.parse(currentXHR.responseText);
                    errorMsg = errorResponse.detail || errorMsg;
                } catch {}
                handleError(new Error(`Server error: ${currentXHR.status} - ${errorMsg}`));
            }
            currentXHR = null;
        };

        currentXHR.onerror = () => {
            handleError(new Error('Network connection failed'));
            currentXHR = null;
        };

        currentXHR.onabort = () => {
            updateProgress('Upload cancelled', 0);
            currentXHR = null;
        };

        // Start processing
        updateProgress('Starting upload...', 5);
        processBtn.disabled = true;
        currentXHR.send(formData);

    } catch (error) {
        handleError(error);
    }
}

function handleError(error) {
    console.error('Error:', error);
    
    let message = error.message;
    if (message.includes('413')) {
        message = 'File too large (max 10GB)';
    } else if (message.includes('400')) {
        message = 'Invalid file format or parameters';
    } else if (message.includes('Network Error')) {
        message = 'Network connection failed';
    }
    
    updateProgress(`Error: ${message}`, 100);
    
    // Show detailed error in results section
    resultsSummary.innerHTML = `
        <div class="error-message">
            <h3>Analysis Failed</h3>
            <p><strong>Reason:</strong> ${message}</p>
            ${error.stack ? `<details><summary>Technical details</summary><pre>${error.stack}</pre></details>` : ''}
        </div>
    `;
    
    processBtn.disabled = false;
    if (currentXHR) {
        currentXHR.abort();
        currentXHR = null;
    }
}

function clearPreviousGraph() {
    if (graphInstance) {
        try {
            graphInstance.pauseAnimation();
            const renderer = graphInstance.renderer();
            if (renderer) {
                renderer.dispose();
                if (renderer.domElement.parentNode) {
                    renderer.domElement.parentNode.removeChild(renderer.domElement);
                }
            }
        } catch (e) {
            console.warn('Error cleaning up graph:', e);
        }
    }

    // Create fresh container
    const newContainer = document.createElement('div');
    newContainer.id = 'graph-container';
    newContainer.style.width = '100%';
    newContainer.style.height = '600px';
    graphContainer.replaceWith(newContainer);
    graphContainer = newContainer;
}

function updateProgress(message, percent) {
    progressText.textContent = message;
    progressBar.value = percent;
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2) + ' ' + sizes[i]);
}

  // 4. GRAPH INITIALIZATION with popup support
  function initVisualization(results) {
    // Create popup element if it doesn't exist
    let popup = document.getElementById('node-popup');
    if (!popup) {
        popup = document.createElement('div');
        popup.id = 'node-popup';
        document.body.appendChild(popup);
    }

    // Prepare data with all needed properties
    const nodes = results.nodes.map((node, i) => ({
        id: node,
        group: results.predictions[i],
        confidence: Math.max(...results.probabilities[i]),
        probabilities: results.probabilities[i],
        index: i,
        features: results.features ? results.features[i] : null
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
    graphContainer.appendChild(graphDiv);

    // Initialize graph with enhanced click handler
    graphInstance = ForceGraph3D()(graphDiv)
        .graphData({ nodes, links })
        .nodeLabel(node => `${node.id}\nType: ${node.group ? 'ATTACK' : 'Normal'}`)
        .nodeColor(node => node.group ? '#ff3333' : '#00cc00')
        .nodeVal(node => node.group ? 2 : 1) // Make attack nodes slightly larger
        .onNodeClick((node, event) => {
            // Get connected nodes
            const connectedNodes = [];
            links.forEach(link => {
                if (link.source === node.id) connectedNodes.push(link.target);
                if (link.target === node.id) connectedNodes.push(link.source);
            });

            // Get the original node data
            const nodeIndex = results.nodes.indexOf(node.id);
            const probabilities = results.probabilities[nodeIndex];
            const features = results.features ? results.features[nodeIndex] : null;

            // Create enhanced popup content
            popup.innerHTML = `
                <h3 style="color: ${node.group ? '#ff3333' : '#00cc00'}; margin-top: 0;">
                    ${node.id}
                </h3>
                <p><strong>Type:</strong> <span style="color: ${node.group ? '#ff3333' : '#00cc00'}">${node.group ? 'ATTACK' : 'Normal'}</span></p>
                <p><strong>Confidence:</strong> ${(node.confidence * 100).toFixed(1)}%</p>
                <p><strong>Connections:</strong> ${connectedNodes.length} nodes</p>
                
                ${node.group ? `
                <div style="margin-top: 10px; border-top: 1px solid #eee; padding-top: 10px;">
                    <h4 style="margin-bottom: 5px;">Attack Indicators</h4>
                    <ul style="margin-top: 5px; padding-left: 20px;">
                        ${getAttackIndicators(probabilities, features).join('')}
                    </ul>
                </div>
                ` : ''}
                
                ${connectedNodes.length > 0 ? `
                <div style="margin-top: 10px; border-top: 1px solid #eee; padding-top: 10px;">
                    <h4 style="margin-bottom: 5px;">Connected Nodes</h4>
                    <div style="max-height: 100px; overflow-y: auto; background: #f5f5f5; padding: 5px; border-radius: 4px;">
                        ${connectedNodes.slice(0, 10).map(n => `<div>${n}</div>`).join('')}
                        ${connectedNodes.length > 10 ? `<div>+ ${connectedNodes.length - 10} more...</div>` : ''}
                    </div>
                </div>
                ` : ''}
                
                ${features ? `
                <div style="margin-top: 10px; border-top: 1px solid #eee; padding-top: 10px;">
                    <h4 style="margin-bottom: 5px;">Key Metrics</h4>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 5px; font-size: 13px;">
                        <div><strong>Connections:</strong> ${Math.round(features[0] * 1000)}</div>
                        <div><strong>Packet Rate:</strong> ${(features[3] * 100).toFixed(1)}/s</div>
                        <div><strong>Avg Size:</strong> ${Math.round(features[5] * 1500)}B</div>
                        <div><strong>SYN Ratio:</strong> ${(features[8] * 100).toFixed(1)}%</div>
                    </div>
                </div>
                ` : ''}
                
                <button onclick="this.parentElement.style.display='none'" 
                        style="margin-top: 15px; width: 100%; padding: 8px; background: #2196F3; color: white; border: none; border-radius: 4px; cursor: pointer;">
                    Close
                </button>
            `;
            
            // Position and show popup
            popup.style.display = 'block';
            popup.style.left = `${Math.min(event.clientX + 15, window.innerWidth - 350)}px`;
            popup.style.top = `${Math.min(event.clientY + 15, window.innerHeight - 400)}px`;
            
            event.stopPropagation();
        });

    // Close popup when clicking elsewhere
    document.addEventListener('click', (e) => {
        if (e.target !== popup && !popup.contains(e.target)) {
            popup.style.display = 'none';
        }
    });

    // Handle window resize
    resizeObserver = new ResizeObserver(() => {
        if (graphInstance) {
            graphInstance.width(graphContainer.offsetWidth)
                       .height(graphContainer.offsetHeight);
        }
    });
    resizeObserver.observe(graphContainer);

    // Helper function to generate attack indicators
    function getAttackIndicators(probabilities, features) {
        const indicators = [];
        
        // Probability indicators
        indicators.push(`<li>High attack probability (${(probabilities[1] * 100).toFixed(1)}%)</li>`);
        
        // Feature-based indicators (if available)
        if (features) {
            if (features[0] > 0.8) indicators.push('<li>Unusually high connection count</li>');
            if (features[3] > 0.7) indicators.push('<li>Abnormal packet rate</li>');
            if (features[6] > 0.75) indicators.push('<li>Suspicious packet size distribution</li>');
            if (features[8] > 0.6) indicators.push('<li>High SYN flag ratio</li>');
            if (features[9] > 0.5) indicators.push('<li>High RST flag ratio</li>');
            if (features[10] > 0.8) indicators.push('<li>Unusual SYN+ACK pattern</li>');
        }
        
        // Fallback if no specific indicators
        if (indicators.length === 1) {
            indicators.push('<li>Anomalous behavior detected</li>');
            indicators.push('<li>Pattern matches known attack signatures</li>');
        }
        
        return indicators;
    }
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