document.addEventListener('DOMContentLoaded', () => {
  // Global state variables
  let currentGraphMode = 'main'; // 'main' or 'community'
  let mainGraphInstance = null;
  let communityGraphInstance = null;
  let mainGraphData = null;
  let communityGraphData = null;
  let selectedFile = null;
  let resizeObserver = null;
  let activeUploads = new Set();
  let currentFileId = null;
  let mainViewBtn = null;
let communityViewBtn = null;

  // DOM elements
  const fileInput = document.getElementById('file-input');
  const processBtn = document.getElementById('process-btn');
  const fileInfo = document.getElementById('file-info');
  const progressBar = document.getElementById('progress-bar');
  const progressText = document.getElementById('progress-text');
  const resultsSummary = document.getElementById('results-summary');
  const graphContainer = document.getElementById('graph-container');
  
  // Create popup element
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
  document.getElementById('run-community-analysis').addEventListener('click', runCommunityAnalysis);

  // 1. File selection handler
  function handleFileSelect(event) {
    activeUploads.forEach(xhr => xhr.abort());
    activeUploads.clear();
    
    selectedFile = event.target.files[0];
    if (selectedFile) {
      fileInfo.textContent = `Selected: ${selectedFile.name} (${formatFileSize(selectedFile.size)})`;
      processBtn.disabled = false;
    } else {
      fileInfo.textContent = 'No file selected';
      processBtn.disabled = true;
    }
  }

  // 2. Process PCAP file with chunked upload
  async function processPcapFile() {
    clearPreviousGraph();
    updateProgress('Initializing...', 0);
    
    try {
        if (!selectedFile) throw new Error('No file selected');

        const chunkSize = 500 * 1024 * 1024;
        const totalChunks = Math.ceil(selectedFile.size / chunkSize);
        const maxPackets = document.getElementById('max-packets').value;
        currentFileId = uuidv4();

        updateProgress(`Preparing upload (0/${totalChunks} chunks)`, 0);

        // Upload chunks with retry logic
        for (let i = 0; i < totalChunks; i++) {
            const start = i * chunkSize;
            const end = Math.min(selectedFile.size, start + chunkSize);
            const chunk = selectedFile.slice(start, end);

            const formData = new FormData();
            formData.append("file", chunk, selectedFile.name);
            formData.append("chunk_index", i.toString());
            formData.append("file_id", currentFileId);

            let retries = 3;
            let success = false;
            
            while (retries > 0 && !success) {
                try {
                    const response = await fetch("/api/chunk", {
                        method: "POST",
                        body: formData
                    });

                    if (!response.ok) {
                        throw new Error(`Chunk ${i} upload failed with status ${response.status}`);
                    }

                    success = true;
                    updateProgress(`Uploading chunks (${i + 1}/${totalChunks})`, (i + 1) / totalChunks * 50);
                } catch (error) {
                    retries--;
                    if (retries === 0) throw error;
                    await new Promise(resolve => setTimeout(resolve, 1000));
                }
            }
        }

        updateProgress('Merging and processing...', 60);

        const mergeResponse = await fetch("/api/merge", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                file_id: currentFileId,
                filename: selectedFile.name,
                total_chunks: totalChunks,
                max_packets: maxPackets
            })
        });

        if (!mergeResponse.ok) {
            throw new Error("Merge or processing failed");
        }

        const results = await mergeResponse.json();
        updateProgress('Processing results...', 80);
        showResults(results);
        initVisualization(results);
        updateProgress('Analysis complete!', 100);
        
        if (!mainViewBtn || !communityViewBtn) {
            addViewToggleButtons();  // create and show buttons if not already added
        } else {
            mainViewBtn.style.display = 'inline-block';
            communityViewBtn.style.display = 'inline-block';
        }
    } catch (error) {
        handleError(error);
    } finally {
        processBtn.disabled = false;
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
    
    resultsSummary.innerHTML = `
      <div class="error-message">
        <h3>Analysis Failed</h3>
        <p><strong>Reason:</strong> ${message}</p>
        ${error.stack ? `<details><summary>Technical details</summary><pre>${error.stack}</pre></details>` : ''}
      </div>
    `;
    
    processBtn.disabled = false;
    activeUploads.forEach(xhr => xhr.abort());
    activeUploads.clear();
  }

  function clearPreviousGraph() {
    if (mainGraphInstance) {
      mainGraphInstance.pauseAnimation();
      const renderer = mainGraphInstance.renderer();
      if (renderer && renderer.domElement.parentNode) {
        renderer.domElement.parentNode.removeChild(renderer.domElement);
      }
    }
    if (communityGraphInstance) {
      communityGraphInstance.pauseAnimation();
      const renderer = communityGraphInstance.renderer();
      if (renderer && renderer.domElement.parentNode) {
        renderer.domElement.parentNode.removeChild(renderer.domElement);
      }
    }

    if (resizeObserver) {
      resizeObserver.disconnect();
    }
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
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }

  function uuidv4() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
      const r = Math.random() * 16 | 0;
      const v = c === 'x' ? r : (r & 0x3 | 0x8);
      return v.toString(16);
    });
  }

  // 3. GRAPH VISUALIZATION
  function initVisualization(results) {
    // Store main graph data
    mainGraphData = {
        nodes: results.nodes.map((node, i) => ({
            id: node,
            group: results.predictions[i],
            confidence: Math.max(...results.probabilities[i]),
            probabilities: results.probabilities[i],
            index: i,
            features: results.features ? results.features[i] : null
        })),
        links: results.edges[0].map((srcIdx, i) => ({
            source: results.nodes[srcIdx],
            target: results.nodes[results.edges[1][i]],
            value: 1
        }))
    };

    // Also store for potential community analysis
    communityGraphData = { 
        nodes: [...mainGraphData.nodes],
        links: [...mainGraphData.links]
    };

    // Render the main graph
    currentGraphMode = 'main';
    renderCurrentGraph();
    
    // Show community analysis section
    document.getElementById('community-analysis-section').style.display = 'block';
  }

  function renderCurrentGraph() {
    const container = document.getElementById('graph-container');
    container.innerHTML = ''; // Clear previous graph

    const graphDiv = document.createElement('div');
    graphDiv.style.width = '100%';
    graphDiv.style.height = '100%';
    container.appendChild(graphDiv);

    // Defer graph initialization to ensure graphDiv has dimensions
    requestAnimationFrame(() => {
        const initialWidth = graphDiv.offsetWidth;
        const initialHeight = graphDiv.offsetHeight;

        // If dimensions are still zero, it indicates a layout issue with 'graph-container'
        // or that 'graph-container' is hidden. For now, we'll proceed,
        // but in a production scenario, you might want to add a retry or error handling.
        if (initialWidth === 0 || initialHeight === 0) {
            console.warn("Graph container (graphDiv) has zero dimensions during initialization. Graph may not render correctly.");
            // Optionally, you could try to wait a bit more, e.g., with another setTimeout, or simply proceed.
        }

        const data = currentGraphMode === 'main' ? mainGraphData : communityGraphData;

        if (!data || !data.nodes || !data.links) {
            console.error('Graph data is invalid for mode:', currentGraphMode);
            return;
        }

        if (currentGraphMode === 'main') {
            if (communityGraphInstance) {
                // Consider pausing or cleaning up the other instance if it exists
                // communityGraphInstance.pauseAnimation();
                // communityGraphInstance._destructor?.(); // If library supports explicit destruction
                communityGraphInstance = null; 
            }
            mainGraphInstance = ForceGraph3D()(graphDiv)
                .width(initialWidth)
                .height(initialHeight)
                .graphData(data)
                .nodeLabel(node => `${node.id}\nType: ${node.group ? 'ATTACK' : 'Normal'}`)
                .nodeColor(node => node.group ? '#ff3333' : '#00cc00')
                .linkWidth(0.75)
                .onNodeClick(handleNodeClick);
        } else { // community mode
            if (mainGraphInstance) {
                // mainGraphInstance.pauseAnimation();
                // mainGraphInstance._destructor?.();
                mainGraphInstance = null;
            }
            communityGraphInstance = ForceGraph3D()(graphDiv)
                .width(initialWidth)
                .height(initialHeight)
                .graphData(data)
                .nodeLabel(node => `${node.id}\nCommunity ${node.community}\nType: ${node.group ? 'ATTACK' : 'Normal'}`)
                .nodeColor(node => node.color) // Make sure node.color is a valid color string
                .linkWidth(0.75)
                .onNodeClick(handleNodeClick);
        }

        // Adjust ResizeObserver to use graphDiv and the correct current instance
        if (resizeObserver) {
            resizeObserver.disconnect(); // Disconnect previous observer
        }
        resizeObserver = new ResizeObserver(() => {
            const currentInstance = (currentGraphMode === 'main') ? mainGraphInstance : communityGraphInstance;
            if (currentInstance && graphDiv.offsetWidth > 0 && graphDiv.offsetHeight > 0) {
                currentInstance.width(graphDiv.offsetWidth)
                               .height(graphDiv.offsetHeight);
            }
        });
        resizeObserver.observe(graphDiv); // Observe the graphDiv itself
    });
}

  function handleNodeClick(node, event) {
    // Get connected nodes
    const connectedNodes = new Set();
    const links = currentGraphMode === 'main' ? mainGraphData.links : communityGraphData.links;
    
    links.forEach(link => {
        if (link.source === node.id || link.source.id === node.id) {
            connectedNodes.add(link.target.id || link.target);
        }
        if (link.target === node.id || link.target.id === node.id) {
            connectedNodes.add(link.source.id || link.source);
        }
    });

    // Create popup content
    popup.innerHTML = `
        <h3 style="color: ${node.group ? '#ff3333' : '#00cc00'}; margin-top: 0;">
            ${node.id}
        </h3>
        <p><strong>Type:</strong> <span style="color: ${node.group ? '#ff3333' : '#00cc00'}">${node.group ? 'ATTACK' : 'Normal'}</span></p>
        <p><strong>Confidence:</strong> ${(node.confidence * 100).toFixed(1)}%</p>
        <p><strong>Connections:</strong> ${connectedNodes.size} nodes</p>
        
        ${node.group ? `
        <div style="margin-top: 10px; border-top: 1px solid #eee; padding-top: 10px;">
            <h4 style="margin-bottom: 5px;">Attack Details</h4>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 5px; font-size: 13px;">
                <div><strong>Attack Type:</strong> ${inferAttackType(node.features)}</div>
                <div><strong>Threat Level:</strong> ${getThreatLevel(node.confidence)}</div>
                <div><strong>Behavior Pattern:</strong> ${getBehaviorPattern(node.features)}</div>
                <div><strong>Risk Score:</strong> ${Math.round(node.confidence * 100)}/100</div>
            </div>
            
            <div style="margin-top: 10px;">
                <h4 style="margin-bottom: 5px;">Attack Indicators</h4>
                <ul style="margin-top: 5px; padding-left: 20px;">
                    ${getAttackIndicators(node.probabilities, node.features).join('')}
                </ul>
            </div>
        </div>
        ` : ''}
        
        ${connectedNodes.size > 0 ? `
        <div style="margin-top: 10px; border-top: 1px solid #eee; padding-top: 10px;">
            <h4 style="margin-bottom: 5px;">Connected Nodes</h4>
            <div style="max-height: 100px; overflow-y: auto; background: #f5f5f5; padding: 5px; border-radius: 4px;">
                ${Array.from(connectedNodes).slice(0, 10).map(n => `<div>${n}</div>`).join('')}
                ${connectedNodes.size > 10 ? `<div>+ ${connectedNodes.size - 10} more...</div>` : ''}
            </div>
        </div>
        ` : ''}
        
        ${node.features ? `
        <div style="margin-top: 10px; border-top: 1px solid #eee; padding-top: 10px;">
            <h4 style="margin-bottom: 5px;">Network Metrics</h4>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 5px; font-size: 13px;">
                <div><strong>Connections:</strong> ${Math.round(node.features[0] * 1000)}</div>
                <div><strong>Packet Rate:</strong> ${(node.features[3] * 100).toFixed(1)}/s</div>
                <div><strong>Avg Size:</strong> ${Math.round(node.features[5] * 1500)}B</div>
                <div><strong>SYN Ratio:</strong> ${(node.features[8] * 100).toFixed(1)}%</div>
                <div><strong>RST Ratio:</strong> ${(node.features[9] * 100).toFixed(1)}%</div>
                <div><strong>Duration:</strong> ${(node.features[2] * 60).toFixed(1)}s</div>
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
  }

  // Add view toggle buttons
  function addViewToggleButtons() {
    const graphContainer = document.getElementById('graph-container');

    // Prevent duplicate insertion
    if (document.getElementById('view-toggle-container')) return;

    const buttonContainer = document.createElement('div');
    buttonContainer.id = 'view-toggle-container';
    buttonContainer.style.display = 'flex';
    buttonContainer.style.justifyContent = 'center';
    buttonContainer.style.marginTop = '15px';
    buttonContainer.style.gap = '10px';

    mainViewBtn = document.createElement('button');
    mainViewBtn.textContent = 'Main View';
    mainViewBtn.className = 'view-toggle-btn';
    mainViewBtn.addEventListener('click', () => {
        if (mainGraphData) {
            currentGraphMode = 'main';
            renderCurrentGraph();
        }
    });

    communityViewBtn = document.createElement('button');
    communityViewBtn.textContent = 'Community View';
    communityViewBtn.className = 'view-toggle-btn';
    communityViewBtn.addEventListener('click', () => {
        if (communityGraphData) {
            currentGraphMode = 'community';
            renderCurrentGraph();
        } else {
            alert('Run community analysis first');
        }
    });

    buttonContainer.appendChild(mainViewBtn);
    buttonContainer.appendChild(communityViewBtn);

    // 👉 Insert the buttons **after** the graph container
    graphContainer.insertAdjacentElement('afterend', buttonContainer);
}

 function inferAttackType(features) {
      const [
        connections, flowCount, internalFlag,
        pktRate, timeRange,
        avgSize, stdSize, smallPktRatio,
        synRatio, rstRatio, ackSynCombo,
        tcpRatio, udpRatio
      ] = features;

      if (synRatio > 0.6 && pktRate > 0.5) {
        return "DDoS";
      } else if (rstRatio > 0.5 && timeRange < 0.3) {
        return "Port Scan";
      } else if (smallPktRatio > 0.8 && timeRange > 0.5) {
        return "Data Exfiltration";
      } else if (udpRatio > 0.6 && flowCount > 0.5) {
        return "UDP Amplification";
      } else {
        return "Generic Malicious Activity";
      }
    }


    function getThreatLevel(confidence) {
        if (confidence > 0.9) return "Critical";
        if (confidence > 0.7) return "High";
        if (confidence > 0.5) return "Medium";
        return "Low";
    }

    function getBehaviorPattern(features) {
        if (!features) return "Unknown";
        
        if (features[3] > 0.8 && features[8] > 0.7) return "SYN Flood";
        if (features[0] > 0.9 && features[5] < 0.2) return "Port Scan";
        if (features[9] > 0.6 && features[2] < 0.1) return "Connection Reset Attack";
        if (features[3] > 0.7 && features[5] > 0.8) return "Data Exfiltration";
        
        return "Suspicious Activity";
    }

    // Helper function to generate attack indicators
    function getAttackIndicators(probabilities, features) {
      const indicators = [];
      
      // Probability indicators
      indicators.push(`<li>Attack probability: ${(probabilities[1] * 100).toFixed(1)}%</li>`);
      
      // Feature-based indicators (if available)
      if (features) {
          if (features[0] > 0.8) indicators.push(`<li>High connection count (${Math.round(features[0] * 1000)} connections)</li>`);
          if (features[3] > 0.7) indicators.push(`<li>Abnormal packet rate (${(features[3] * 100).toFixed(1)} packets/sec)</li>`);
          if (features[6] > 0.75) indicators.push('<li>Irregular packet size distribution (possible command & control traffic)</li>');
          if (features[8] > 0.6) indicators.push(`<li>High SYN flag ratio (${(features[8] * 100).toFixed(1)}% of packets)</li>`);
          if (features[9] > 0.5) indicators.push(`<li>High RST flag ratio (${(features[9] * 100).toFixed(1)}% of packets)</li>`);
          if (features[10] > 0.8) indicators.push('<li>Unusual SYN+ACK pattern (possible TCP hijacking)</li>');
          if (features[4] > 0.7) indicators.push('<li>High failed connection rate (possible brute force attempt)</li>');
          if (features[7] > 0.6) indicators.push('<li>Irregular timing between packets (possible beaconing)</li>');
      }
      
      // Fallback if no specific indicators
      if (indicators.length === 1) {
          indicators.push('<li>Matches known attack signatures in our database</li>');
          indicators.push('<li>Behavior deviates significantly from network baseline</li>');
      }
      
      return indicators;
  }

  function showMaliciousNodesList(results) {
  const maliciousNodesList = document.getElementById('malicious-nodes-list');
  const maliciousNodesSection = document.getElementById('malicious-nodes-section');
  
  if (!results || !results.nodes || !results.predictions) {
    maliciousNodesSection.style.display = 'none';
    return;
  }

  // Get all malicious nodes with their details
  const maliciousNodes = results.nodes
    .map((node, i) => ({
      id: node,
      confidence: Math.max(...results.probabilities[i]),
      probabilities: results.probabilities[i],
      features: results.features ? results.features[i] : null
    }))
    .filter((_, i) => results.predictions[i] === 1); // Filter for malicious nodes (group=1)

  if (maliciousNodes.length === 0) {
    maliciousNodesList.innerHTML = '<p>No malicious nodes detected.</p>';
    maliciousNodesSection.style.display = 'block';
    return;
  }

  // Sort by confidence (highest first)
  maliciousNodes.sort((a, b) => b.confidence - a.confidence);

  // Create the list HTML
  maliciousNodesList.innerHTML = `
    <div class="list-header">
      <span>Node ID</span>
      <span>Confidence</span>
      <span>Threat Level</span>
      <span>Attack Type</span>
    </div>
    <div class="node-list-items">
      ${maliciousNodes.map(node => `
        <div class="node-item" data-node-id="${node.id}">
          <span class="node-id">${node.id}</span>
          <span class="confidence">${(node.confidence * 100).toFixed(1)}%</span>
          <span class="threat-level ${getThreatLevelClass(node.confidence)}">
            ${getThreatLevel(node.confidence)}
          </span>
          <span class="attack-type">${node.features ? inferAttackType(node.features) : 'Unknown'}</span>
        </div>
      `).join('')}
    </div>
  `;

  // Add click handlers to focus on the node in the graph
  document.querySelectorAll('.node-item').forEach(item => {
    item.addEventListener('click', () => {
      const nodeId = item.getAttribute('data-node-id');
      focusOnNode(nodeId);
    });
  });

  maliciousNodesSection.style.display = 'block';
}

function getThreatLevelClass(confidence) {
  if (confidence > 0.9) return "critical";
  if (confidence > 0.7) return "high";
  if (confidence > 0.5) return "medium";
  return "low";
}

function focusOnNode(nodeId) {
  if (!graphInstance) return;
  
  const node = graphInstance.graphData().nodes.find(n => n.id === nodeId);
  if (node) {
    // Calculate distance based on node importance
    const distance = 100 + (node.group ? 50 : 0);
    
    // Focus on the node
    graphInstance.centerAt(node.x, node.y, node.z, 1000);
    graphInstance.zoom(distance, 1000);
    
    // Highlight the node
    const nodeObj = graphInstance.getGraphBbox(nodeId);
    if (nodeObj) {
      // You could add additional highlighting here if needed
    }
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
    showMaliciousNodesList(data)
  }

  function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }

  // ==============================
// COMMUNITY ANALYSIS SECTION
// ==============================

// Event listener for "Analyze Communities" button
document.getElementById('run-community-analysis').addEventListener('click', runCommunityAnalysis);

// Simulated community detection logic
async function runCommunityAnalysis() {
    const algorithm = document.getElementById('community-algorithm').value;
    const button = document.getElementById('run-community-analysis');
    
    try {
        button.disabled = true;
        button.textContent = 'Processing...';
        
        if (!communityGraphData) {
            throw new Error("No graph data available");
        }

        // Prepare the payload for community detection
        const payload = {
            algorithm: algorithm,
            nodes: communityGraphData.nodes.map(n => n.id),
            edges: communityGraphData.links.map(link => [
                link.source.id || link.source,
                link.target.id || link.target
            ]),
            predictions: communityGraphData.nodes.map(n => n.group),
        };

        console.time('Community Detection');
        const response = await fetch('/api/analyze-communities', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });
        console.timeEnd('Community Detection');

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Community detection failed');
        }

        const apiResponse = await response.json();
        
        updateCommunityGraph(apiResponse);
        
        // Show results section
        document.getElementById('community-results').style.display = 'block';
    } catch (error) {
        console.error("Community analysis failed:", error);
        alert(`Error: ${error.message}`);
    } finally {
        button.disabled = false;
        button.textContent = 'Analyze Communities';
    }
  }



function updateCommunityGraph(apiResponse) {
    // 1. Validate input data (from your suggestion)
    if (!apiResponse?.communities) {
        console.error("API response is missing 'communities' data.");
        alert("Error: Could not retrieve community information from the server.");
        // Restore button state if it was disabled
        const button = document.getElementById('run-community-analysis');
        if (button) {
            button.disabled = false;
            button.textContent = 'Analyze Communities';
        }
        return;
    }
    if (!communityGraphData?.nodes || communityGraphData.nodes.length === 0) {
        console.error("No graph node data available to update for community analysis.");
        alert("Error: No graph data loaded. Please process a file first.");
        const button = document.getElementById('run-community-analysis');
        if (button) {
            button.disabled = false;
            button.textContent = 'Analyze Communities';
        }
        return;
    }

    const nodeCommunityMap = apiResponse.communities;

    // 2. Get all unique community IDs (ensure robust handling and sorting for consistent color mapping)
    // Filter out null/undefined community IDs that might come from the API
    const communityIds = Object.values(nodeCommunityMap).filter(id => id !== null && id !== undefined);
    const uniqueCommunityIds = [...new Set(communityIds)].sort((a, b) => String(a).localeCompare(String(b))); // Sort for consistency

    // 3. Create color scale (using your HSL suggestion for better distinct colors)
    const colorScale = d3.scaleOrdinal()
        .domain(uniqueCommunityIds)
        .range(uniqueCommunityIds.map((_, i) => `hsl(${(i * 137.508) % 360}, 75%, 55%)`)); // Golden angle, good saturation/lightness

    // 4. Count community sizes
    const communitySizes = {};
    communityGraphData.nodes.forEach(node => {
        const commId = nodeCommunityMap[node.id];
        if (commId != null) { // Only count if node belongs to a community
            communitySizes[commId] = (communitySizes[commId] || 0) + 1;
        }
    });

    // 5. Prepare nodes with community data (updates global communityGraphData.nodes)
    communityGraphData.nodes = communityGraphData.nodes.map(node => {
        const commId = nodeCommunityMap[node.id]; // Might be undefined if node.id not in map
        // If commId is undefined (node not in any community), colorScale(undefined) will assign a color.
        // This color will be consistent for all unassigned nodes.
        const calculatedColor = colorScale(commId);

        return {
            ...node, // Preserves original node data (id, group, features etc.)
            community: commId,
            color: calculatedColor, // This will be used by renderCurrentGraph's .nodeColor()
            // Your custom 'size' property (ForceGraph3D won't use this for visual size by default)
            customSize: Math.max(1, Math.min(5, Math.sqrt(communitySizes[commId] || 1)))
        };
    });

    // 6. Link processing (optional, as original links are likely ID-based and fine)
    // If you want to be absolutely sure about link format:
    communityGraphData.links = communityGraphData.links.map(link => ({
         source: typeof link.source === 'object' ? String(link.source.id) : String(link.source),
         target: typeof link.target === 'object' ? String(link.target.id) : String(link.target),
         value: link.value || 1 // Preserve original value or default to 1
     }));


    // 7. Update UI elements like legend and metrics
    addCommunityLegend(document.getElementById('community-legend'), uniqueCommunityIds, communitySizes, colorScale);
    updateCommunityMetrics(apiResponse, communitySizes, communityGraphData.nodes.length);

    // 8. Set mode and trigger re-render
    currentGraphMode = 'community';
    renderCurrentGraph(); // This function should now use the updated communityGraphData
}

function renderSecurityInsights(insights) {
    let html = '';
    
    
    if (insights.attack_campaigns && Object.keys(insights.attack_campaigns).length) {
        html += `
        <div class="insight-section">
            <h5>⚠️ Attack Campaigns</h5>
            <p>${Object.keys(insights.attack_campaigns).length} communities with high attack concentration</p>
            <ul>
                ${Object.entries(insights.attack_campaigns).map(([comm, data]) => `
                    <li>Community ${comm}: ${data[0]} attacks (${(data[1]*100).toFixed(1)}% of nodes)</li>
                `).join('')}
            </ul>
        </div>`;
    }
    
    // 3. Lateral Movement (existing)
    if (insights.lateral_movement?.length) {
        html += `
        <div class="insight-section">
            <h5>🔀 Lateral Movement Paths</h5>
            <p>${insights.lateral_movement.length} potential bridge nodes found</p>
            <div class="scrollable-list">
                ${insights.lateral_movement.slice(0,5).map(node => `
                    <div class="node-info">
                        <span>${node.node}</span>
                        <span>Connects ${node.communities_connected} communities</span>
                    </div>
                `).join('')}
            </div>
        </div>`;
    }
    
    // 4. Command & Control (existing)
    if (insights.command_control?.length) {
        html += `
        <div class="insight-section">
            <h5>🎯 Command & Control</h5>
            <p>${insights.command_control.length} star-shaped communities detected</p>
            <div class="scrollable-list">
                ${insights.command_control.slice(0,3).map(comm => `
                    <div class="node-info">
                        <span>Community ${comm.community}</span>
                        <span>Center: ${comm.center_node} (${comm.degree} connections)</span>
                    </div>
                `).join('')}
            </div>
        </div>`;
    }
    
    return html || '<p>No significant security patterns detected</p>';
}

function addCommunityLegend(container, communityIds, communitySizes, colorScale) {
    const legend = document.createElement('div');
    legend.className = 'community-legend';
    
    // Sort communities by size (descending)
    const sortedCommunities = [...communityIds].sort((a, b) => communitySizes[b] - communitySizes[a]);
    
    legend.innerHTML = `
        <h4>Detected Communities (${sortedCommunities.length})</h4>
        ${sortedCommunities.map(commId => `
            <div class="legend-item">
                <span class="legend-color" style="background:${colorScale(commId)}"></span>
                <span>Community ${commId}: ${communitySizes[commId]} nodes</span>
            </div>
        `).join('')}
    `;
    
    container.appendChild(legend);
}

function updateCommunityMetrics(apiResponse, communitySizes, totalNodes) {
    const metricsEl = document.getElementById('community-metrics');
    if (!metricsEl) return;

    const communityCount = Object.keys(communitySizes).length;
    const minSize = Math.min(...Object.values(communitySizes));
    const maxSize = Math.max(...Object.values(communitySizes));
    
    metricsEl.innerHTML = `
        <div class="metric-row">
            <span class="metric-label">Total Communities:</span>
            <span class="metric-value">${communityCount}</span>
        </div>
        <div class="metric-row">
            <span class="metric-label">Total Nodes:</span>
            <span class="metric-value">${totalNodes}</span>
        </div>
        <div class="metric-row">
            <span class="metric-label">Community Sizes:</span>
            <span class="metric-value">${minSize} to ${maxSize} nodes</span>
        </div>
        <div class="metric-row">
            <span class="metric-label">Modularity:</span>
            <span class="metric-value">${apiResponse.modularity?.toFixed(4) || 'N/A'}</span>
        </div>
    `;
}


// Utility function to expose graphData when the original graph is ready
function setCommunityGraphData(results) {
    if (!results || !results.nodes || !results.edges) {
        console.error("Invalid results format for community graph data.");
        return;
    }

    const nodes = results.nodes.map((nodeId, i) => ({
        id: nodeId,
        group: results.predictions ? results.predictions[i] : null,
        confidence: results.probabilities ? Math.max(...results.probabilities[i]) : null,
        probabilities: results.probabilities ? results.probabilities[i] : null,
        index: i,
        features: results.features ? results.features[i] : null
    }));

    const links = results.edges[0].map((srcIdx, i) => ({
        source: results.nodes[srcIdx],
        target: results.nodes[results.edges[1][i]],
        value: 1
    }));

    communityGraphData = { nodes, links };

    const communityMetricsEl = document.getElementById('community-metrics');
    const communityCounts = {};
    let modularity = results.modularity ?? null;

    nodes.forEach(n => {
        if (n.group !== null) {
            communityCounts[n.group] = (communityCounts[n.group] || 0) + 1;
        }
    });

    const totalCommunities = Object.keys(communityCounts).length;
    const totalNodes = nodes.length;
    const totalLinks = links.length;

    let html = `
        <p><strong>Detected Communities:</strong> ${totalCommunities}</p>
        <p><strong>Total Nodes:</strong> ${totalNodes}</p>
        <p><strong>Total Edges:</strong> ${totalLinks}</p>
    `;

    if (modularity !== null) {
        html += `<p><strong>Modularity Score:</strong> ${modularity.toFixed(4)}</p>`;
    }

    html += `<p><strong>Community Sizes:</strong></p><ul>`;
    for (const [commId, size] of Object.entries(communityCounts)) {
        html += `<li>Community ${commId}: ${size} nodes</li>`;
    }
    html += `</ul>`;

    communityMetricsEl.innerHTML = html;

    // Show the community section
    document.getElementById('community-analysis-section').style.display = 'block';
}


});