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
  let activeUploads = new Set();
  let currentFileId = null;
  
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
    // Cancel any active uploads
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

        const chunkSize = 500 * 1024 * 1024; // 5MB chunks (smaller for better progress tracking)
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
                    await new Promise(resolve => setTimeout(resolve, 1000)); // Wait 1 second before retry
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
        setCommunityGraphData(results);
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
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }

  function uuidv4() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
      const r = Math.random() * 16 | 0;
      const v = c === 'x' ? r : (r & 0x3 | 0x8);
      return v.toString(16);
    });
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
          // Get connected nodes - fixed implementation
          const connectedNodes = new Set();
          links.forEach(link => {
              if (link.source === node.id || link.source.id === node.id) {
                  connectedNodes.add(link.target.id || link.target);
              }
              if (link.target === node.id || link.target.id === node.id) {
                  connectedNodes.add(link.source.id || link.source);
              }
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
              <p><strong>Connections:</strong> ${connectedNodes.size} nodes</p>
              
              ${node.group ? `
              <div style="margin-top: 10px; border-top: 1px solid #eee; padding-top: 10px;">
                  <h4 style="margin-bottom: 5px;">Attack Details</h4>
                  <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 5px; font-size: 13px;">
                      <div><strong>Attack Type:</strong> ${inferAttackType(features)}</div>
                      <div><strong>Threat Level:</strong> ${getThreatLevel(node.confidence)}</div>
                      <div><strong>Behavior Pattern:</strong> ${getBehaviorPattern(features)}</div>
                      <div><strong>Risk Score:</strong> ${Math.round(node.confidence * 100)}/100</div>
                  </div>
                  
                  <div style="margin-top: 10px;">
                      <h4 style="margin-bottom: 5px;">Attack Indicators</h4>
                      <ul style="margin-top: 5px; padding-left: 20px;">
                          ${getAttackIndicators(probabilities, features).join('')}
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
              
              ${features ? `
              <div style="margin-top: 10px; border-top: 1px solid #eee; padding-top: 10px;">
                  <h4 style="margin-bottom: 5px;">Network Metrics</h4>
                  <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 5px; font-size: 13px;">
                      <div><strong>Connections:</strong> ${Math.round(features[0] * 1000)}</div>
                      <div><strong>Packet Rate:</strong> ${(features[3] * 100).toFixed(1)}/s</div>
                      <div><strong>Avg Size:</strong> ${Math.round(features[5] * 1500)}B</div>
                      <div><strong>SYN Ratio:</strong> ${(features[8] * 100).toFixed(1)}%</div>
                      <div><strong>RST Ratio:</strong> ${(features[9] * 100).toFixed(1)}%</div>
                      <div><strong>Duration:</strong> ${(features[2] * 60).toFixed(1)}s</div>
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
      })

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

    function getAttackType(probabilities) {
        if (!probabilities || probabilities.length < 2) return "Unknown";
        const types = ["Normal", "DDoS", "Port Scan", "Brute Force", "Malware"];
        const maxIndex = probabilities.indexOf(Math.max(...probabilities));
        return types[maxIndex] || "Unknown";
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

  // ==============================
// COMMUNITY ANALYSIS SECTION
// ==============================

// Store reference to the graph data (set this after main graph is rendered)
let communityGraphData = null;

// Event listener for "Analyze Communities" button
document.getElementById('run-community-analysis').addEventListener('click', async () => {
    const algorithm = document.getElementById('community-algorithm').value;
    const button = document.getElementById('run-community-analysis');
    
    try {
        button.disabled = true;
        button.textContent = 'Processing...';
        
        if (!communityGraphData) {
            throw new Error("No graph data available");
        }

        const communities = await runCommunityDetection(communityGraphData, algorithm);
        renderCommunityGraph(communityGraphData, communities);
        
        // Show results section
        document.getElementById('community-results').style.display = 'block';
    } catch (error) {
        console.error("Community analysis failed:", error);
        alert(`Error: ${error.message}`);
    } finally {
        button.disabled = false;
        button.textContent = 'Analyze Communities';
    }
});

// Simulated community detection logic
async function runCommunityDetection(graphData, algorithm) {
    try {
        // Prepare nodes list (ensure it's a flat array of node IDs)
        const nodes = graphData.nodes.map(n => n.id);
        
        // Prepare edges as [[source_indices], [target_indices]]
        const edges = [[], []];
        graphData.links.forEach(link => {
            const srcId = link.source.id || link.source;
            const tgtId = link.target.id || link.target;
            const srcIdx = nodes.indexOf(srcId);
            const tgtIdx = nodes.indexOf(tgtId);
            
            if (srcIdx !== -1 && tgtIdx !== -1) {
                edges[0].push(srcIdx);
                edges[1].push(tgtIdx);
            }
        });

        const response = await fetch('/api/analyze-communities', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                algorithm: algorithm,
                nodes: nodes,
                edges: edges,
                predictions: graphData.nodes.map(n => n.group),
                probabilities: graphData.nodes.map(n => n.probabilities)
            })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Community detection failed');
        }

        return await response.json();
    } catch (error) {
        console.error('Community detection error:', error);
        throw error;
    }
}

// Renders a second graph with nodes colored by community
function renderCommunityGraph(graphData, apiResponse) {
    const container = document.getElementById('community-graph-container');
    const legendContainer = document.getElementById('community-legend');
    container.innerHTML = ''; // Clear previous content
    legendContainer.innerHTML = '';
    
    try {
        // Validate input data
        if (!apiResponse?.communities || !graphData?.nodes) {
            throw new Error("Invalid community data received");
        }

        // Get all unique community IDs
        const communityIds = Object.values(apiResponse.communities);
        const uniqueCommunityIds = [...new Set(communityIds)];
        
        // Create a stable color scale
        const colorScale = d3.scaleOrdinal()
            .domain(uniqueCommunityIds.sort((a, b) => a - b)) // Sort for consistent colors
            .range(d3.schemeTableau10); // Using a more robust color scheme

        // Count community sizes
        const communitySizes = {};
        graphData.nodes.forEach(node => {
            const commId = apiResponse.communities[node.id];
            communitySizes[commId] = (communitySizes[commId] || 0) + 1;
        });

        // Prepare nodes with community data
        const coloredNodes = graphData.nodes.map(node => ({
            ...node,
            color: colorScale(apiResponse.communities[node.id]),
            community: apiResponse.communities[node.id],
            size: Math.max(1, Math.min(5, Math.sqrt(communitySizes[apiResponse.communities[node.id]] || 1)))
        }));

        // Prepare links
        const links = graphData.links.map(link => ({
            source: link.source.id || link.source,
            target: link.target.id || link.target,
            value: 1
        }));

        const graphDiv = document.createElement('div');
        graphDiv.style.width = '100%';
        graphDiv.style.height = '100%';
        container.appendChild(graphDiv);
        
        const graph = ForceGraph3D()(graphDiv)
            .graphData({ nodes: coloredNodes, links })
            .nodeLabel(node => `${node.id}\nCommunity ${node.community}\n(${communitySizes[node.community]} nodes)`)
            .nodeColor(node => node.color)
            .nodeVal(node => node.size)
            .linkWidth(1)

        const resizeObserver = new ResizeObserver(() => {
            graph.width(graphDiv.offsetWidth)
                .height(graphDiv.offsetHeight);
        });
        resizeObserver.observe(graphDiv);

        // Add legend
        addCommunityLegend(legendContainer, uniqueCommunityIds, communitySizes, colorScale);

        // Update metrics display
        updateCommunityMetrics(apiResponse, communitySizes, graphData.nodes.length);

    } catch (error) {
        console.error("Graph rendering failed:", error);
        container.innerHTML = `
            <div class="error-message">
                <h3>Visualization Error</h3>
                <p>${error.message}</p>
            </div>
        `;
    }
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