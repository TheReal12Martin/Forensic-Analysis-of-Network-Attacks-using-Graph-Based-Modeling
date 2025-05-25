import os
import shutil
import time
import tempfile
import traceback
import uuid
from typing import Optional
from fastapi import FastAPI, Form, Header, UploadFile, File, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import networkx as nx
import numpy as np
from pydantic import BaseModel
import torch
from .pcap_processor import PCAPProcessor
from .classifier import NetworkAttackClassifier
from torch_geometric.data import Data
from .graph_algorithms import GraphAnalyzer
import asyncio
from pathlib import Path
from starlette.formparsers import MultiPartParser
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
import aiofiles

app = FastAPI(
    max_upload_size=10 * 1024 * 1024 * 1024,  # 10GB
    debug=True
)

# Simplified CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

UPLOAD_DIR = Path("upload_chunks")
UPLOAD_DIR.mkdir(exist_ok=True)

frontend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
print(f"Static files path: {frontend_dir}")  # Debug output

# Mount static files
app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

# Initialize system
def init_system():
    torch.set_num_threads(4)
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['MKL_NUM_THREADS'] = '4'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    torch.backends.cudnn.benchmark = True
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = os.path.join(os.path.dirname(__file__), "best_model.pt")
    return {
        'processor': PCAPProcessor(),
        'classifier': NetworkAttackClassifier(model_path, device=device),
        'device': device
    }

system = init_system()
system['graph_analyzer'] = GraphAnalyzer()

@app.get("/")
async def serve_frontend():
    return FileResponse(os.path.join(frontend_dir, "index.html"))

# api.py - updated /api/chunk endpoint
@app.post("/api/chunk")
async def upload_chunk(
    file: UploadFile = File(...),
    chunk_index: int = Form(...),
    file_id: str = Form(...)
):
    try:
        chunk_folder = UPLOAD_DIR / file_id
        chunk_folder.mkdir(parents=True, exist_ok=True)
        chunk_path = chunk_folder / f"{chunk_index}"
        
        async with aiofiles.open(chunk_path, "wb") as buffer:
            while content := await file.read(500 * 1024 * 1024):  # 1MB chunks
                await buffer.write(content)
                
        return {"status": "ok", "chunk": chunk_index}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    

class MergeRequest(BaseModel):
    file_id: str
    filename: str
    total_chunks: int
    max_packets: int

# api.py - updated merge endpoint
@app.post("/api/merge")
async def merge_chunks(req: MergeRequest):
    try:
        chunk_folder = UPLOAD_DIR / req.file_id
        merged_file_path = chunk_folder / req.filename

        # Verify all chunks exist
        missing_chunks = []
        for i in range(req.total_chunks):
            chunk_path = chunk_folder / str(i)
            if not chunk_path.exists():
                missing_chunks.append(i)
        
        if missing_chunks:
            raise HTTPException(
                status_code=400,
                detail=f"Missing chunks: {missing_chunks}"
            )

        # Merge chunks
        async with aiofiles.open(merged_file_path, 'wb') as outfile:
            for i in range(req.total_chunks):
                chunk_path = chunk_folder / str(i)
                async with aiofiles.open(chunk_path, 'rb') as infile:
                    while content := await infile.read(1024 * 1024):  # 1MB chunks
                        await outfile.write(content)
                # Remove the chunk after merging
                try:
                    os.unlink(chunk_path)
                except:
                    pass

        # Process the merged file
        try:
            results = await process_pcap_file(str(merged_file_path), req.filename, req.max_packets)
            return results
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Processing failed: {str(e)}"
            )
        finally:
            # Clean up merged file
            try:
                os.unlink(merged_file_path)
                os.rmdir(chunk_folder)
            except:
                pass

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Merge failed: {str(e)}"
        )
    
# api.py - add cleanup endpoint
@app.delete("/api/cleanup")
async def cleanup_upload(file_id: str):
    try:
        chunk_folder = UPLOAD_DIR / file_id
        if chunk_folder.exists():
            for file in chunk_folder.glob("*"):
                try:
                    os.unlink(file)
                except:
                    pass
            try:
                os.rmdir(chunk_folder)
            except:
                pass
        return {"status": "cleaned"}
    except Exception as e:
        raise HTTPException(500, detail=str(e))

async def process_pcap_file(file_path: str, filename: str, max_packets: int):
    start_time = time.time()
    file_size = os.path.getsize(file_path)
    
    try:
        # Validate magic number
        valid_magic_numbers = {
            b"\xa1\xb2\xc3\xd4",  # pcap
            b"\xd4\xc3\xb2\xa1",
            b"\xa1\xb2\x3c\x4d",
            b"\x4d\x3c\xb2\xa1",
            b"\x0a\x0d\x0d\x0a",  # pcapng
        }
        
        async with aiofiles.open(file_path, "rb") as f:
            magic = await f.read(4)
            if magic not in valid_magic_numbers:
                raise HTTPException(400, "Invalid PCAP file format")

        # Process the file
        raw_graph = system['processor'].process_pcap(file_path, max_packets)

        if not raw_graph or len(raw_graph['nodes']) < 2:
            raise HTTPException(400, "Not enough network nodes detected (min 2 required)")

        graph_data = Data(
            x=raw_graph['x'],
            edge_index=raw_graph['edge_index'],
            edge_attr=raw_graph.get('edge_attr'),
            y=raw_graph.get('y')
        )
        graph_data.nodes = raw_graph['nodes']  # preserve custom node info

        results = system['classifier'].classify(graph_data)
        attack_count = int(np.sum(results['predictions']))

        return {
            "meta": {
                "filename": filename,
                "processing_time": round(time.time() - start_time, 2),
                "device": str(system['device']),
                "attack_count": attack_count,
                "node_count": len(raw_graph['nodes']),
                "edge_count": raw_graph['edge_index'].size(1),
                "file_size_mb": round(file_size / (1024*1024), 2)
            },
            "nodes": results['nodes'],
            "predictions": convert_for_json(results['predictions']),
            "probabilities": convert_for_json(results['probabilities']),
            "edges": convert_for_json(raw_graph['edge_index']),
            "features": convert_for_json(raw_graph['x']) if 'x' in raw_graph else None
        }

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Unexpected server error: {e}")
    
@app.post("/api/analyze-communities")
async def analyze_communities(request: Request):
    try:
        data = await request.json()
        algorithm = data.get('algorithm', 'louvain')
        
        # Validate input data
        if 'nodes' not in data or 'edges' not in data:
            raise HTTPException(status_code=400, detail="Missing required fields: nodes or edges")
        
        # Convert to NetworkX graph
        try:
            G = nx.Graph()
            
            # Add nodes with their attributes
            if isinstance(data['nodes'], list):
                G.add_nodes_from(data['nodes'])
            elif isinstance(data['nodes'], dict):
                G.add_nodes_from(data['nodes'].items())
            
            # Add edges (assuming edges is [[source_indices], [target_indices]])
            if isinstance(data['edges'], list) and len(data['edges']) == 2:
                for src_idx, tgt_idx in zip(data['edges'][0], data['edges'][1]):
                    if src_idx < len(data['nodes']) and tgt_idx < len(data['nodes']):
                        src = data['nodes'][src_idx]
                        tgt = data['nodes'][tgt_idx]
                        G.add_edge(src, tgt)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid graph data: {str(e)}")
        
        # Run community detection
        result = system['graph_analyzer'].detect_communities(
            {'nodes': list(G.nodes()), 'edges': list(G.edges())},
            algorithm=algorithm
        )
        
        # Calculate metrics on the original partition
        metrics = system['graph_analyzer'].get_community_metrics(
            G, result['partition']
        )

        predictions = {node: pred for node, pred in zip(data['nodes'], data.get('predictions', []))}
        
        security_insights = {
            'attack_campaigns': system['graph_analyzer'].detect_attack_campaigns(
                G, result['partition'], predictions),
            'lateral_movement': system['graph_analyzer'].detect_lateral_movement(
                G, result['partition']),
            'command_control': system['graph_analyzer'].detect_command_control(
                G, result['partition']),
            # Add other detection methods here
        }
        
        return {
            'status': 'success',
            'algorithm': algorithm,
            'communities': result['partition'],
            'metrics': metrics,
            'modularity': float(result.get('modularity', 0)),
            'security_insights': security_insights
        }
        
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

def convert_for_json(obj):
    import torch, numpy as np
    if torch.is_tensor(obj):
        return obj.cpu().numpy().tolist()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    elif isinstance(obj, dict):
        return {k: convert_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_for_json(item) for item in obj]
    return obj

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)