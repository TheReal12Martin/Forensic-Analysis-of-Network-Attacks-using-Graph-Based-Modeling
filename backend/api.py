import os
import time
import tempfile
import traceback
from typing import Optional
from fastapi import FastAPI, Form, UploadFile, File, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import torch
from .pcap_processor import PCAPProcessor
from .classifier import NetworkAttackClassifier
from torch_geometric.data import Data
import asyncio

app = FastAPI(
    max_upload_size=10 * 1024 * 1024 * 1024,  # 10GB max upload size
    timeout=3600  # 1 hour timeout
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

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

# Timeout middleware for large files
@app.middleware("http")
async def timeout_middleware(request: Request, call_next):
    try:
        # Set timeout to 1 hour for large files
        timeout = 3600 if request.url.path == "/api/analyze" else 60
        return await asyncio.wait_for(call_next(request), timeout=timeout)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Processing timeout")

@app.get("/")
async def serve_frontend():
    return FileResponse(os.path.join(frontend_dir, "index.html"))

@app.post("/api/analyze")
async def analyze_pcap(
    request: Request,
    file: UploadFile = File(...),
    max_packets: str = Form("100000")
):
    import time, os, tempfile, traceback
    start_time = time.time()
    temp_dir = temp_path = None
    file_size = 0

    print(f"[DEBUG] Start Analyze")

    try:
        # === 1. BASIC INPUT VALIDATION ===
        print("[DEBUG] analyze_pcap() called")
        if not file or not file.filename:
            raise HTTPException(400, "No file provided")

        print(f"[DEBUG] Received file: {file.filename}")
        print(f"[DEBUG] file.content_type: {file.content_type}")

        try:
            max_packets = int(max_packets)
            if max_packets < 1000:
                max_packets = 1000
        except ValueError:
            raise HTTPException(400, "max_packets must be an integer")

        print("[DEBUG] Passed Max Packets Validation")

        # === 2. CREATE TEMP FILE ===
        temp_dir = tempfile.mkdtemp(prefix="pcap_")
        temp_path = os.path.join(temp_dir, "upload.pcap")
        print(f"[DEBUG] Temp file real path: {temp_path}")

        # === 3. STREAM FILE TO DISK ===
        try:
            with open(temp_path, "wb") as f:
                chunk_count = 0
                while True:
                    chunk = await file.read(1024 * 1024)
                    if not chunk:
                        print(f"[DEBUG] EOF after {chunk_count} chunks, {file_size} bytes")
                        break
                    f.write(chunk)
                    file_size += len(chunk)
                    chunk_count += 1
                    if chunk_count % 100 == 0:
                        print(f"[DEBUG] Read {chunk_count} chunks ({file_size / (1024*1024):.2f} MB)")
        except Exception as e:
            raise HTTPException(500, f"Failed to save file: {e}")

        print(f"[DEBUG] File exists after write? {os.path.exists(temp_path)}")
        print(f"[DEBUG] File size on disk: {os.path.getsize(temp_path)} bytes")

        if os.path.getsize(temp_path) == 0:
            raise HTTPException(400, "Empty file received")

        # === 4. VALIDATE MAGIC NUMBER ===
        valid_magic_numbers = {
            b"\xa1\xb2\xc3\xd4",  # pcap
            b"\xd4\xc3\xb2\xa1",
            b"\xa1\xb2\x3c\x4d",
            b"\x4d\x3c\xb2\xa1",
            b"\x0a\x0d\x0d\x0a",  # pcapng
        }
        with open(temp_path, "rb") as f:
            magic = f.read(4)
            print(f"[DEBUG] PCAP magic: {magic.hex()}")
            if magic not in valid_magic_numbers:
                raise HTTPException(400, "Invalid PCAP file format")

        # === 5. PARSE FILE ===
        print(f"=== PROCESSING PCAP: {os.path.basename(temp_path)} ===")
        raw_graph = system['processor'].process_pcap(temp_path, max_packets)

        if not raw_graph or len(raw_graph['nodes']) < 2:
            raise HTTPException(400, "Not enough network nodes detected (min 2 required)")

        from torch_geometric.data import Data
        graph_data = Data(
            x=raw_graph['x'],
            edge_index=raw_graph['edge_index'],
            edge_attr=raw_graph.get('edge_attr'),
            y=raw_graph.get('y')
        )
        graph_data.nodes = raw_graph['nodes']  # preserve custom node info

        results = system['classifier'].classify(graph_data)

        # === 6. FORMAT RESPONSE ===
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

        attack_count = int(np.sum(results['predictions']))

        return JSONResponse(content={
            "meta": {
                "filename": file.filename,
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
        })

    except HTTPException as he:
        raise he
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Unexpected server error: {e}")
    finally:
        if temp_path and os.path.exists(temp_path):
            try: os.unlink(temp_path)
            except: pass
        if temp_dir and os.path.exists(temp_dir):
            try: os.rmdir(temp_dir)
            except: pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)