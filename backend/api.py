import os
import time
import tempfile
from typing import Optional
from fastapi import FastAPI, Form, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import torch
from pcap_processor import PCAPProcessor
from classifier import NetworkAttackClassifier
from torch_geometric.data import Data

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

frontend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
print(f"Static files path: {frontend_dir}")  # Debug output

# Mount static files - this should be after app = FastAPI()
app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

# Initialize system (your original config)
def init_system():
    torch.set_num_threads(4)
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['MKL_NUM_THREADS'] = '4'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    torch.backends.cudnn.benchmark = True
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return {
        'processor': PCAPProcessor(),
        'classifier': NetworkAttackClassifier("best_model.pt", device=device),
        'device': device
    }

system = init_system()

@app.get("/")
async def serve_frontend():
    return FileResponse(os.path.join(frontend_dir, "index.html"))

@app.post("/api/analyze")
async def analyze_pcap(file: UploadFile = File(...), max_packets: Optional[int] = Form(None)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pcap") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        start_time = time.time()
        raw_graph = system['processor'].process_pcap(tmp_path, max_packets=max_packets)
        
        if not raw_graph or len(raw_graph['nodes']) < 2:
            raise HTTPException(status_code=400, detail="Not enough nodes for analysis")

        results = system['classifier'].classify(Data(**raw_graph))
        
        # Ensure we're working with tensors for calculations
        predictions = results['predictions'] if torch.is_tensor(results['predictions']) else torch.tensor(results['predictions'])
        attack_count = int(torch.sum(predictions).item())

        # Convert all data to JSON-serializable formats
        def convert_for_json(obj):
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

        response_data = {
            "meta": {
                "filename": file.filename,
                "processing_time": time.time() - start_time,
                "device": str(system['device']),
                "attack_count": attack_count,
                "threshold": 0.87,  # Example - use your actual threshold
                "probability_range": [0.8531, 0.8808]  # Example - use your actual range
            },
            "nodes": results['nodes'],
            "predictions": convert_for_json(results['predictions']),
            "probabilities": convert_for_json(results['probabilities']),
            "edges": convert_for_json(raw_graph['edge_index'])
        }

        return JSONResponse(content=response_data)

    except Exception as e:
        print(f"Full error during analysis: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed at final stage: {str(e)}"
        )
    finally:
        if 'tmp_path' in locals():
            os.unlink(tmp_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)