# VibeVoice Complete Setup Guide

This guide provides step-by-step instructions for setting up and running the VibeVoice TTS API server with proper virtual environment configuration.

## Prerequisites

- **Python 3.9+** installed on your system
- **NVIDIA GPU** with CUDA support (for GPU acceleration)
- **Git** for cloning the repository
- **CUDA Toolkit** (recommended: CUDA 12.1 or compatible version)

## Step-by-Step Setup

### 1. Clone the VibeVoice Repository

```bash
git clone https://github.com/microsoft/VibeVoice
cd VibeVoice
```

### 2. Create a Python Virtual Environment

Creating a virtual environment ensures isolated dependencies and avoids conflicts with system packages.

**On Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\activate.ps1
```

**On Linux/Mac:**
```bash
python -m venv venv
source venv/bin/activate
```

> [!NOTE]
> After activation, you should see `(venv)` prefix in your terminal prompt.

### 3. Install Dependencies

#### Option A: Install All Dependencies (Recommended)

Install the package in editable mode with all dependencies:

```bash
pip install -e .
```

This will install all required packages from `pyproject.toml` including:
- `fastapi` - Web framework
- `uvicorn[standard]` - ASGI server
- `torch` - Deep learning framework
- `transformers` - Hugging Face library
- `accelerate` - Training acceleration
- `diffusers` - Diffusion models
- `gradio` - UI components
- And other dependencies

#### Option B: Manual Installation (If Option A Fails)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install fastapi uvicorn[standard] transformers accelerate diffusers gradio
pip install numpy scipy librosa ml-collections absl-py av aiortc pydantic
```

### 4. Install CUDA-Enabled PyTorch (Critical for GPU Support)

> [!WARNING]
> The default `pip install` may install CPU-only PyTorch. For GPU acceleration, you **must** install CUDA-enabled PyTorch.

**Uninstall CPU-only version (if already installed):**
```bash
pip uninstall -y torch torchvision torchaudio
```

**Install CUDA-enabled PyTorch:**

For CUDA 12.1:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

For CUDA 11.8:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Verify CUDA is available:**
```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

Expected output:
```
PyTorch version: 2.5.1+cu121
CUDA available: True
```

### 5. Download Voice Presets

Ensure voice preset files are available in the `voices/streaming_model/` directory. These `.pt` files contain pre-trained voice embeddings.

If not present, download them from the VibeVoice repository or model hub.

### 6. Start the API Server

**Basic usage:**
```bash
python demo/server.py --model microsoft/VibeVoice-Realtime-0.5B --device cuda --port 8888
```

**For the 1.5B model:**
```bash
python demo/server.py --model microsoft/VibeVoice-1.5B --device cuda --port 8888
```

**Available command-line options:**
- `--model` - Model path or HuggingFace repo ID (default: `microsoft/VibeVoice-Realtime-0.5B`)
- `--device` - Device to run inference on: `cuda`, `cpu`, or `mps` (default: `cuda`)
- `--port` - Port to run the server on (default: `8888`)
- `--host` - Host to bind the server to (default: `0.0.0.0`)
- `--inference-steps` - Number of inference steps (default: `5`)

### 7. Verify Server is Running

Once started, the server will be available at `http://localhost:8888`.

**Available API endpoints:**
- `GET /api/health` - Health check endpoint
- `GET /api/config` - Get available voices and configuration
- `POST /api/synthesize` - One-shot synthesis (returns complete audio)
- `WS /api/stream` - WebSocket streaming audio endpoint

**Test the health endpoint:**
```bash
curl http://localhost:8888/api/health
```

Expected response:
```json
{"status": "ok", "timestamp": "2025-12-17 22:45:00.000"}
```

## Troubleshooting

### Error: `ModuleNotFoundError: No module named 'fastapi'`

**Cause**: Dependencies not installed in the virtual environment.

**Solution**:
```bash
.\venv\Scripts\activate.ps1  # Windows
# or
source venv/bin/activate     # Linux/Mac

pip install -e .
```

### Error: `AssertionError: Torch not compiled with CUDA enabled`

**Cause**: CPU-only PyTorch installed instead of CUDA-enabled version.

**Solution**:
```bash
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Verify CUDA:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### Error: `ImportError: flash_attn seems to be not installed`

**Note**: This is expected and not a critical error. The server automatically falls back to SDPA attention implementation, which works fine.

If you want to install Flash Attention 2 for potential performance improvements:
```bash
pip install flash-attn --no-build-isolation
```

### Error: `RuntimeError: Voices directory not found`

**Cause**: Voice preset files are missing.

**Solution**: Ensure voice `.pt` files are present in `voices/streaming_model/` directory.

### Server starts but model fails to load

**Possible causes**:
1. Insufficient GPU memory - Try the smaller 0.5B model instead of 1.5B
2. CUDA out of memory - Reduce batch size or use CPU mode
3. Model not downloaded - First run downloads from HuggingFace (requires internet)

**Solution for GPU memory issues**:
```bash
# Use CPU mode (slower but works without GPU)
python demo/server.py --model microsoft/VibeVoice-Realtime-0.5B --device cpu --port 8888
```

## Best Practices

1. **Always activate the virtual environment** before running the server:
   ```bash
   .\venv\Scripts\activate.ps1  # Windows
   source venv/bin/activate     # Linux/Mac
   ```

2. **Use GPU acceleration** for better performance (requires CUDA-enabled PyTorch)

3. **Start with the smaller model** (0.5B) for testing, then upgrade to 1.5B if needed

4. **Monitor GPU memory** usage with `nvidia-smi` to ensure sufficient resources

5. **Keep dependencies updated** but be aware of compatibility:
   ```bash
   pip install --upgrade pip
   pip install -e . --upgrade
   ```

## Quick Reference

**Activate virtual environment:**
```bash
.\venv\Scripts\activate.ps1  # Windows PowerShell
source venv/bin/activate     # Linux/Mac
```

**Start server (GPU):**
```bash
python demo/server.py --model microsoft/VibeVoice-Realtime-0.5B --device cuda --port 8888
```

**Start server (CPU):**
```bash
python demo/server.py --model microsoft/VibeVoice-Realtime-0.5B --device cpu --port 8888
```

**Check CUDA availability:**
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

**Deactivate virtual environment:**
```bash
deactivate
```

## Additional Resources

- [VibeVoice GitHub Repository](https://github.com/microsoft/VibeVoice)
- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Transformers Documentation](https://huggingface.co/docs/transformers/)

---

**Need help?** Check the [GitHub Issues](https://github.com/microsoft/VibeVoice/issues) or refer to the troubleshooting section above.
