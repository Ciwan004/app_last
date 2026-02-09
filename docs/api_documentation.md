# Production REST API: Complete Documentation

## Overview

Production-ready FastAPI REST API for quantum portfolio optimization with guaranteed feasible solutions using the hybrid classical-quantum approach.

## Implementation

### API Structure

```
/app/api/
├── main.py          - FastAPI application entry point
├── endpoints.py     - Route handlers (/optimize, /health, /)
├── schemas.py       - Pydantic models for validation
└── __init__.py
```

### Endpoints

#### 1. `POST /api/optimize` - Run Portfolio Optimization

**Description**: Optimize portfolio using hybrid VQE + classical projection.

**Request Body** (`OptimizationRequest`):
```json
{
  "expected_returns": [0.06, 0.08, 0.12, 0.15],
  "covariance_matrix": [
    [0.04, 0.01, 0.02, 0.01],
    [0.01, 0.09, 0.03, 0.02],
    [0.02, 0.03, 0.16, 0.04],
    [0.01, 0.02, 0.04, 0.25]
  ],
  "risk_aversion": 0.5,
  "budget": 1.0,
  "bounds": [0.0, 1.0],
  "num_bins": 5,
  "penalty_coefficient": 1000.0,
  "ansatz_type": "RealAmplitudes",
  "ansatz_reps": 2,
  "max_vqe_iterations": 100,
  "optimizer": "COBYLA"
}
```

**Parameters**:
- `expected_returns` (required): Expected return for each asset
- `covariance_matrix` (required): Asset covariance matrix (symmetric, positive definite)
- `risk_aversion` (default: 0.5): Risk aversion coefficient λ (≥0)
- `budget` (default: 1.0): Total budget constraint
- `bounds` (default: [0, 1]): [min, max] weight bounds
- `num_bins` (default: 5): Discretization bins (3-11)
- `penalty_coefficient` (default: 1000): Constraint penalty
- `ansatz_reps` (default: 2): Circuit depth (1-5)
- `max_vqe_iterations` (default: 100): Max VQE iterations (10-500)

**Response** (`OptimizationResult`):
```json
{
  "weights": [0.417, 0.0, 0.542, 0.042],
  "is_feasible": true,
  "metrics": {
    "expected_return": 0.0962,
    "volatility": 0.2560,
    "variance": 0.0655,
    "sharpe_ratio": 0.2979,
    "risk_free_rate": 0.02
  },
  "vqe_iterations": 100,
  "vqe_convergence_energy": 12870.4,
  "vqe_execution_time": 28.98,
  "projection_distance": 0.361,
  "vqe_weights": [0.625, 0.0, 0.75, 0.25],
  "vqe_was_feasible": false,
  "classical_weights": [0.379, 0.217, 0.193, 0.211],
  "classical_objective": -0.027032,
  "hybrid_objective": -0.024156,
  "objective_gap": 0.002876,
  "approximation_ratio": 0.894,
  "num_qubits": 20,
  "num_bins": 5
}
```

**Status Codes**:
- `200`: Success
- `400`: Invalid input (validation error)
- `500`: Internal error

**Example cURL**:
```bash
curl -X POST http://localhost:8002/api/optimize \
  -H "Content-Type: application/json" \
  -d @portfolio_data.json
```

#### 2. `GET /api/health` - Health Check

**Description**: Check system health and component status.

**Response** (`HealthResponse`):
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "backend": "aer_simulator",
  "backend_available": true,
  "components": {
    "classical_solver": "ok",
    "quantum_mapper": "ok",
    "vqe_optimizer": "ok",
    "hybrid_solver": "ok"
  }
}
```

**Status Values**:
- `healthy`: All components operational
- `degraded`: Some components failing
- `unhealthy`: System not operational

**Example**:
```bash
curl http://localhost:8002/api/health
```

#### 3. `GET /api/` - API Information

**Description**: Get API metadata and available endpoints.

**Response**:
```json
{
  "name": "Quantum Portfolio Optimization API",
  "version": "1.0.0",
  "description": "Hybrid classical-quantum portfolio optimization using VQE",
  "endpoints": {
    "POST /api/optimize": "Run portfolio optimization",
    "GET /api/health": "Health check",
    "GET /api/docs": "OpenAPI documentation",
    "GET /api/redoc": "ReDoc documentation"
  },
  "documentation": "/api/docs"
}
```

### Interactive Documentation

**Swagger UI**: `http://localhost:8002/api/docs`
- Interactive API explorer
- Try endpoints directly
- View schemas and examples

**ReDoc**: `http://localhost:8002/api/redoc`
- Clean, readable documentation
- Schema browser
- Code examples

## Features

### 1. Request Validation

**Automatic validation using Pydantic**:
- Type checking (lists, floats, ints)
- Range validation (e.g., `risk_aversion >= 0`)
- Dimension checking (covariance matrix size)
- Symmetry validation (covariance)

**Example validation errors**:
```json
{
  "detail": [
    {
      "loc": ["body", "risk_aversion"],
      "msg": "ensure this value is greater than or equal to 0",
      "type": "value_error.number.not_ge"
    }
  ]
}
```

### 2. Error Handling

**Three-tier error handling**:

**Tier 1: Validation Errors (400)**
- Pydantic schema violations
- Data dimension mismatches
- Invalid parameter ranges

**Tier 2: Business Logic Errors (400)**
- Non-positive definite covariance
- Infeasible constraints
- Invalid portfolio data

**Tier 3: Internal Errors (500)**
- VQE convergence failures
- Backend errors
- Unexpected exceptions

**Error Response Format**:
```json
{
  "error": "ValidationError",
  "message": "Covariance matrix must be symmetric",
  "details": {
    "row": 0,
    "col": 1,
    "difference": 0.05
  }
}
```

### 3. Performance

**Optimization Time Breakdown**:
| Component | Time (3 bins, 12 qubits) | Time (5 bins, 20 qubits) |
|-----------|--------------------------|---------------------------|
| Classical baseline | ~0.1s | ~0.1s |
| VQE (50 iter) | ~2-3s | ~5-10s |
| VQE (100 iter) | ~5-6s | ~15-30s |
| Projection | <0.1s | <0.1s |
| **Total** | **2-6s** | **5-30s** |

**Scalability**:
- Linear in VQE iterations
- Exponential in qubits (simulation)
- Constant projection time

### 4. Configuration

**Quantum Parameters**:
- `num_bins`: 3 (fast), 5 (balanced), 7-11 (precise)
- `ansatz_reps`: 1-2 (NISQ), 3-5 (better quality)
- `max_vqe_iterations`: 50 (quick), 100 (standard), 200+ (thorough)

**Trade-offs**:
| Configuration | Speed | Quality | Qubits |
|---------------|-------|---------|--------|
| Fast (3 bins, 50 iter) | ★★★★★ | ★★☆☆☆ | 12 |
| Balanced (5 bins, 100 iter) | ★★★☆☆ | ★★★★☆ | 20 |
| Precise (7 bins, 200 iter) | ★☆☆☆☆ | ★★★★★ | 28 |

## Deployment

### Local Development

```bash
cd /app
uvicorn api.main:app --host 0.0.0.0 --port 8002 --reload
```

### Production

```bash
# With multiple workers
uvicorn api.main:app \
  --host 0.0.0.0 \
  --port 8002 \
  --workers 4 \
  --log-level info

# With gunicorn (recommended)
gunicorn api.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8002 \
  --timeout 120
```

### Docker

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8002
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8002"]
```

### Environment Variables

```bash
# Optional configuration
export API_HOST=0.0.0.0
export API_PORT=8002
export LOG_LEVEL=info
export QUANTUM_BACKEND=aer_simulator
export MAX_WORKERS=4
```

## Usage Examples

### Python Client

```python
import requests

# API endpoint
url = "http://localhost:8002/api/optimize"

# Portfolio data
data = {
    "expected_returns": [0.06, 0.08, 0.12, 0.15],
    "covariance_matrix": [
        [0.04, 0.01, 0.02, 0.01],
        [0.01, 0.09, 0.03, 0.02],
        [0.02, 0.03, 0.16, 0.04],
        [0.01, 0.02, 0.04, 0.25]
    ],
    "risk_aversion": 0.5,
    "num_bins": 5,
    "max_vqe_iterations": 100
}

# Send request
response = requests.post(url, json=data, timeout=120)

# Parse response
if response.status_code == 200:
    result = response.json()
    print(f"Optimal weights: {result['weights']}")
    print(f"Sharpe ratio: {result['metrics']['sharpe_ratio']:.4f}")
    print(f"Feasible: {result['is_feasible']}")
else:
    print(f"Error: {response.json()}")
```

### JavaScript/TypeScript

```typescript
const response = await fetch('http://localhost:8002/api/optimize', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    expected_returns: [0.06, 0.08, 0.12, 0.15],
    covariance_matrix: [
      [0.04, 0.01, 0.02, 0.01],
      [0.01, 0.09, 0.03, 0.02],
      [0.02, 0.03, 0.16, 0.04],
      [0.01, 0.02, 0.04, 0.25]
    ],
    risk_aversion: 0.5,
    num_bins: 5
  })
});

const result = await response.json();
console.log('Optimal weights:', result.weights);
console.log('Sharpe ratio:', result.metrics.sharpe_ratio);
```

## Monitoring

### Metrics to Track

**Performance**:
- Request latency (p50, p95, p99)
- VQE iteration count
- Projection distance

**Quality**:
- Feasibility rate (should be 100%)
- Approximation ratio vs classical
- Sharpe ratio distribution

**Health**:
- Backend availability
- Error rate
- Component status

### Logging

**Request logs**:
```
INFO: POST /api/optimize - 200 OK (28.5s)
INFO: VQE iterations: 100, energy: 12870.4
INFO: Projection distance: 0.361
```

**Error logs**:
```
ERROR: Optimization failed: ValidationError
ERROR: Details: Covariance matrix not positive definite
```

## Testing

### API Component Tests

```bash
# Direct component testing (no HTTP)
python test_api_direct.py
```

**Output**:
```
✓ Schemas: PASS
✓ Backend: PASS
✓ Optimization: PASS
```

### Full API Tests

```bash
# Start API first
uvicorn api.main:app --host 0.0.0.0 --port 8002 &

# Run tests
python test_api.py
```

**Tests**:
1. Root endpoint
2. Health check
3. Valid optimization request
4. Invalid data handling

## Production Checklist

- [x] Request validation (Pydantic schemas)
- [x] Error handling (3-tier)
- [x] Health check endpoint
- [x] CORS configuration
- [x] Logging (startup/shutdown)
- [x] Documentation (Swagger + ReDoc)
- [x] Timeout handling
- [ ] Rate limiting (add in production)
- [ ] Authentication (add if needed)
- [ ] Result caching (add for performance)
- [ ] Async job queue (for long-running optimizations)

## Future Enhancements

### 1. Async Job Queue

For long-running optimizations:
```python
@router.post("/optimize/async")
async def optimize_async(request: OptimizationRequest):
    job_id = start_background_job(request)
    return {"job_id": job_id, "status": "queued"}

@router.get("/results/{job_id}")
async def get_results(job_id: str):
    return get_job_result(job_id)
```

### 2. Result Caching

Cache results for identical requests:
```python
from functools import lru_cache

@lru_cache(maxsize=100)
def optimize_cached(request_hash):
    return optimize(request)
```

### 3. Batch Optimization

Optimize multiple portfolios in one request:
```python
@router.post("/optimize/batch")
async def optimize_batch(requests: List[OptimizationRequest]):
    results = await asyncio.gather(*[optimize(r) for r in requests])
    return results
```

### 4. WebSocket Updates

Real-time VQE progress:
```python
@router.websocket("/ws/optimize")
async def optimize_ws(websocket: WebSocket):
    await websocket.accept()
    # Stream VQE iterations
    for iteration, energy in vqe_optimize():
        await websocket.send_json({"iteration": iteration, "energy": energy})
```

## Conclusion

**Production-ready REST API** with:
- ✅ Comprehensive validation
- ✅ Robust error handling
- ✅ Interactive documentation
- ✅ Health monitoring
- ✅ Hybrid solver integration
- ✅ Guaranteed feasible outputs

**Status**: Ready for deployment and production use.
