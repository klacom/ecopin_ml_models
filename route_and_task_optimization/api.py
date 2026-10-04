from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from task_generation_model import TaskGenerationModel

app = FastAPI(title="Ecopin Task Generation API")

class StartLocation(BaseModel):
    lat: float
    lon: float

class TargetReport(BaseModel):
    id: str
    lat: float
    lon: float
    base_severity: float
    created_at: str

class GenerateRequest(BaseModel):
    aging_factor: float = 1.5
    max_escalation_cap: float = 10.0
    max_detour_minutes: float = 5.0
    max_detour_time_per_task: float = 15.0
    start_location: StartLocation
    clusters: List[TargetReport]
    individual_reports: List[TargetReport]

@app.post("/generate_tasks")
async def generate_tasks(request: GenerateRequest):
    try:
        model = TaskGenerationModel(
            aging_factor=request.aging_factor,
            max_escalation_cap=request.max_escalation_cap,
            max_detour_minutes=request.max_detour_minutes,
            max_detour_time_per_task=request.max_detour_time_per_task
        )

        from datetime import datetime
        
        def parse_date(date_str: str) -> datetime:
            # Assuming standard ISO format from web client
            return datetime.fromisoformat(date_str.replace('Z', '+00:00'))

        # Prepare data for model
        clusters = []
        for c in request.clusters:
            clusters.append({
                'id': c.id,
                'lat': c.lat,
                'lon': c.lon,
                'base_severity': c.base_severity,
                'created_at': parse_date(c.created_at)
            })
            
        reports = []
        for r in request.individual_reports:
            reports.append({
                'id': r.id,
                'lat': r.lat,
                'lon': r.lon,
                'base_severity': r.base_severity,
                'created_at': parse_date(r.created_at)
            })
            
        start_loc = {
            'lat': request.start_location.lat,
            'lon': request.start_location.lon
        }
        
        tasks = model.generate_tasks(clusters, reports, start_loc)
        return {"tasks": tasks}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
