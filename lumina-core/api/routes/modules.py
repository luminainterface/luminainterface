from fastapi import APIRouter
from typing import List, Dict
import os
import json

router = APIRouter()

class ModuleInfo(BaseModel):
    name: str
    version: str
    description: str
    status: str
    endpoints: List[Dict[str, str]]

@router.get("/modules", response_model=List[ModuleInfo])
async def list_modules():
    modules = []
    modules_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "..", "modules")
    
    for module_name in os.listdir(modules_dir):
        manifest_path = os.path.join(modules_dir, module_name, "manifest.json")
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path) as f:
                    module_data = json.load(f)
                    modules.append(ModuleInfo(
                        name=module_name,
                        version=module_data.get("version", "0.1.0"),
                        description=module_data.get("description", ""),
                        status="healthy",  # TODO: Implement health check
                        endpoints=module_data.get("endpoints", [])
                    ))
            except Exception as e:
                print(f"Error loading module {module_name}: {e}")
    
    return modules 