"""Tools for listing and loading stored phantoms."""

from __future__ import annotations

import json

from agent.tools.base_tool import MRISimulationBaseTool
from agent.tools.phantom_tool import set_cached_phantom
from mri_sim.phantom_database import (
    PHANTOM_DATABASE_DIR,
    list_phantom_database,
    load_phantom_dataset,
)


class ListPhantomDatabaseTool(MRISimulationBaseTool):
    name = "list_phantom_database"
    description = "List available phantoms in the local phantom database."

    def _run(self, query: str) -> str:
        if not PHANTOM_DATABASE_DIR.exists():
            return json.dumps({"status": "error", "message": f"Database path does not exist: {PHANTOM_DATABASE_DIR}"})

        phantoms = list_phantom_database(PHANTOM_DATABASE_DIR)
        return json.dumps({"status": "success", "phantoms": phantoms, "count": len(phantoms)}, ensure_ascii=False)


class LoadPhantomFromDatabaseTool(MRISimulationBaseTool):
    name = "load_phantom_from_database"
    description = "Load a phantom from the local phantom database. JSON params: phantom_name."

    def _run(self, query: str) -> str:
        params = json.loads(query or "{}")
        phantom_name = params.get("phantom_name")
        if not phantom_name:
            return json.dumps({"status": "error", "message": "phantom_name is required."})

        try:
            dataset = load_phantom_dataset(str(phantom_name), PHANTOM_DATABASE_DIR)
            phantom = dataset.build_phantom()
        except Exception as exc:
            return json.dumps({"status": "error", "message": str(exc)}, ensure_ascii=False)

        set_cached_phantom(phantom, dataset.rho, dataset.t1, dataset.t2)

        return json.dumps(
            {
                "status": "success",
                "phantom_name": phantom_name,
                "shape": [phantom.Nz, phantom.Nx, phantom.Ny],
                "fov": [phantom.fov_x, phantom.fov_y],
                "slice_thickness": phantom.slice_thickness,
                "rx_coils": phantom.RxCoilNum,
                "tx_coils": phantom.TxCoilNum,
                "optional_arrays": sorted(dataset.optional_arrays),
            },
            ensure_ascii=False,
        )
