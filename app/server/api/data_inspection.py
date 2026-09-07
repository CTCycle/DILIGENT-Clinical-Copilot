from __future__ import annotations

from fastapi import APIRouter

from api.inspection.catalogs import InspectionCatalogEndpoint
from api.inspection.rag import InspectionRagEndpoint
from api.inspection.revisions import InspectionRevisionEndpoint
from api.inspection.sessions import InspectionSessionEndpoint
from api.inspection.timeline import InspectionTimelineEndpoint
from services.inspection.factory import build_data_inspection_service
from services.runtime.jobs import get_job_manager

router = APIRouter(prefix="/inspection", tags=["inspection"])

###############################################################################
def register_inspection_routes(router: APIRouter) -> None:
    service = build_data_inspection_service(get_job_manager())
    InspectionSessionEndpoint(router=router, service=service).add_routes()
    InspectionRevisionEndpoint(router=router, service=service).add_routes()
    InspectionTimelineEndpoint(router=router, service=service).add_routes()
    catalog_endpoint = InspectionCatalogEndpoint(router=router, service=service)
    catalog_endpoint.add_routes()
    InspectionRagEndpoint(router=router, service=service).add_routes()
    catalog_endpoint.add_update_job_discovery_route()


register_inspection_routes(router)
