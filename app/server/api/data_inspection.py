# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

from fastapi import APIRouter

from api.inspection.catalogs import InspectionCatalogEndpoint
from api.inspection.dilirank import InspectionDiliRankEndpoint
from api.inspection.rag import InspectionRagEndpoint
from api.inspection.revisions import InspectionRevisionEndpoint
from api.inspection.sessions import InspectionSessionEndpoint
from api.inspection.terminology import InspectionTerminologyEndpoint
from api.inspection.timeline import InspectionTimelineEndpoint
from services.inspection.factory import build_data_inspection_service
from services.runtime.jobs import get_job_manager
from services.terminology.index import get_terminology_index_cache

router = APIRouter(prefix="/inspection", tags=["inspection"])

###############################################################################
def register_inspection_routes(router: APIRouter) -> None:
    service = build_data_inspection_service(get_job_manager())
    endpoints = [
        InspectionSessionEndpoint(router=router, service=service),
        InspectionRevisionEndpoint(router=router, service=service),
        InspectionTimelineEndpoint(router=router, service=service),
    ]
    for endpoint in endpoints:
        endpoint.add_routes()
    catalog_endpoint = InspectionCatalogEndpoint(router=router, service=service)
    catalog_endpoint.add_routes()
    endpoints.append(catalog_endpoint)
    endpoints.extend(
        [
            InspectionDiliRankEndpoint(router=router, service=service),
            InspectionRagEndpoint(router=router, service=service),
            InspectionTerminologyEndpoint(router=router, service=service),
        ]
    )
    for endpoint in endpoints[-3:]:
        endpoint.add_routes()
    catalog_endpoint.add_update_job_discovery_route()
    _inspection_endpoints.extend(endpoints)


_inspection_endpoints = []
register_inspection_routes(router)


###############################################################################
def rebind_database_services() -> None:
    # The fuzzy index is keyed by terminology revision, which is per database.
    get_terminology_index_cache().invalidate()
    service = build_data_inspection_service(get_job_manager())
    for endpoint in _inspection_endpoints:
        endpoint.service = service
