import asyncio
import json
import sys
import logging
import os
from typing import Dict, Any, Optional
from mcp.server.fastmcp import FastMCP

from fhir_tools import UniversalFhirMcpServer
from datetime import datetime

# Disable authentication for MCP server
os.environ["DANGEROUSLY_OMIT_AUTH"] = "true"

logger = logging.getLogger(__name__)

# Create the FastMCP server
mcp = FastMCP(name="Universal FHIR MCP Server")
fhir_server = UniversalFhirMcpServer()

@mcp.tool()
async def switch_server(server_name: str) -> Any:
    """Switch to a different FHIR server"""
    return await fhir_server.switch_server(server_name)

@mcp.tool()
async def test_server_connectivity(server_name: str) -> Any:
    """Test connectivity to a FHIR server"""
    return await fhir_server.test_server_connectivity(server_name)

@mcp.tool()
async def find_patient(search_criteria: Dict[str, Any], servers: Optional[list[str]] = None) -> Any:
    """Search for a patient"""
    return await fhir_server.find_patient(search_criteria=search_criteria, servers=servers)

@mcp.tool()
async def get_patient_observations(patient_id: str, servers: Optional[list[str]] = None, category: Optional[str] = None, code: Optional[str] = None, date_from: Optional[str] = None, date_to: Optional[str] = None, count: Optional[str] = None) -> Any:
    """Get patient's observations"""
    return await fhir_server.get_patient_observations(patient_id=patient_id, servers=servers, category=category, code=code, date_from=date_from, date_to=date_to, count=count)

@mcp.tool()
async def get_patient_conditions(patient_id: str, servers: Optional[list[str]] = None, clinical_status: Optional[str] = None, count: Optional[str] = None) -> Any:
    """Get patient's conditions"""
    return await fhir_server.get_patient_conditions(patient_id=patient_id, servers=servers, clinical_status=clinical_status, count=count)

@mcp.tool()
async def get_patient_medications(patient_id: str, servers: Optional[list[str]] = None, status: Optional[str] = None, count: Optional[str] = None) -> Any:
    """Get patient's medications"""
    return await fhir_server.get_patient_medications(patient_id=patient_id, servers=servers, status=status, count=count)

@mcp.tool()
async def get_patient_encounters(patient_id: str, servers: Optional[list[str]] = None, status: Optional[str] = None, count: Optional[str] = None) -> Any:
    """Get patient's encounters"""
    return await fhir_server.get_patient_encounters(patient_id=patient_id, servers=servers, status=status, count=count)

@mcp.tool()
async def get_patient_allergies(patient_id: str, servers: Optional[list[str]] = None, clinical_status: Optional[str] = None, count: Optional[str] = None) -> Any:
    """Get patient's allergies"""
    return await fhir_server.get_patient_allergies(patient_id=patient_id, servers=servers, clinical_status=clinical_status, count=count)

@mcp.tool()
async def get_patient_procedures(patient_id: str, servers: Optional[list[str]] = None, status: Optional[str] = None, count: Optional[str] = None) -> Any:
    """Get patient's procedures"""
    return await fhir_server.get_patient_procedures(patient_id=patient_id, servers=servers, status=status, count=count)

@mcp.tool()
async def get_vital_signs(patient_id: str, servers: Optional[list[str]] = None, date_from: Optional[str] = None, date_to: Optional[str] = None, count: Optional[str] = None) -> Any:
    """Get patient's vital signs"""
    return await fhir_server.get_vital_signs(patient_id=patient_id, servers=servers, date_from=date_from, date_to=date_to, count=count)

# @mcp.tool()
# async def get_lab_results(patient_id: str, servers: Optional[list[str]] = None, code: Optional[str] = None, date_from: Optional[str] = None, date_to: Optional[str] = None, count: Optional[str] = None) -> Any:
#     """Get patient's laboratory results"""
#     return await fhir_server.get_lab_results(patient_id=patient_id, servers=servers, code=code, date_from=date_from, date_to=date_to, count=count)

# @mcp.tool()
# async def clinical_query(resource_type: str, query_params: Dict[str, Any], servers: Optional[list[str]] = None) -> Any:
#     """Execute custom FHIR queries"""
#     return await fhir_server.clinical_query(resource_type=resource_type, query_params=query_params, servers=servers)

# @mcp.tool()
# async def diagnose_fhir_server(server_name: Optional[str] = None) -> Any:
#     """Diagnose FHIR server capabilities"""
#     return await fhir_server.diagnose_fhir_server(server_name)

@mcp.tool()
def list_available_servers() -> Any:
    """List all available FHIR servers"""
    return fhir_server.list_available_servers()

# @mcp.tool()
# def get_server_registry() -> Any:
#     """Get the complete server registry"""
#     return fhir_server.get_server_registry()

@mcp.tool()
async def extract_clinical_keywords(text: str) -> Any:
    """Extract clinical keywords from text"""
    return await fhir_server.extract_clinical_keywords(text)

# @mcp.tool
# async def map_to_fhir_codes(clinical_data: Dict[str, Any]) -> Dict[str, Any]:
#     """Map clinical terms to FHIR codes using actual server data with fuzzy matching (fast version with clean output)"""
#     return await fhir_server.map_to_fhir_codes(
#         clinical_data=clinical_data,
#     )

@mcp.tool()
async def map_to_fhir_codes_fast(clinical_data: Dict[str, Any], servers: Optional[list[str]] = None, similarity_threshold: Optional[float] = 0.6, max_matches: Optional[int] = 3) -> Any:
    """Map clinical terms to FHIR codes using actual server data with fuzzy matching (fast version with clean output)"""
    return await fhir_server.map_to_fhir_codes_fast(
        clinical_data=clinical_data,
        servers=servers,
        similarity_threshold=similarity_threshold,
        max_matches=max_matches
    )

# @mcp.tool(
# async def find_similar_patients(criteria: Dict[str, Any], servers: Optional[list[str]] = None, max_results: Optional[int] = None) -> Dict[str, Any]:
#     """Find patients with similar clinical profiles"""
#     return await fhir_server.find_similar_patients(criteria, servers, max_results)

@mcp.tool()
async def find_similar_patients_simple(clinical_data: Dict[str, Any], servers: Optional[list[str]] = None, max_results: Optional[int] = 10, age_tolerance: Optional[int] = 10) -> Any:
    """Find similar patients using age, gender, and condition codes from map_to_fhir_codes_fast output"""
    return await fhir_server.find_similar_patients_simple(
        clinical_data=clinical_data,
        servers=servers,
        max_results=max_results,
        age_tolerance=age_tolerance
    )


@mcp.tool()
async def get_observations_by_category(patient_id: str, category_code: str, servers: Optional[list[str]] = None, date_from: Optional[str] = None, date_to: Optional[str] = None, count: Optional[str] = None) -> Any:
    """Get patient observations by specific FHIR category (vital-signs, laboratory, imaging, exam, procedure, survey, social-history, therapy, activity)"""
    return await fhir_server.get_observations_by_category(patient_id=patient_id, category_code=category_code, servers=servers, date_from=date_from, date_to=date_to, count=count)

# @mcp.tool()
# def clear_condition_cache(server_name: Optional[str] = None) -> Any:
#     """Clear condition codes cache for specific server or all servers"""
#     fhir_server.clear_condition_cache(server_name)
#     return {"status": "success", "message": f"Cache cleared for {server_name or 'all servers'}"}

# @mcp.tool()
# def get_condition_cache_stats() -> Any:
#     """Get statistics about condition codes cache"""
#     return fhir_server.get_condition_cache_stats()

@mcp.tool()
async def get_comprehensive_patient_info(patient_id: str, server_name: Optional[str] = None) -> Any:
    """Get comprehensive patient information from a single FHIR server"""
    return await fhir_server.get_comprehensive_patient_info(patient_id=patient_id, server_name=server_name)

@mcp.tool()
def ping() -> Any:
    """Ping the MCP server to check if it is alive"""
    return {
        "status": "ok",
        "message": "MCP server is alive",
        "timestamp": datetime.utcnow().isoformat() + 'Z'
    }

async def main():
    try:
        # Initialize logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Initialize the FHIR server
        logger.info("Initializing FHIR server...")
        try:
            await fhir_server.initialize()
            logger.info("FHIR server initialized successfully")
        except Exception as init_error:
            logger.error(f"Failed to initialize FHIR server: {str(init_error)}")
            return
        
        # Test connectivity to HAPI FHIR server
        logger.info("Testing connectivity to HAPI FHIR server...")
        result = await fhir_server.test_server_connectivity("hapi_r4")
        print(json.dumps(result, indent=2))
        
        # Run the MCP server if connectivity test succeeds
        if result.get("status") == "connected":
            logger.info("Starting MCP server...")
            await mcp.run_stdio_async()
        else:
            logger.error(f"Failed to connect to HAPI FHIR server: {result.get('error')}")
            logger.error("Server details:")
            logger.error(json.dumps(result, indent=2))
    except Exception as e:
        logger.error(f"Error in main: {str(e)}", exc_info=True)
        raise
    finally:
        logger.info("Closing FHIR server connection...")
        await fhir_server.close()

if __name__ == "__main__":
    asyncio.run(main()) 