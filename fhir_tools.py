#!/usr/bin/env python3

import asyncio
import aiohttp
import json
import sys
import logging
import os
import re
import nest_asyncio
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from urllib.parse import urljoin, urlparse
import base64
from dotenv import load_dotenv

# Set up logging to stderr only
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

# Add file handler for persistent logging
log_dir = os.path.join(os.path.dirname(__file__), 'logs')
os.makedirs(log_dir, exist_ok=True)

# Add handler for mcp_dev_inspector.log
inspector_handler = logging.FileHandler(os.path.join(log_dir, 'mcp_dev_inspector.log'))
inspector_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
inspector_handler.setLevel(logging.DEBUG)  # Set to DEBUG to capture all levels
logger.addHandler(inspector_handler)

# Load environment variables from .env
load_dotenv()
logger.info("Starting FHIR Tools with environment configuration")

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

def get_or_create_eventloop():
    try:
        return asyncio.get_event_loop()
    except RuntimeError as ex:
        if "There is no current event loop in thread" in str(ex):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop
        raise

class FhirServerRegistry:
    """Enterprise-grade FHIR server registry supporting all major vendors"""
    
    def __init__(self):
        self.servers = self._initialize_comprehensive_servers()
        
    def _initialize_comprehensive_servers(self) -> Dict[str, Dict[str, Any]]:
        """Initialize comprehensive FHIR server registry for all major vendors"""
        return {
            # Active Servers - Local Development and Testing
            "firely_local": {
                "name": "Firely Server Local",
                "base_url": "http://localhost:9090",
                "version": "R4",
                "vendor": "Firely",
                "type": "local_development",
                "auth_type": "none",
                "description": "Local Firely FHIR server for development",
                "supported_resources": ["*"],
                "search_strategy": "standard",
                "status": "local",
                "special_params": {"_format": "json"}  # Required for bundle output
            },
            "hapi_r4": {
                "name": "HAPI FHIR R4 Public",
                "base_url": "http://hapi.fhir.org/baseR4",
                "version": "R4",
                "vendor": "HAPI",
                "type": "public_test",
                "auth_type": "none",
                "description": "Open source HAPI FHIR test server",
                "supported_resources": ["*"],
                "search_strategy": "standard",
                "known_issues": ["Observation queries with _summary=count may fail"]
            }
            
            # "grahame_r4": {
            #     "name": "Grahame's Test Server R4",
            #     "base_url": "http://test.fhir.org/r4",
            #     "version": "R4",
            #     "vendor": "HL7",
            #     "type": "public_test",
            #     "auth_type": "none",
            #     "description": "HL7 reference implementation",
            #     "supported_resources": ["*"],
            #     "search_strategy": "standard",
            #     "status": "intermittent",
            #     "known_issues": ["Server may be down or slow"]
            # },
            
            # # Azure/Microsoft Servers
            # "azure_telstra": {
            #     "name": "Azure Telstra Health",
            #     "base_url": "http://sqlonfhir-r4.azurewebsites.net/fhir",
            #     "version": "R4",
            #     "vendor": "Microsoft",
            #     "type": "cloud_test",
            #     "auth_type": "none",
            #     "description": "Azure SQL-on-FHIR implementation",
            #     "supported_resources": ["Patient", "Observation", "Condition", "Encounter"],
            #     "search_strategy": "sql_fhir"
            # },
            
            # # Cerner Servers
            # "cerner_open": {
            #     "name": "Cerner Open Sandbox",
            #     "base_url": "https://fhir-open.cerner.com/r4/ec2458f2-1e24-41c8-b71b-0e701af7583d",
            #     "version": "R4",
            #     "vendor": "Cerner",
            #     "type": "vendor_sandbox",
            #     "auth_type": "none",
            #     "description": "Cerner's open FHIR sandbox",
            #     "supported_resources": ["Patient", "Observation", "Condition", "MedicationRequest", "Encounter", "AllergyIntolerance"],
            #     "search_strategy": "cerner_optimized"
            # },
            
            # # Epic Servers
            # "epic_sandbox": {
            #     "name": "Epic FHIR Sandbox",
            #     "base_url": "https://fhir.epic.com/interconnect-fhir-oauth/api/FHIR/R4",
            #     "version": "R4",
            #     "vendor": "Epic",
            #     "type": "vendor_sandbox",
            #     "auth_type": "smart_on_fhir",
            #     "description": "Epic's SMART-on-FHIR sandbox",
            #     "oauth_config": {
            #         "authorize_url": "https://fhir.epic.com/interconnect-fhir-oauth/oauth2/authorize",
            #         "token_url": "https://fhir.epic.com/interconnect-fhir-oauth/oauth2/token"
            #     },
            #     "supported_resources": ["*"],
            #     "search_strategy": "epic_optimized"
            # },
            
            # # Google Cloud Healthcare - Updated URL
            # "gcp_healthcare": {
            #     "name": "Google Cloud Healthcare API",
            #     "base_url": "https://healthcare.googleapis.com/v1/projects/demo-project/locations/us-central1/datasets/demo-dataset/fhirStores/demo-fhir-store/fhir",
            #     "version": "R4",
            #     "vendor": "Google",
            #     "type": "cloud_production",
            #     "auth_type": "gcp_oauth",
            #     "description": "Google Cloud Healthcare FHIR API (Demo - requires auth)",
            #     "oauth_config": {
            #         "token_url": "https://oauth2.googleapis.com/token",
            #         "scope": "https://www.googleapis.com/auth/cloud-healthcare"
            #     },
            #     "supported_resources": ["*"],
            #     "search_strategy": "gcp_optimized",
            #     "status": "requires_auth",
            #     "known_issues": ["Requires Google Cloud authentication"]
            # },
            
            # # Aidbox (Health Samurai) - Updated URL
            # "aidbox_demo": {
            #     "name": "Aidbox Demo Server",
            #     "base_url": "https://aidbox.app/fhir",
            #     "version": "R4",
            #     "vendor": "Aidbox",
            #     "type": "vendor_demo",
            #     "auth_type": "basic",
            #     "description": "Aidbox demo FHIR server",
            #     "supported_resources": ["*"],
            #     "search_strategy": "aidbox_optimized",
            #     "status": "may_require_auth",
            #     "known_issues": ["Demo endpoint structure may change"]
            # },
            
            # # InterSystems IRIS for Health - Alternative URL
            # "intersystems_demo": {
            #     "name": "InterSystems IRIS Demo",
            #     "base_url": "https://fhir-server.smarthealthit.org",
            #     "version": "R4",
            #     "vendor": "InterSystems",
            #     "type": "vendor_demo",
            #     "auth_type": "none",
            #     "description": "SMART Health IT demo server (alternative)",
            #     "supported_resources": ["*"],
            #     "search_strategy": "intersystems_optimized",
            #     "status": "alternative_endpoint"
            # }
        }
    
    def get_server_config(self, server_name: str) -> Optional[Dict[str, Any]]:
        """Get server configuration by name"""
        return self.servers.get(server_name)
    
    def list_servers_by_vendor(self, vendor: str) -> List[str]:
        """List all servers for a specific vendor"""
        return [name for name, config in self.servers.items() if config.get("vendor") == vendor]

class SearchStrategyManager:
    """Manages different search strategies for different FHIR server vendors"""
    
    def get_search_params(self, server_config: Dict[str, Any], resource_type: str, 
                         patient_id: str, **kwargs) -> List[Dict[str, Any]]:
        """Get optimized search parameters for a server and resource type"""
        strategy_name = server_config.get("search_strategy", "standard")
        
        if strategy_name == "sql_fhir":
            return self._sql_fhir_strategy(resource_type, patient_id, **kwargs)
        elif strategy_name in ["cerner_optimized", "epic_optimized"]:
            return self._vendor_optimized_strategy(resource_type, patient_id, **kwargs)
        else:
            return self._standard_strategy(resource_type, patient_id, **kwargs)
    
    def _standard_strategy(self, resource_type: str, patient_id: str, **kwargs) -> List[Dict[str, Any]]:
        """Standard FHIR search strategy"""
        base_params = {"_count": kwargs.get("_count", "50")}
        strategies = []
        
        # Primary strategy
        if resource_type == "AllergyIntolerance":
            base_params["patient"] = f"Patient/{patient_id}"
        else:
            base_params["subject"] = f"Patient/{patient_id}"
        
        strategies.append(base_params.copy())
        
        # Fallback strategy - try without Patient/ prefix
        fallback_params = base_params.copy()
        if resource_type == "AllergyIntolerance":
            fallback_params["patient"] = patient_id
        else:
            fallback_params["subject"] = patient_id
        
        strategies.append(fallback_params)
        return strategies
    
    def _sql_fhir_strategy(self, resource_type: str, patient_id: str, **kwargs) -> List[Dict[str, Any]]:
        """SQL-on-FHIR optimized strategy"""
        strategies = []
        
        # Strategy 1: Direct patient ID
        params1 = {"_count": "100"}
        if resource_type == "AllergyIntolerance":
            params1["patient"] = patient_id
        else:
            params1["subject"] = patient_id
        strategies.append(params1)
        
        # Strategy 2: With Patient/ prefix
        params2 = {"_count": "100"}
        if resource_type == "AllergyIntolerance":
            params2["patient"] = f"Patient/{patient_id}"
        else:
            params2["subject"] = f"Patient/{patient_id}"
        strategies.append(params2)
        
        return strategies
    
    def _vendor_optimized_strategy(self, resource_type: str, patient_id: str, **kwargs) -> List[Dict[str, Any]]:
        """Vendor-specific optimized strategy"""
        strategies = self._standard_strategy(resource_type, patient_id, **kwargs)
        
        # Vendor-specific optimizations
        for strategy in strategies:
            strategy["_count"] = min(int(kwargs.get("_count", "50")), 20)  # Most vendors prefer smaller counts
        
        return strategies

class UniversalFhirMcpServer:
    """Universal FHIR MCP Server supporting all major vendors with enterprise-grade features"""
    
    def __init__(self):
        self.registry = FhirServerRegistry()
        self.search_manager = SearchStrategyManager()
        self.session = None
        self.current_server = "firely_local"  # Default to local Firely server
        self.request_stats = {}  # Track request statistics per server
        
        # Performance optimization: Condition codes cache
        self.condition_codes_cache = {}  # Cache condition codes per server
        self.cache_expiry = {}  # Track cache expiry times
        self.cache_duration = 3600  # 1 hour cache
        
        # Initialize OpenAI client
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not self.openai_api_key:
            logger.warning("OPENAI_API_KEY not found in environment variables. Clinical text extraction will not work.")
        
    async def initialize(self):
        """Initialize HTTP session with enterprise-grade configuration"""
        try:
            if self.session is not None:
                logger.info("Session already initialized")
                return
                
            # Ensure we have an event loop
            loop = get_or_create_eventloop()
            
            connector = aiohttp.TCPConnector(
                limit=100,  # Increased connection pool
                limit_per_host=20,
                keepalive_timeout=30,
                ssl=False  # Disable SSL verification for test servers
            )
            
            # Universal headers for maximum compatibility
            headers = {
                "Accept": "application/fhir+json, application/json, */*",
                "User-Agent": "Universal-FHIR-MCP-Server/2.0 (Enterprise Interoperability)",
                "Content-Type": "application/fhir+json",
                "Cache-Control": "no-cache",
                "Pragma": "no-cache"
            }
            
            self.session = aiohttp.ClientSession(
                connector=connector,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=180, connect=30),
                loop=loop
            )
            
            logger.info("Universal FHIR MCP Server initialized with enterprise configuration")
            
        except Exception as e:
            logger.error(f"Failed to initialize FHIR server: {str(e)}", exc_info=True)
            self.session = None
            raise

    async def close(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()

    def _normalize_parameters(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize parameters, converting count to _count for FHIR queries"""
        normalized = kwargs.copy()
        
        # Convert 'count' to '_count' for FHIR API compatibility
        if 'count' in normalized:
            normalized['_count'] = normalized.pop('count')
        
        # Ensure _count has a default value if not provided
        if '_count' not in normalized or normalized['_count'] is None:
            normalized['_count'] = "50"
        
        # Convert all parameter values to strings to avoid aiohttp errors
        for key, value in normalized.items():
            if value is None:
                normalized[key] = ""
            elif not isinstance(value, str):
                normalized[key] = str(value)
        
        return normalized

    def _get_example_patient(self, server_name: str) -> Optional[str]:
        """Get a known working example patient ID for the specified server"""
        # Known working patient IDs for each server
        example_patients = {
            "firely_local": "1",               # Local Firely server test patient
            "hapi_r4": "example",              # HAPI test patient
            # Commented out servers - add back as needed
            # "azure_telstra": "example",      # Azure Telstra test patient
            # "grahame_r4": "patient-example-a", # Grahame's server test patient
            # "cerner_open": "4342009",        # Cerner sandbox test patient
            # "epic_sandbox": "Tbt3KuCY0B5PSrJvCu2j-PlK.aiHsu2xUjUM8bWpetXoB", # Epic test patient
            # "intersystems_demo": "1",        # InterSystems demo patient
        }
        
        return example_patients.get(server_name)

    def _update_request_stats(self, server_name: str, success: bool, response_time: float):
        """Update request statistics for monitoring"""
        if server_name not in self.request_stats:
            self.request_stats[server_name] = {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "avg_response_time": 0,
                "last_request": None
            }
        
        stats = self.request_stats[server_name]
        stats["total_requests"] += 1
        stats["last_request"] = datetime.now().isoformat()
        
        if success:
            stats["successful_requests"] += 1
        else:
            stats["failed_requests"] += 1
        
        # Update average response time
        if stats["total_requests"] > 1:
            stats["avg_response_time"] = (
                (stats["avg_response_time"] * (stats["total_requests"] - 1) + response_time) / 
                stats["total_requests"]
            )
        else:
            stats["avg_response_time"] = response_time 

    async def switch_server(self, server_name: str) -> Dict[str, Any]:
        """Switch to a different FHIR server with full validation"""
        server_config = self.registry.get_server_config(server_name)
        if not server_config:
            return {
                "status": "error",
                "error": f"Server '{server_name}' not found in registry",
                "available_servers": list(self.registry.servers.keys())
            }
        
        try:
            # Test connectivity
            connectivity_result = await self.test_server_connectivity(server_name)
            if connectivity_result.get("status") != "connected":
                return {
                    "status": "error",
                    "error": f"Cannot connect to server '{server_name}'",
                    "connectivity_details": connectivity_result
                }
            
            # Update current server
            old_server = self.current_server
            self.current_server = server_name
            
            logger.info(f"Successfully switched from '{old_server}' to '{server_name}'")
            
            return {
                "status": "success",
                "previous_server": old_server,
                "current_server": server_name,
                "server_info": server_config
            }
            
        except Exception as e:
            logger.error(f"Failed to switch to server '{server_name}': {str(e)}")
            return {
                "status": "error",
                "error": f"Failed to switch to server '{server_name}': {str(e)}"
            }

    async def test_server_connectivity(self, server_name: str) -> Dict[str, Any]:
        """Test connectivity to a FHIR server with comprehensive validation"""
        try:
            # Ensure session is initialized
            if self.session is None:
                logger.info("Initializing FHIR server session...")
                await self.initialize()
                if self.session is None:
                    return {
                        "status": "error",
                        "error": "Failed to initialize FHIR server session"
                    }
            
            server_config = self.registry.get_server_config(server_name)
            if not server_config:
                return {
                    "status": "error",
                    "error": f"Server {server_name} not found in registry",
                    "available_servers": list(self.registry.servers.keys())
                }
            
            # Determine best endpoint to test connectivity based on server type
            base_url = server_config["base_url"]
            vendor = server_config.get("vendor", "").lower()
            
            # Choose the appropriate endpoint based on server type
            if "hapi" in vendor or "health" in vendor:
                # HAPI and other health servers often use /metadata directly
                metadata_urls = [
                    f"{base_url}/metadata",
                    f"{base_url}/CapabilityStatement",
                    f"{base_url}"
                ]
            elif "epic" in vendor:
                # Epic FHIR servers
                metadata_urls = [
                    f"{base_url}/metadata",
                    f"{base_url}/api/FHIR/R4/metadata"
                ]
            elif "cerner" in vendor:
                # Cerner FHIR servers
                metadata_urls = [
                    f"{base_url}/metadata",
                    f"{base_url}/fhir/metadata"
                ]
            else:
                # Try standard paths
                metadata_urls = [
                    f"{base_url}/metadata",
                    urljoin(base_url, "/metadata"),
                    f"{base_url}/fhir/metadata",
                    f"{base_url}/fhir/r4/metadata",
                    f"{base_url}"
                ]
            
            # Try each metadata URL
            last_error = None
            for metadata_url in metadata_urls:
                try:
                    # Prepare headers based on vendor
                    headers = {
                        "Accept": "application/fhir+json, application/json",
                        "User-Agent": "Universal-FHIR-MCP-Server/2.0",
                        "Cache-Control": "no-cache"
                    }
                    
                    if "epic" in vendor:
                        headers["Epic-Client-ID"] = "medplum_test"
                    elif "cerner" in vendor:
                        headers["Prefer"] = "handling=strict"
                    
                    logger.info(f"Testing connectivity to {server_name} at {metadata_url}")
                    
                    start_time = datetime.now()
                    async with self.session.get(metadata_url, headers=headers, timeout=15) as response:
                        response_time = (datetime.now() - start_time).total_seconds()
                        
                        if response.status == 200:
                            try:
                                metadata = await response.json()
                                fhir_version = metadata.get("fhirVersion", "unknown")
                                software = metadata.get("software", {})
                                
                                result = {
                                    "server": server_config["name"],
                                    "status": "connected",
                                    "fhir_version": fhir_version,
                                    "software": software,
                                    "vendor": server_config.get("vendor"),
                                    "type": server_config.get("type"),
                                    "url": server_config["base_url"],
                                    "endpoint_used": metadata_url,
                                    "response_time_ms": round(response_time * 1000, 2),
                                    "auth_type": server_config.get("auth_type"),
                                    "supported_resources": server_config.get("supported_resources", [])
                                }
                                
                                # Update request stats
                                self._update_request_stats(server_name, True, response_time)
                                
                                return result
                                
                            except json.JSONDecodeError:
                                # Handle XML or other non-JSON responses
                                text_response = await response.text()
                                
                                # Check if it's XML (common for FHIR)
                                if "<" in text_response and ">" in text_response:
                                    # Try to extract fhir version from XML
                                    import re
                                    fhir_version_match = re.search(r'fhirVersion\s*value\s*=\s*["\']([^"\']+)["\']', text_response)
                                    fhir_version = fhir_version_match.group(1) if fhir_version_match else "4.0.1"
                                    
                                    result = {
                                        "server": server_config["name"],
                                        "status": "connected",
                                        "fhir_version": fhir_version,
                                        "software": {"name": "Unknown (XML response)", "version": "Unknown"},
                                        "vendor": server_config.get("vendor"),
                                        "url": server_config["base_url"],
                                        "endpoint_used": metadata_url,
                                        "response_time_ms": round(response_time * 1000, 2),
                                        "note": "XML response received"
                                    }
                                    
                                    self._update_request_stats(server_name, True, response_time)
                                    return result
                                
                                # Not XML or JSON but got 200 status
                                last_error = f"Non-parseable response from {metadata_url}"
                                continue
                        
                        elif response.status in [401, 403]:
                            # Authentication required
                            last_error = f"Authentication required (status {response.status})"
                            continue
                        elif response.status == 404:
                            # Try the next URL
                            last_error = f"Endpoint not found: {metadata_url}"
                            continue
                        else:
                            last_error = f"HTTP status {response.status} from {metadata_url}"
                            continue
                            
                except asyncio.TimeoutError:
                    last_error = f"Timeout connecting to {metadata_url}"
                    continue
                except Exception as e:
                    last_error = f"Error connecting to {metadata_url}: {str(e)}"
                    continue
            
            # All URLs failed
            self._update_request_stats(server_name, False, 0)
            return {
                "server": server_config["name"],
                "status": "error",
                "error": last_error or "Failed to connect to any endpoint",
                "url": server_config["base_url"]
            }
                    
        except Exception as e:
            logger.error(f"Error testing server connectivity: {str(e)}", exc_info=True)
            self._update_request_stats(server_name, False, 0)
            return {
                "server": server_config["name"] if server_config else server_name,
                "status": "error",
                "error": f"Error testing server connectivity: {str(e)}",
                "url": server_config["base_url"] if server_config else None
            }

    async def universal_fhir_query(self, resource_type: str, patient_id: str = None, 
                                 server_name: str = None, **kwargs) -> Dict[str, Any]:
        """Universal FHIR query with vendor-specific optimizations"""
        # Ensure session is initialized
        if self.session is None:
            await self.initialize()
            
        target_server = server_name or self.current_server
        server_config = self.registry.get_server_config(target_server)
        
        if not server_config:
            return {
                "status": "error",
                "error": f"Server '{target_server}' not found in registry"
            }
            
        base_url = server_config["base_url"]
        vendor = server_config.get("vendor", "").lower()
        
        # Normalize parameters
        params = self._normalize_parameters(kwargs)
        
        # Add server-specific special parameters
        if server_config.get("special_params"):
            for key, value in server_config["special_params"].items():
                if key not in params:  # Don't override if already specified
                    params[key] = value
        
        # Special handling for condition code queries
        if resource_type == "Condition" and "code" in params:
            code_value = params["code"]
            # Extract just the code if it includes system prefix
            if ":" in code_value:
                code_value = code_value.split(":")[-1].strip()
            
            # For HAPI server, we need to use the code directly
            if "hapi" in vendor:
                params["code"] = code_value
            else:
                # For other servers, keep the original format
                params["code"] = code_value
                if "SNOMED" in str(kwargs.get("code", "")):
                    params["system"] = "http://snomed.info/sct"
                elif "ICD-10" in str(kwargs.get("code", "")):
                    params["system"] = "http://hl7.org/fhir/sid/icd-10"
        
        # Remove _text parameter for HAPI server as it's not well supported
        if "hapi" in vendor and "_text" in params:
            del params["_text"]
        
        try:
            # Build query URL - handle special cases for Azure Telstra server
            base_url = server_config["base_url"]
            
            # Server-specific URL handling
            vendor = server_config.get("vendor", "").lower()
            
            if target_server == "azure_telstra":
                # Special handling for Azure Telstra server which might use a different structure
                alternative_urls = [
                    f"{base_url}/{resource_type}",  # Standard
                    f"{base_url}/resources/{resource_type}",  # Some servers use this
                    f"{base_url}/api/{resource_type}",  # Azure-specific sometimes
                    f"{base_url}/fhir/{resource_type}"  # Another common pattern
                ]
            elif "epic" in vendor:
                # Epic FHIR servers typically use this pattern
                alternative_urls = [
                    f"{base_url}/{resource_type}",
                    f"{base_url}/api/FHIR/R4/{resource_type}"
                ]
            elif "cerner" in vendor:
                # Cerner FHIR servers
                alternative_urls = [
                    f"{base_url}/{resource_type}",
                    f"{base_url}/fhir/{resource_type}"
                ]
            elif "hapi" in vendor or "intersystems" in vendor:
                # HAPI and InterSystems typically use standard paths
                alternative_urls = [f"{base_url}/{resource_type}"]
            else:
                # Default to standard FHIR path plus some common alternatives
                alternative_urls = [
                    f"{base_url}/{resource_type}",
                    f"{base_url}/fhir/{resource_type}",
                    f"{base_url}/fhir/r4/{resource_type}"
                ]
            
            # Get optimized search parameters
            if patient_id:
                search_strategies = self.search_manager.get_search_params(
                    server_config, resource_type, patient_id, **kwargs
                )
                
                # Add vendor-specific reference formats
                if "epic" in vendor:
                    # Epic often requires specific formatting for references
                    epic_strategies = []
                    for strategy in search_strategies:
                        epic_strategy = strategy.copy()
                        if "subject" in epic_strategy and not epic_strategy["subject"].startswith("Patient/"):
                            epic_strategy["subject"] = f"Patient/{epic_strategy['subject']}"
                        epic_strategies.append(epic_strategy)
                    search_strategies.extend(epic_strategies)
                
                elif "cerner" in vendor:
                    # Cerner has specific requirements
                    cerner_strategies = []
                    for strategy in search_strategies:
                        cerner_strategy = strategy.copy()
                        # Ensure Cerner-specific params
                        if "_count" in cerner_strategy and int(cerner_strategy["_count"]) > 50:
                            cerner_strategy["_count"] = "50"  # Cerner often limits to smaller page sizes
                        cerner_strategies.append(cerner_strategy)
                    search_strategies.extend(cerner_strategies)
                
                elif target_server == "azure_telstra":
                    # Azure SQL on FHIR specific strategies
                    azure_strategies = []
                    
                    for strategy in search_strategies:
                        azure_strategy = strategy.copy()
                        
                        # Try without any prefix/suffix
                        if "subject" in azure_strategy:
                            if azure_strategy["subject"].startswith("Patient/"):
                                # Create a version without the prefix
                                new_strategy = azure_strategy.copy()
                                new_strategy["subject"] = azure_strategy["subject"].replace("Patient/", "")
                                azure_strategies.append(new_strategy)
                                
                                # Try with just the ID
                                id_only_strategy = azure_strategy.copy()
                                id_only_strategy["subject"] = patient_id
                                azure_strategies.append(id_only_strategy)
                                
                                # Try with a different parameter name
                                patient_strategy = azure_strategy.copy()
                                patient_strategy["patient"] = patient_id
                                del patient_strategy["subject"]
                                azure_strategies.append(patient_strategy)
                                
                        elif "patient" in azure_strategy:
                            if azure_strategy["patient"].startswith("Patient/"):
                                # Create a version without the prefix
                                new_strategy = azure_strategy.copy()
                                new_strategy["patient"] = azure_strategy["patient"].replace("Patient/", "")
                                azure_strategies.append(new_strategy)
                                
                                # Try with just the ID
                                id_only_strategy = azure_strategy.copy()
                                id_only_strategy["patient"] = patient_id
                                azure_strategies.append(id_only_strategy)
                                
                                # Try with a different parameter name
                                subject_strategy = azure_strategy.copy()
                                subject_strategy["subject"] = patient_id
                                del subject_strategy["patient"]
                                azure_strategies.append(subject_strategy)
                    
                    # Add the Azure strategies to our search strategies
                    search_strategies.extend(azure_strategies)
            else:
                # For non-patient queries, use basic parameters
                search_strategies = [self._normalize_parameters(kwargs)]
            
            # Add a basic strategy for just testing if the resource type exists
            if patient_id and target_server == "azure_telstra":
                # Try a basic query without patient reference to see if the resource type exists
                basic_strategy = {"_count": "1"}
                search_strategies.append(basic_strategy)
            
            all_errors = []
            
            # Try each URL first
            for url_attempt, current_url in enumerate(alternative_urls):
                # Try each search strategy for this URL
                for strategy_index, params in enumerate(search_strategies):
                    try:
                        start_time = datetime.now()
                        
                        headers = {
                            "Accept": "application/fhir+json, application/json",
                            "Content-Type": "application/fhir+json"
                        }
                        
                        # Add server-specific headers
                        if "epic" in vendor:
                            headers["Epic-Client-ID"] = "medplum_test"
                        elif "cerner" in vendor:
                            headers["Prefer"] = "handling=strict"
                        
                        logger.info(f"URL {url_attempt+1}/{len(alternative_urls)}, Strategy {strategy_index+1}/{len(search_strategies)}: "
                                  f"Querying {target_server} - {current_url} with params: {params}")
                        
                        async with self.session.get(current_url, params=params, headers=headers) as response:
                            response_time = (datetime.now() - start_time).total_seconds()
                            response_status = response.status
                            
                            if response_status == 200:
                                try:
                                    data = await response.json()
                                    resources = []
                                    
                                    if "entry" in data:
                                        for entry in data["entry"]:
                                            resources.append(entry.get("resource", {}))
                                    
                                    result = {
                                        "server": server_config["name"],
                                        "server_key": target_server,
                                        "vendor": server_config.get("vendor"),
                                        "resource_type": resource_type,
                                        "total": data.get("total", len(resources)),
                                        "resources": resources,
                                        "status": "success",
                                        "query_params": params,
                                        "url_used": current_url,
                                        "strategy_used": strategy_index + 1,
                                        "response_time_ms": round(response_time * 1000, 2)
                                    }
                                    
                                    self._update_request_stats(target_server, True, response_time)
                                    logger.info(f"Success: Found {len(resources)} {resource_type} resources on {target_server}")
                                    return result
                                    
                                except json.JSONDecodeError:
                                    error_text = await response.text()
                                    logger.warning(f"Non-JSON response from {target_server}: {error_text[:200]}")
                                    all_errors.append(f"Non-JSON response with status {response_status}")
                                    continue
                            
                            elif response_status == 404:
                                # Resource type might not be supported, try next strategy or URL
                                error_text = await response.text()
                                all_errors.append(f"404 Not Found on URL {current_url}: {error_text[:100]}")
                                logger.debug(f"Resource type {resource_type} not found at {current_url} on {target_server}")
                                # Continue with next strategy or URL
                            
                            else:
                                # Server error, try next strategy
                                error_text = await response.text()
                                all_errors.append(f"HTTP {response_status}: {error_text[:100]}")
                                logger.warning(f"HTTP {response_status} from {target_server}: {error_text[:200]}")
                                continue
                                
                    except asyncio.TimeoutError:
                        all_errors.append(f"Timeout for URL {current_url}, strategy {strategy_index + 1}")
                        logger.warning(f"Timeout for URL {current_url}, strategy {strategy_index + 1} on {target_server}")
                        continue
                    except Exception as e:
                        all_errors.append(f"Error: {str(e)}")
                        logger.warning(f"URL {current_url}, Strategy {strategy_index + 1} failed on {target_server}: {str(e)}")
                        continue
            
            # Try one last specific approach for Azure Telstra
            if target_server == "azure_telstra" and patient_id:
                try:
                    # Try direct SQL-on-FHIR query format
                    special_url = f"{base_url}/{resource_type}?_count=50"
                    
                    logger.info(f"Final attempt: Querying {target_server} with SQL-on-FHIR format: {special_url}")
                    
                    async with self.session.get(special_url) as response:
                        if response.status == 200:
                            try:
                                data = await response.json()
                                resources = []
                                
                                if "entry" in data:
                                    # Filter manually by patient ID
                                    for entry in data["entry"]:
                                        resource = entry.get("resource", {})
                                        
                                        # Check if this resource belongs to our patient
                                        is_match = False
                                        subject = resource.get("subject", {})
                                        if isinstance(subject, dict) and "reference" in subject:
                                            ref = subject["reference"]
                                            if patient_id in ref:
                                                is_match = True
                                        
                                        # Also check 'patient' reference
                                        patient_ref = resource.get("patient", {})
                                        if isinstance(patient_ref, dict) and "reference" in patient_ref:
                                            ref = patient_ref["reference"]
                                            if patient_id in ref:
                                                is_match = True
                                        
                                        if is_match:
                                            resources.append(resource)
                                
                                if resources:
                                    return {
                                        "server": server_config["name"],
                                        "server_key": target_server,
                                        "vendor": server_config.get("vendor"),
                                        "resource_type": resource_type,
                                        "total": len(resources),
                                        "resources": resources,
                                        "status": "success",
                                        "query_params": {"_count": "50"},
                                        "url_used": special_url,
                                        "strategy_used": "final_sql_fhir_attempt",
                                        "response_time_ms": 0
                                    }
                            except:
                                pass
                except Exception as e:
                    all_errors.append(f"Final attempt failed: {str(e)}")
            
            # Try a special case for HAPI FHIR server
            if "hapi" in vendor and resource_type.lower() == "metadata":
                try:
                    # HAPI sometimes uses /metadata directly
                    special_url = f"{base_url}/metadata"
                    logger.info(f"Trying HAPI special case: {special_url}")
                    
                    async with self.session.get(special_url, headers=headers) as response:
                        if response.status == 200:
                            try:
                                data = await response.json()
                                return {
                                    "server": server_config["name"],
                                    "server_key": target_server,
                                    "vendor": server_config.get("vendor"),
                                    "resource_type": resource_type,
                                    "total": 1,
                                    "resources": [data],
                                    "status": "success",
                                    "url_used": special_url,
                                    "strategy_used": "hapi_metadata",
                                    "response_time_ms": 0
                                }
                            except:
                                pass
                except Exception as e:
                    all_errors.append(f"HAPI special case failed: {str(e)}")
            
            # All strategies failed
            self._update_request_stats(target_server, False, 0)
            
            # Check for common auth issues in the errors
            auth_required = any("401" in err or "403" in err or "Authentication" in err or "Authorization" in err 
                               for err in all_errors if err)
            
            if auth_required:
                return {
                    "server": server_config["name"],
                    "server_key": target_server,
                    "resource_type": resource_type,
                    "status": "error",
                    "error": f"Authentication required for {resource_type} on {target_server}",
                    "auth_type": server_config.get("auth_type", "unknown"),
                    "note": "This server requires authentication. Please configure credentials or use a public server."
                }
            else:
                return {
                    "server": server_config["name"],
                    "server_key": target_server,
                    "resource_type": resource_type,
                    "status": "error",
                    "error": f"All search strategies failed for {resource_type} on {target_server}",
                    "error_details": all_errors[:5] if all_errors else None,  # Include the first few errors
                    "strategies_attempted": len(search_strategies),
                    "urls_attempted": len(alternative_urls)
                }
            
        except Exception as e:
            logger.error(f"Universal query failed for {resource_type} on {target_server}: {str(e)}")
            self._update_request_stats(target_server, False, 0)
            return {
                "server": server_config["name"],
                "server_key": target_server,
                "resource_type": resource_type,
                "status": "error",
                "error": str(e)
            }

    async def find_patient(self, search_criteria: Dict[str, str], servers: List[str] = None) -> Dict[str, Any]:
        """Universal patient search across multiple servers"""
        target_servers = servers or [self.current_server]
        results = {}
        
        for server_name in target_servers:
            server_config = self.registry.get_server_config(server_name)
            if not server_config:
                results[server_name] = {
                    "status": "error",
                    "error": f"Server {server_name} not found"
                }
                continue
            
            try:
                # Direct URL construction for more reliable searching
                base_url = server_config["base_url"]
                patient_endpoint = f"{base_url}/Patient"
                
                # Build search parameters - simpler approach for better compatibility
                all_patients = []
                all_patient_ids = []
                search_strategies = []
                
                # Strategy 1: Direct name search (most servers support this)
                if "name" in search_criteria:
                    strategy1 = {"name": search_criteria["name"]}
                    if "_count" in search_criteria:
                        strategy1["_count"] = search_criteria["_count"]
                    else:
                        strategy1["_count"] = "20"
                    search_strategies.append(("full_name", strategy1))
                
                # Strategy 2: Family name only
                if "name" in search_criteria and " " in search_criteria["name"]:
                    family_name = search_criteria["name"].split()[-1]
                    strategy2 = {"family": family_name, "_count": "20"}
                    search_strategies.append(("family_name", strategy2))
                
                # Strategy 3: Given name only
                if "name" in search_criteria and " " in search_criteria["name"]:
                    given_name = search_criteria["name"].split()[0]
                    strategy3 = {"given": given_name, "_count": "20"}
                    search_strategies.append(("given_name", strategy3))
                
                # Add specific strategies for Azure SQL on FHIR
                if server_name == "azure_telstra":
                    # Azure SQL on FHIR sometimes requires this format
                    if "name" in search_criteria:
                        name_parts = search_criteria["name"].split()
                        if len(name_parts) > 1:
                            # Try with exact match syntax
                            strategy4 = {"name:exact": search_criteria["name"], "_count": "20"}
                            search_strategies.append(("exact_name", strategy4))
                            
                            # Try with contains search
                            strategy5 = {"name:contains": name_parts[0], "_count": "20"}
                            search_strategies.append(("contains_name", strategy5))
                
                successful_strategy = None
                
                # Try each search strategy
                for strategy_name, params in search_strategies:
                    try:
                        logger.info(f"Trying {strategy_name} strategy on {server_name} with params: {params}")
                        
                        # Direct HTTP request for better reliability
                        start_time = datetime.now()
                        async with self.session.get(patient_endpoint, params=params) as response:
                            response_time = (datetime.now() - start_time).total_seconds()
                            
                            if response.status == 200:
                                try:
                                    data = await response.json()
                                    
                                    if "entry" in data and data["entry"]:
                                        successful_strategy = strategy_name
                                        
                                        # Process patients
                                        for entry in data["entry"]:
                                            resource = entry.get("resource", {})
                                            
                                            # Extract name data
                                            name_data = {}
                                            if "name" in resource and resource["name"]:
                                                name_obj = resource["name"][0] if isinstance(resource["name"], list) else resource["name"]
                                                if isinstance(name_obj, dict):
                                                    name_data = {
                                                        "family": name_obj.get("family", ""),
                                                        "given": name_obj.get("given", []),
                                                        "text": name_obj.get("text", ""),
                                                        "prefix": name_obj.get("prefix", [])
                                                    }
                                            
                                            patient = {
                                                "id": resource.get("id"),
                                                "name": name_data,
                                                "birthDate": resource.get("birthDate"),
                                                "gender": resource.get("gender"),
                                                "identifier": resource.get("identifier", [])
                                            }
                                            
                                            if patient.get("id") and patient["id"] not in all_patient_ids:
                                                all_patients.append(patient)
                                                all_patient_ids.append(patient["id"])
                                        
                                        # Break after finding patients
                                        if all_patients:
                                            break
                                except json.JSONDecodeError:
                                    logger.warning(f"Invalid JSON response from {server_name}")
                                    continue
                    except Exception as e:
                        logger.warning(f"Strategy {strategy_name} failed: {str(e)}")
                        continue
                
                # If specific fix for Azure needed:
                if server_name == "azure_telstra" and not all_patients:
                    # Last resort - try a broader search that should return some results
                    try:
                        logger.info("Attempting broader search for Azure SQL on FHIR")
                        async with self.session.get(f"{patient_endpoint}?_count=50") as response:
                            if response.status == 200:
                                data = await response.json()
                                if "entry" in data and data["entry"]:
                                    # Find anyone with a similar name
                                    name_parts = search_criteria.get("name", "").lower().split()
                                    for entry in data["entry"]:
                                        resource = entry.get("resource", {})
                                        if "name" in resource and resource["name"]:
                                            name_obj = resource["name"][0] if isinstance(resource["name"], list) else resource["name"]
                                            patient_family = name_obj.get("family", "").lower()
                                            patient_given = [g.lower() for g in name_obj.get("given", [])]
                                            
                                            # Check for any name part match
                                            matched = False
                                            for part in name_parts:
                                                if (part in patient_family or 
                                                    any(part in given for given in patient_given)):
                                                    matched = True
                                                    break
                                            
                                            if matched or not name_parts:  # Include if no name provided
                                                # Extract name data
                                                name_data = {
                                                    "family": name_obj.get("family", ""),
                                                    "given": name_obj.get("given", []),
                                                    "text": name_obj.get("text", ""),
                                                    "prefix": name_obj.get("prefix", [])
                                                }
                                                
                                                patient = {
                                                    "id": resource.get("id"),
                                                    "name": name_data,
                                                    "birthDate": resource.get("birthDate"),
                                                    "gender": resource.get("gender"),
                                                    "identifier": resource.get("identifier", [])
                                                }
                                                
                                                if patient.get("id") and patient["id"] not in all_patient_ids:
                                                    all_patients.append(patient)
                                                    all_patient_ids.append(patient["id"])
                                        
                                        successful_strategy = "fallback_broader_search"
                    except Exception as e:
                        logger.warning(f"Broader search fallback failed: {str(e)}")
                
                # Create results
                results[server_name] = {
                    "server": server_config["name"],
                    "server_key": server_name,
                    "vendor": server_config.get("vendor"),
                    "total": len(all_patients),
                    "patients": all_patients,
                    "all_patient_ids": all_patient_ids,
                    "status": "success",
                    "search_strategy": successful_strategy or "none_successful",
                    "strategies_attempted": len(search_strategies) + (1 if server_name == "azure_telstra" and not all_patients else 0)
                }
                
                logger.info(f"Found {len(all_patients)} patients on {server_name} using strategy: {successful_strategy}")
                
            except Exception as e:
                logger.error(f"Patient search failed on {server_name}: {str(e)}")
                results[server_name] = {
                    "server": server_config["name"],
                    "server_key": server_name,
                    "status": "error",
                    "error": str(e)
                }
        
        return {
            "search_criteria": search_criteria,
            "results": results
        }

    async def _get_patient_resources(self, resource_type: str, patient_id: Union[str, List[str]], 
                                   servers: List[str] = None, include_example: bool = False, **kwargs) -> Dict[str, Any]:
        """Universal method to get patient resources from multiple servers"""
        logger.info(f"Getting {resource_type} resources for patient(s)")
        target_servers = servers or [self.current_server]
        results = {}
        
        # Normalize parameters and ensure _count is a string
        normalized_kwargs = self._normalize_parameters(kwargs)
        if '_count' not in normalized_kwargs or normalized_kwargs['_count'] is None:
            normalized_kwargs['_count'] = "50"  # Set default count
        
        for server_name in target_servers:
            try:
                server_config = self.registry.get_server_config(server_name)
                if not server_config:
                    logger.warning(f"Server {server_name} not found")
                    results[server_name] = {
                        "status": "error",
                        "error": f"Server {server_name} not found"
                    }
                    continue
                
                # Get patient IDs to query
                if isinstance(patient_id, list):
                    patient_ids = patient_id.copy()
                else:
                    patient_ids = [patient_id]
                    
                # Initialize example_patient before the conditional block
                example_patient = None
                    
                # Add example patient if requested and available
                if include_example:
                    example_patient = self._get_example_patient(server_name)
                    if example_patient and example_patient not in patient_ids:
                        logger.debug(f"Adding example patient '{example_patient}' for server {server_name}")
                        patient_ids.append(example_patient)
                
                aggregated_resources = []
                total_count = 0
                error_messages = []
                auth_required = False
                
                for pid in patient_ids:
                    try:
                        logger.debug(f"Querying {resource_type} for patient {pid} on {server_name}")
                        
                        # Add category parameter for observations if it exists
                        if resource_type == "Observation" and "category" in normalized_kwargs:
                            query_params = normalized_kwargs.copy()
                            query_params["category"] = normalized_kwargs["category"]
                        else:
                            query_params = normalized_kwargs
                            
                        query_result = await self.universal_fhir_query(
                            resource_type, 
                            patient_id=pid, 
                            server_name=server_name, 
                            **query_params
                        )
                        
                        if query_result["status"] == "success":
                            total_count += query_result.get("total", 0)
                            
                            for resource in query_result.get("resources", []):
                                # Add patient_id_used for tracking
                                enhanced_resource = resource.copy()
                                enhanced_resource["patient_id_used"] = pid
                                aggregated_resources.append(enhanced_resource)
                                
                            logger.debug(f"Found {len(query_result.get('resources', []))} {resource_type} for patient {pid}")
                        else:
                            error_msg = query_result.get('error', 'Unknown error')
                            error_messages.append(f"Patient ID {pid}: {error_msg}")
                            logger.warning(f"Error querying {resource_type} for patient {pid}: {error_msg}")
                            
                            # Check if this is an authentication error
                            if "Authentication required" in error_msg or "auth" in error_msg.lower():
                                auth_required = True
                            
                    except Exception as e:
                        error_messages.append(f"Patient ID {pid}: {str(e)}")
                        logger.error(f"Error querying {resource_type} for patient {pid}: {str(e)}", exc_info=True)
                
                # Determine result status and add helpful messages
                status = "success" if aggregated_resources else "error"
                message = None
                
                if not aggregated_resources:
                    if auth_required:
                        message = f"Authentication required to access {resource_type} data on {server_name}"
                    elif example_patient and example_patient in patient_ids:
                        message = f"No {resource_type} data found for any patients on {server_name}, including example patient"
                    else:
                        message = f"No {resource_type} data found for specified patients on {server_name}"
                
                results[server_name] = {
                    "server": server_config["name"],
                    "server_key": server_name,
                    "vendor": server_config.get("vendor"),
                    "resource_type": resource_type,
                    "total": total_count,
                    "resources": aggregated_resources,
                    "status": status,
                    "patient_ids_queried": patient_ids,
                    "errors": error_messages if error_messages else None,
                    "message": message
                }
                
                logger.info(f"Total {resource_type} found on {server_name}: {total_count} across {len(patient_ids)} patient IDs")
                
            except Exception as e:
                logger.error(f"Error processing server {server_name}: {str(e)}", exc_info=True)
                results[server_name] = {
                    "status": "error",
                    "error": str(e),
                    "server": server_name,
                    "resource_type": resource_type
                }
        
        return results

    async def get_patient_observations(self, patient_id: Union[str, List[str]], 
                                     servers: List[str] = None, **kwargs) -> Dict[str, Any]:
        """Get patient observations from one or multiple servers"""
        logger.info(f"Fetching observations for patient(s): {patient_id}")
        try:
            # Handle include_example parameter
            include_example = kwargs.pop('include_example', False)
            
            # Get the result from _get_patient_resources
            result = await self._get_patient_resources(
                "Observation", 
                patient_id, 
                servers, 
                include_example=include_example,
                **kwargs
            )
            
            logger.info(f"Retrieved observations from {len(result) if result else 0} servers")
            return result
            
        except Exception as e:
            logger.error(f"Error fetching observations: {str(e)}", exc_info=True)
            return {}

    async def get_patient_conditions(self, patient_id: Union[str, List[str]], 
                                   servers: List[str] = None, **kwargs) -> Dict[str, Any]:
        """Get patient conditions from one or multiple servers"""
        return await self._get_patient_resources("Condition", patient_id, servers, **kwargs)

    async def get_patient_medications(self, patient_id: Union[str, List[str]], 
                                    servers: List[str] = None, **kwargs) -> Dict[str, Any]:
        """Get patient medications from one or multiple servers"""
        return await self._get_patient_resources("MedicationRequest", patient_id, servers, **kwargs)

    async def get_patient_encounters(self, patient_id: Union[str, List[str]], 
                                   servers: List[str] = None, **kwargs) -> Dict[str, Any]:
        """Get patient encounters from one or multiple servers"""
        return await self._get_patient_resources("Encounter", patient_id, servers, **kwargs)

    async def get_patient_allergies(self, patient_id: Union[str, List[str]], 
                                  servers: List[str] = None, **kwargs) -> Dict[str, Any]:
        """Get patient allergies from one or multiple servers"""
        return await self._get_patient_resources("AllergyIntolerance", patient_id, servers, **kwargs)

    async def get_patient_procedures(self, patient_id: Union[str, List[str]], 
                                   servers: List[str] = None, **kwargs) -> Dict[str, Any]:
        """Get patient procedures from one or multiple servers"""
        return await self._get_patient_resources("Procedure", patient_id, servers, **kwargs)

    async def get_patient_careteam(self, patient_id: Union[str, List[str]], 
                                 servers: List[str] = None, **kwargs) -> Dict[str, Any]:
        """Get patient care team from one or multiple servers"""
        return await self._get_patient_resources("CareTeam", patient_id, servers, **kwargs)

    async def get_patient_careplans(self, patient_id: Union[str, List[str]], 
                                  servers: List[str] = None, **kwargs) -> Dict[str, Any]:
        """Get patient care plans from one or multiple servers"""
        return await self._get_patient_resources("CarePlan", patient_id, servers, **kwargs)

    async def get_vital_signs(self, patient_id: Union[str, List[str]], 
                            servers: List[str] = None, **kwargs) -> Dict[str, Any]:
        """Get patient vital signs from one or multiple servers"""
        logger.info(f"Fetching vital signs for patient(s): {patient_id}")
        return await self.get_observations_by_category(
            patient_id, 
            "vital-signs", 
            servers=servers, 
            **kwargs
        )

    async def get_lab_results(self, patient_id: Union[str, List[str]], 
                            servers: List[str] = None, **kwargs) -> Dict[str, Any]:
        """Get patient lab results from one or multiple servers"""
        logger.info(f"Fetching lab results for patient(s): {patient_id}")
        return await self.get_observations_by_category(
            patient_id, 
            "laboratory", 
            servers=servers, 
            **kwargs
        )

    async def get_observations_by_category(self, patient_id: Union[str, List[str]], 
                                         category_code: str,
                                         servers: List[str] = None, **kwargs) -> Dict[str, Any]:
        """Get patient observations by specific category with proper FHIR category search handling"""
        logger.info(f"Fetching {category_code} observations for patient(s): {patient_id}")
        
        # Normalize parameters
        normalized_kwargs = self._normalize_parameters(kwargs)
        
        # Handle include_example if it's in kwargs
        include_example = normalized_kwargs.pop('include_example', False)
        
        target_servers = servers or [self.current_server]
        results = {}
        
        for server_name in target_servers:
            try:
                server_config = self.registry.get_server_config(server_name)
                if not server_config:
                    logger.warning(f"Server {server_name} not found")
                    results[server_name] = {
                        "status": "error",
                        "error": f"Server {server_name} not found"
                    }
                    continue
                
                # Get patient IDs to query
                if isinstance(patient_id, list):
                    patient_ids = patient_id.copy()
                else:
                    patient_ids = [patient_id]
                    
                # Add example patient if requested and available
                if include_example:
                    example_patient = self._get_example_patient(server_name)
                    if example_patient and example_patient not in patient_ids:
                        logger.debug(f"Adding example patient '{example_patient}' for server {server_name}")
                        patient_ids.append(example_patient)
                
                aggregated_resources = []
                total_count = 0
                error_messages = []
                auth_required = False
                
                for pid in patient_ids:
                    try:
                        logger.debug(f"Querying {category_code} observations for patient {pid} on {server_name}")
                        
                        # Try multiple category search strategies for different FHIR servers
                        category_strategies = self._get_category_search_strategies(category_code, server_config)
                        
                        query_success = False
                        for strategy_name, category_params in category_strategies.items():
                            try:
                                # Combine category parameters with other search parameters
                                query_params = normalized_kwargs.copy()
                                query_params.update(category_params)
                                
                                logger.debug(f"Trying {strategy_name} for {category_code} on {server_name}: {query_params}")
                                
                                query_result = await self.universal_fhir_query(
                                    "Observation", 
                                    patient_id=pid, 
                                    server_name=server_name, 
                                    **query_params
                                )
                                
                                if query_result["status"] == "success" and query_result.get("resources"):
                                    total_count += query_result.get("total", 0)
                                    
                                    for resource in query_result.get("resources", []):
                                        # Add patient_id_used and strategy for tracking
                                        enhanced_resource = resource.copy()
                                        enhanced_resource["patient_id_used"] = pid
                                        enhanced_resource["category_strategy"] = strategy_name
                                        aggregated_resources.append(enhanced_resource)
                                        
                                    logger.debug(f"Found {len(query_result.get('resources', []))} {category_code} observations for patient {pid} using {strategy_name}")
                                    query_success = True
                                    break  # Success, no need to try other strategies
                                    
                            except Exception as e:
                                logger.debug(f"Strategy {strategy_name} failed for patient {pid}: {str(e)}")
                                continue
                        
                        if not query_success:
                            error_msg = f"All category search strategies failed for {category_code}"
                            error_messages.append(f"Patient ID {pid}: {error_msg}")
                            logger.warning(f"Error querying {category_code} observations for patient {pid}: {error_msg}")
                            
                    except Exception as e:
                        error_messages.append(f"Patient ID {pid}: {str(e)}")
                        logger.error(f"Error querying {category_code} observations for patient {pid}: {str(e)}", exc_info=True)
                
                # Determine result status and add helpful messages
                status = "success" if aggregated_resources else "error"
                message = None
                
                if not aggregated_resources:
                    if auth_required:
                        message = f"Authentication required to access {category_code} observations on {server_name}"
                    else:
                        message = f"No {category_code} observations found for specified patients on {server_name}"
                
                results[server_name] = {
                    "server": server_config["name"],
                    "server_key": server_name,
                    "vendor": server_config.get("vendor"),
                    "resource_type": "Observation",
                    "category": category_code,
                    "total": total_count,
                    "resources": aggregated_resources,
                    "status": status,
                    "patient_ids_queried": patient_ids,
                    "errors": error_messages if error_messages else None,
                    "message": message
                }
                
                logger.info(f"Total {category_code} observations found on {server_name}: {total_count} across {len(patient_ids)} patient IDs")
                
            except Exception as e:
                logger.error(f"Error processing server {server_name}: {str(e)}", exc_info=True)
                results[server_name] = {
                    "status": "error",
                    "error": str(e),
                    "server": server_name,
                    "resource_type": "Observation",
                    "category": category_code
                }
        
        return results

    def _get_category_search_strategies(self, category_code: str, server_config: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
        """Get universal category search strategies for all FHIR servers"""
        # Base system URL for observation categories
        category_system = "http://terminology.hl7.org/CodeSystem/observation-category"
        
        strategies = {}
        
        # Strategy 1: Simple code-only search (most common)
        strategies["simple_code"] = {"category": category_code}
        
        # Strategy 2: System|code format (FHIR standard)
        strategies["system_code"] = {"category": f"{category_system}|{category_code}"}
        
        # Strategy 3: Text search fallback
        strategies["text_fallback"] = {"_text": category_code.replace("-", " ")}
        
        # Strategy 4: Category contains search
        strategies["category_contains"] = {"category:contains": category_code}
        
        # Add common alternative codes for specific categories
        if category_code == "vital-signs":
            strategies["vital_signs_alt"] = {"category": "vitals"}
            strategies["vital_signs_legacy"] = {"category": "vital"}
            
        elif category_code == "laboratory":
            strategies["lab_alt"] = {"category": "lab"}
            strategies["lab_legacy"] = {"category": "laboratory-test"}
            
        elif category_code == "imaging":
            strategies["imaging_alt"] = {"category": "radiology"}
            
        elif category_code == "procedure":
            strategies["procedure_alt"] = {"category": "procedures"}
            
        elif category_code == "social-history":
            strategies["social_alt"] = {"category": "social"}
            
        return strategies

    async def get_lab_results(self, patient_id: Union[str, List[str]], 
                            servers: List[str] = None, **kwargs) -> Dict[str, Any]:
        """Get patient lab results from one or multiple servers"""
        logger.info(f"Fetching lab results for patient(s): {patient_id}")
        return await self.get_observations_by_category(
            patient_id, 
            "laboratory", 
            servers=servers, 
            **kwargs
        )

    async def get_medications_history(self, patient_id: Union[str, List[str]], 
                                    servers: List[str] = None, **kwargs) -> Dict[str, Any]:
        """Get patient medication history from one or multiple servers"""
        return await self._get_patient_resources("MedicationStatement", patient_id, servers, **kwargs)

    async def clinical_query(self, resource_type: str, query_params: Dict[str, Any], 
                           servers: List[str] = None) -> Dict[str, Any]:
        """Execute custom FHIR queries across multiple servers"""
        target_servers = servers or [self.current_server]
        results = {}
        
        for server_name in target_servers:
            try:
                result = await self.universal_fhir_query(
                    resource_type, server_name=server_name, **query_params
                )
                results[server_name] = result
            except Exception as e:
                results[server_name] = {
                    "status": "error",
                    "error": str(e),
                    "resource_type": resource_type
                }
        
        return results

    async def diagnose_fhir_server(self, server_name: str = None) -> Dict[str, Any]:
        """Comprehensive diagnosis of a FHIR server's capabilities"""
        target_server = server_name or self.current_server
        server_config = self.registry.get_server_config(target_server)
        
        if not server_config:
            return {"error": f"Server {target_server} not found"}
        
        diagnostics = {
            "server_info": {
                "name": server_config["name"],
                "vendor": server_config.get("vendor"),
                "type": server_config.get("type"),
                "base_url": server_config["base_url"],
                "auth_type": server_config.get("auth_type"),
                "known_issues": server_config.get("known_issues", [])
            },
            "connectivity": {},
            "resource_availability": {},
            "performance": self.request_stats.get(target_server, {})
        }
        
        # Test connectivity
        try:
            connectivity_result = await self.test_server_connectivity(target_server)
            diagnostics["connectivity"] = connectivity_result
        except Exception as e:
            diagnostics["connectivity"] = {"status": "error", "error": str(e)}
        
        # Test resource availability
        resource_types = ["Patient", "Observation", "Condition", "MedicationRequest", 
                         "Encounter", "AllergyIntolerance", "Procedure", "DiagnosticReport"]
        
        for resource_type in resource_types:
            try:
                # Check for known issues with specific servers and resource types
                is_hapi = "hapi" in server_config.get("vendor", "").lower()
                use_summary_count = True
                
                # HAPI FHIR has issues with Observation queries using _summary=count
                if is_hapi and resource_type == "Observation":
                    use_summary_count = False
                    logger.info(f"Skipping _summary=count for {resource_type} on HAPI FHIR due to known issues")
                
                if use_summary_count:
                    query_result = await self.universal_fhir_query(
                        resource_type, server_name=target_server, _count="1", _summary="count"
                    )
                else:
                    # Fallback to regular query without _summary for problematic combinations
                    query_result = await self.universal_fhir_query(
                        resource_type, server_name=target_server, _count="1"
                    )
                
                if query_result["status"] == "success":
                    diagnostics["resource_availability"][resource_type] = {
                        "available": True,
                        "total_records": query_result.get("total", 0),
                        "response_time_ms": query_result.get("response_time_ms", 0),
                        "query_method": "_summary=count" if use_summary_count else "regular"
                    }
                else:
                    # Try fallback approach if _summary failed
                    if use_summary_count and "500" in str(query_result.get("error", "")):
                        logger.info(f"Retrying {resource_type} query without _summary=count due to server error")
                        fallback_result = await self.universal_fhir_query(
                            resource_type, server_name=target_server, _count="1"
                        )
                        
                        if fallback_result["status"] == "success":
                            diagnostics["resource_availability"][resource_type] = {
                                "available": True,
                                "total_records": fallback_result.get("total", 0),
                                "response_time_ms": fallback_result.get("response_time_ms", 0),
                                "query_method": "fallback_regular",
                                "note": "Server failed with _summary=count, succeeded with regular query"
                            }
                        else:
                            diagnostics["resource_availability"][resource_type] = {
                                "available": False,
                                "error": fallback_result.get("error", "Unknown error")
                            }
                    else:
                        diagnostics["resource_availability"][resource_type] = {
                            "available": False,
                            "error": query_result.get("error", "Unknown error")
                        }
                    
            except Exception as e:
                diagnostics["resource_availability"][resource_type] = {
                    "available": False,
                    "error": str(e)
                }
        
        return diagnostics

    def get_server_registry(self) -> Dict[str, Any]:
        """Get the complete server registry"""
        return {
            "current_server": self.current_server,
            "total_servers": len(self.registry.servers),
            "servers": self.registry.servers,
            "servers_by_vendor": {
                vendor: self.registry.list_servers_by_vendor(vendor)
                for vendor in set(config.get("vendor") for config in self.registry.servers.values())
                if vendor is not None
            },
            "request_stats": self.request_stats
        }

    async def extract_clinical_keywords(self, text: str) -> Dict[str, Any]:
        """Extract clinical keywords and concepts from free text using OpenAI"""
        if not self.openai_api_key:
            return {
                "error": "OpenAI API key not configured",
                "raw_text": text
            }
        
        try:
            # Ensure session is initialized
            if self.session is None:
                logger.info("Initializing session for clinical keyword extraction...")
                await self.initialize()
                if self.session is None:
                    return {
                        "error": "Failed to initialize HTTP session",
                        "raw_text": text
                    }
            
            # Create the prompt for OpenAI
            prompt = f"""
            Extract clinical information from the following doctor's note. Return a JSON object with:
            - age: patient's age as a number (null if not mentioned)
            - gender: patient's gender (null if not mentioned)
            - conditions: array of medical conditions/diagnoses (use standard medical terms)
            - stage: cancer stage if mentioned (e.g., "IIIA", "IV", null if not applicable)
            - procedures: array of procedures/treatments mentioned (past and current)
            - medications: array of medications mentioned
            - biomarkers: array of biomarkers/receptors (e.g., "HER2-positive", "ER-positive")
            - symptoms: array of symptoms mentioned
            - timeline: key timeline information (e.g., "completed AC chemo", "3 months post-op")
            
            Doctor's note: "{text}"
            
            Return ONLY valid JSON, no additional text.
            """
            
            # Make async HTTP request to OpenAI
            headers = {
                "Authorization": f"Bearer {self.openai_api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "gpt-4o",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a clinical data extraction assistant. Extract structured medical information from clinical notes."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.3,  # Low temperature for consistent extraction
                "response_format": { "type": "json_object" }
            }
            
            async with self.session.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    extracted_data = json.loads(result['choices'][0]['message']['content'])
                    extracted_data['raw_text'] = text
                    extracted_data['extraction_method'] = 'openai_gpt4'
                    
                    logger.info(f"Successfully extracted clinical keywords: {extracted_data}")
                    return extracted_data
                else:
                    error_text = await response.text()
                    logger.error(f"OpenAI API error: {response.status} - {error_text}")
                    return {
                        "error": f"OpenAI API error: {response.status}",
                        "raw_text": text
                    }
                    
        except Exception as e:
            logger.error(f"Error extracting clinical keywords: {str(e)}")
            return {
                "error": str(e),
                "raw_text": text
            }

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings using multiple methods"""
        text1 = text1.lower().strip()
        text2 = text2.lower().strip()
        
        if text1 == text2:
            return 1.0
        
        # Method 1: Check if one is contained in the other
        if text1 in text2 or text2 in text1:
            return 0.8
        
        # Method 2: Word overlap scoring
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        word_similarity = len(intersection) / len(union) if union else 0.0
        
        # Method 3: Check for common medical term patterns
        # Handle cases like "liver steatosis" vs "steatosis of liver"
        key_words1 = [w for w in words1 if len(w) > 3]  # Focus on meaningful words
        key_words2 = [w for w in words2 if len(w) > 3]
        
        if key_words1 and key_words2:
            key_intersection = set(key_words1).intersection(set(key_words2))
            key_union = set(key_words1).union(set(key_words2))
            key_similarity = len(key_intersection) / len(key_union) if key_union else 0.0
            
            # Give more weight to key word similarity
            final_similarity = max(word_similarity, key_similarity * 1.2)
        else:
            final_similarity = word_similarity
        
        return min(final_similarity, 1.0)  # Cap at 1.0

    async def map_to_fhir_codes(self, clinical_data: Dict[str, Any], 
                               servers: List[str] = None,
                               similarity_threshold: float = 0.6,
                               max_matches: int = 3,
                               clean_output: bool = True) -> Dict[str, Any]:
        """Map extracted clinical terms to FHIR codes using actual server data with fuzzy matching"""
        logger.info("Starting FHIR code mapping using actual server data")
        logger.debug(f"Input clinical data: {json.dumps(clinical_data, indent=2)}")
        
        try:
            conditions = clinical_data.get("conditions", [])
            procedures = clinical_data.get("procedures", [])
            medications = clinical_data.get("medications", [])
            
            logger.debug(f"Found {len(conditions)} conditions, {len(procedures)} procedures, {len(medications)} medications to map")
            
            if not (conditions or procedures or medications):
                logger.warning("No clinical terms found to map to FHIR codes.")
                return clinical_data
            
            # Extract actual condition codes from FHIR servers
            logger.info("Extracting condition codes from FHIR servers...")
            server_condition_codes = await self.extract_condition_codes_from_fhir(servers)
            
            # Aggregate all conditions from all servers
            all_fhir_conditions = []
            for server_data in server_condition_codes.values():
                if server_data["status"] == "success":
                    all_fhir_conditions.extend(server_data["conditions"])
            
            logger.info(f"Total FHIR conditions available for matching: {len(all_fhir_conditions)}")
            
            # Map conditions using fuzzy matching
            condition_codes = []
            for condition in conditions:
                logger.debug(f"Mapping condition: {condition}")
                
                best_matches = []
                condition_lower = condition.lower().strip()
                
                # Find all matches above threshold
                for fhir_condition in all_fhir_conditions:
                    similarity = self._calculate_text_similarity(condition_lower, fhir_condition["searchable_name"])
                    
                    if similarity >= similarity_threshold:
                        match = {
                            "similarity": similarity,
                            "name": fhir_condition["display"] or fhir_condition["text"],
                            "code": fhir_condition["code"],
                            "system": fhir_condition["system"]
                        }
                        best_matches.append(match)
                
                # Sort by similarity and take top matches
                best_matches.sort(key=lambda x: x["similarity"], reverse=True)
                top_matches = best_matches[:max_matches]
                
                if top_matches:
                    if clean_output:
                        # Clean output format - just essential info
                        condition_mapping = {
                            "term": condition,
                            "matches": [
                                {
                                    "name": match["name"],
                                    "code": match["code"],
                                    "similarity": round(match["similarity"], 2)
                                }
                                for match in top_matches
                            ]
                        }
                    else:
                        # Full output format for compatibility
                        codes = []
                        for match in top_matches:
                            if match["system"] and match["code"]:
                                codes.append(f"{match['system']}|{match['code']}")
                            elif match["code"]:
                                codes.append(match["code"])
                        
                        condition_mapping = {
                            "term": condition,
                            "codes": codes,
                            "matches": [
                                {
                                    "display": match["name"],
                                    "similarity": round(match["similarity"], 3),
                                    "code": match["code"],
                                    "system": match["system"]
                                }
                                for match in top_matches
                            ]
                        }
                    
                    condition_codes.append(condition_mapping)
                    logger.debug(f"Found {len(top_matches)} matches for '{condition}'")
                else:
                    logger.debug(f"No matches found for condition: {condition} (threshold: {similarity_threshold})")
            
            # For now, we'll focus on conditions. Later we can extend to procedures and medications
            # TODO: Add similar logic for procedures and medications
            procedure_codes = []
            medication_codes = []
            
            # Add the mapped codes to clinical data
            clinical_data['condition_codes'] = condition_codes
            clinical_data['procedure_codes'] = procedure_codes  # Placeholder
            clinical_data['medication_codes'] = medication_codes  # Placeholder
            clinical_data['mapping_method'] = 'fhir_server_fuzzy_matching'
            clinical_data['similarity_threshold'] = similarity_threshold
            
            # Only include server condition codes if not using clean output
            if not clean_output:
                clinical_data['server_condition_codes'] = server_condition_codes
            
            logger.info(f"Successfully mapped {len(condition_codes)} conditions using FHIR server data")
            
            # Log detailed mapping results
            for condition_mapping in condition_codes:
                logger.debug(f"Condition mapping: {condition_mapping['term']} -> {len(condition_mapping['matches'])} matches")
                for match in condition_mapping['matches']:
                    logger.debug(f"  - {match['name']} (similarity: {match.get('similarity', 'N/A')})")
                    
            return clinical_data
                    
        except Exception as e:
            logger.error(f"Error mapping to FHIR codes using server data: {str(e)}", exc_info=True)
            # Fallback to original clinical data
            return clinical_data

    async def map_to_fhir_codes_fast(self, clinical_data: Dict[str, Any], 
                                    servers: List[str] = None,
                                    similarity_threshold: float = 0.6,
                                    max_matches: int = 3) -> Dict[str, Any]:
        """Fast version of FHIR code mapping with optimized defaults to avoid timeouts"""
        logger.info("Starting FAST FHIR code mapping using actual server data")
        
        try:
            conditions = clinical_data.get("conditions", [])
            procedures = clinical_data.get("procedures", [])
            medications = clinical_data.get("medications", [])
            
            if not (conditions or procedures or medications):
                logger.warning("No clinical terms found to map to FHIR codes.")
                return clinical_data
            
            # Use only the first server or current server to avoid timeouts
            target_servers = [servers[0]] if servers else [self.current_server]
            
            # Extract actual condition codes from FHIR servers with fast limits
            logger.info("Extracting condition codes from FHIR servers (FAST mode)...")
            server_condition_codes = await self.extract_condition_codes_from_fhir(
                servers=target_servers,
                max_conditions=500,  # Reduced from 5000
                use_cache=True,
                max_pages=2  # Reduced from 10
            )
            
            # Aggregate all conditions from all servers
            all_fhir_conditions = []
            for server_data in server_condition_codes.values():
                if server_data["status"] == "success":
                    all_fhir_conditions.extend(server_data["conditions"])
            
            logger.info(f"Total FHIR conditions available for matching (FAST): {len(all_fhir_conditions)}")
            
            # Map conditions using fuzzy matching
            condition_codes = []
            for condition in conditions:
                logger.debug(f"Mapping condition: {condition}")
                
                best_matches = []
                condition_lower = condition.lower().strip()
                
                # Find all matches above threshold
                for fhir_condition in all_fhir_conditions:
                    similarity = self._calculate_text_similarity(condition_lower, fhir_condition["searchable_name"])
                    
                    if similarity >= similarity_threshold:
                        match = {
                            "similarity": similarity,
                            "name": fhir_condition["display"] or fhir_condition["text"],
                            "code": fhir_condition["code"]
                        }
                        best_matches.append(match)
                
                # Sort by similarity and take top matches
                best_matches.sort(key=lambda x: x["similarity"], reverse=True)
                top_matches = best_matches[:max_matches]
                
                if top_matches:
                    # Clean output format - just essential info
                    condition_mapping = {
                        "term": condition,
                        "matches": [
                            {
                                "name": match["name"],
                                "code": match["code"],
                                "similarity": round(match["similarity"], 2)
                            }
                            for match in top_matches
                        ]
                    }
                    condition_codes.append(condition_mapping)
                    logger.debug(f"Found {len(top_matches)} matches for '{condition}'")
                else:
                    logger.debug(f"No matches found for condition: {condition} (threshold: {similarity_threshold})")
            
            # Update clinical data with results
            clinical_data['condition_codes'] = condition_codes
            clinical_data['procedure_codes'] = []  # Placeholder
            clinical_data['medication_codes'] = []  # Placeholder
            clinical_data['mapping_method'] = 'fhir_server_fuzzy_matching_fast'
            clinical_data['similarity_threshold'] = similarity_threshold
            
            logger.info(f"Successfully mapped {len(condition_codes)} conditions using FAST FHIR server data")
            
            return clinical_data
                    
        except Exception as e:
            logger.error(f"Error in fast FHIR code mapping: {str(e)}", exc_info=True)
            # Fallback to original clinical data
            return clinical_data

    async def find_similar_patients(self, criteria: Dict[str, Any], 
                                  servers: List[str] = None, 
                                  max_results: int = 10) -> Dict[str, Any]:
        """Find patients with similar clinical profiles"""
        target_servers = servers or [self.current_server]
        
        # First, map clinical terms to codes if not already done
        if 'condition_codes' not in criteria:
            criteria = await self.map_to_fhir_codes(criteria)
        
        all_matches = []
        
        for server_name in target_servers:
            server_matches = []
            
            # Search using condition codes
            condition_codes = []
            for condition_mapping in criteria.get("condition_codes", []):
                condition_codes.extend(condition_mapping.get("codes", []))
            
            # Also search by raw condition terms if no codes found
            if not condition_codes and criteria.get("conditions"):
                for condition in criteria["conditions"]:
                    condition_codes.append(condition)
            
            # Search for matching conditions
            for code in condition_codes:
                try:
                    # Try searching by code first
                    condition_result = await self.universal_fhir_query(
                        "Condition",
                        server_name=server_name,
                        code=code,
                        _count="100"
                    )
                    
                    # If no results with code, try text search
                    if condition_result["status"] != "success" or not condition_result.get("resources"):
                        condition_result = await self.universal_fhir_query(
                            "Condition",
                            server_name=server_name,
                            _text=code,  # Text search
                            _count="100"
                        )
                    
                    if condition_result["status"] == "success":
                        # Extract unique patient IDs from conditions
                        patient_ids = set()
                        for resource in condition_result.get("resources", []):
                            subject = resource.get("subject", {})
                            if subject.get("reference"):
                                patient_id = subject["reference"].split("/")[-1]
                                patient_ids.add(patient_id)
                        
                        # Score each patient
                        for patient_id in patient_ids:
                            match_score = await self._calculate_patient_match_score_advanced(
                                patient_id, criteria, server_name
                            )
                            
                            if match_score["total_score"] > 0:
                                server_matches.append({
                                    "patient_id": patient_id,
                                    "server": server_name,
                                    "match_details": match_score,
                                    "score": match_score["total_score"]
                                })
                
                except Exception as e:
                    logger.error(f"Error searching conditions on {server_name}: {str(e)}")
            
            # Sort by match score
            server_matches.sort(key=lambda x: x["score"], reverse=True)
            all_matches.extend(server_matches[:max_results])
        
        # Remove duplicates and sort
        unique_matches = {}
        for match in all_matches:
            key = f"{match['server']}:{match['patient_id']}"
            if key not in unique_matches or match['score'] > unique_matches[key]['score']:
                unique_matches[key] = match
        
        sorted_matches = sorted(unique_matches.values(), key=lambda x: x['score'], reverse=True)
        
        # Get detailed information for top matches
        detailed_matches = []
        for match in sorted_matches[:max_results]:
            try:
                # Get patient demographics
                patient_data = await self.universal_fhir_query(
                    "Patient",
                    server_name=match["server"],
                    _id=match["patient_id"]
                )
                
                if patient_data["status"] == "success" and patient_data.get("resources"):
                    patient_resource = patient_data["resources"][0]
                    
                    # Get care plans
                    care_plans = await self.get_patient_careplans(
                        match["patient_id"],
                        servers=[match["server"]]
                    )
                    
                    # Get recent procedures
                    procedures = await self.get_patient_procedures(
                        match["patient_id"],
                        servers=[match["server"]],
                        _count="10"
                    )
                    
                    # Get outcomes (observations)
                    outcomes = await self.get_patient_observations(
                        match["patient_id"],
                        servers=[match["server"]],
                        _count="20"
                    )
                    
                    detailed_match = {
                        "patient_id": match["patient_id"],
                        "server": match["server"],
                        "match_score": match["score"],
                        "match_details": match["match_details"],
                        "demographics": self._extract_patient_demographics(patient_resource),
                        "care_plans": care_plans.get(match["server"], {}).get("resources", []),
                        "recent_procedures": procedures.get(match["server"], {}).get("resources", []),
                        "outcomes": self._extract_significant_outcomes(
                            outcomes.get(match["server"], {}).get("resources", [])
                        )
                    }
                    
                    detailed_matches.append(detailed_match)
            
            except Exception as e:
                logger.error(f"Error getting details for patient {match['patient_id']}: {str(e)}")
        
        return {
            "search_criteria": criteria,
            "total_matches": len(unique_matches),
            "matches": detailed_matches
        }

    async def _calculate_patient_match_score_advanced(self, patient_id: str, criteria: Dict[str, Any], 
                                                    server_name: str) -> Dict[str, Any]:
        """Calculate advanced match score using extracted clinical data"""
        score_details = {
            "age_score": 0,
            "gender_score": 0,
            "condition_score": 0,
            "procedure_score": 0,
            "medication_score": 0,
            "biomarker_score": 0,
            "stage_score": 0,
            "total_score": 0
        }
        
        try:
            # Get patient data
            patient_result = await self.universal_fhir_query(
                "Patient",
                server_name=server_name,
                _id=patient_id
            )
            
            if patient_result["status"] != "success" or not patient_result.get("resources"):
                return score_details
            
            patient = patient_result["resources"][0]
            
            # Age scoring
            if criteria.get("age") and patient.get("birthDate"):
                patient_age = self._calculate_age(patient["birthDate"])
                age_diff = abs(patient_age - criteria["age"])
                
                if age_diff == 0:
                    score_details["age_score"] = 15
                elif age_diff <= 2:
                    score_details["age_score"] = 12
                elif age_diff <= 5:
                    score_details["age_score"] = 8
                elif age_diff <= 10:
                    score_details["age_score"] = 4
            
            # Gender scoring
            if criteria.get("gender") and patient.get("gender"):
                if criteria["gender"].lower() == patient["gender"].lower():
                    score_details["gender_score"] = 5
            
            # Get all patient data for comprehensive matching
            conditions = await self.get_patient_conditions(patient_id, servers=[server_name], _count="50", include_example=False)
            procedures = await self.get_patient_procedures(patient_id, servers=[server_name], _count="50", include_example=False)
            medications = await self.get_patient_medications(patient_id, servers=[server_name], _count="50", include_example=False)
            observations = await self.get_patient_observations(patient_id, servers=[server_name], _count="50", include_example=False)
            
            # Convert resources to searchable text
            patient_data_text = {
                "conditions": json.dumps(conditions.get(server_name, {}).get("resources", [])).lower(),
                "procedures": json.dumps(procedures.get(server_name, {}).get("resources", [])).lower(),
                "medications": json.dumps(medications.get(server_name, {}).get("resources", [])).lower(),
                "observations": json.dumps(observations.get(server_name, {}).get("resources", [])).lower()
            }
            
            # Score conditions
            for condition in criteria.get("conditions", []):
                if condition.lower() in patient_data_text["conditions"]:
                    score_details["condition_score"] += 10
            
            # Score biomarkers
            for biomarker in criteria.get("biomarkers", []):
                if biomarker.lower() in patient_data_text["conditions"] or biomarker.lower() in patient_data_text["observations"]:
                    score_details["biomarker_score"] += 15  # High weight for biomarker match
            
            # Score stage
            if criteria.get("stage"):
                stage_pattern = f"stage.*{criteria['stage'].lower()}"
                import re
                if re.search(stage_pattern, patient_data_text["conditions"]):
                    score_details["stage_score"] = 20  # High weight for stage match
            
            # Score procedures
            for procedure in criteria.get("procedures", []):
                if procedure.lower() in patient_data_text["procedures"]:
                    score_details["procedure_score"] += 8
            
            # Score medications
            for medication in criteria.get("medications", []):
                if medication.lower() in patient_data_text["medications"]:
                    score_details["medication_score"] += 5
            
            # Calculate total score
            score_details["total_score"] = sum(
                score for key, score in score_details.items() 
                if key != "total_score"
            )
            
        except Exception as e:
            logger.error(f"Error calculating advanced match score for patient {patient_id}: {str(e)}")
        
        return score_details

    def _calculate_age(self, birth_date: str) -> int:
        """Calculate age from birth date string"""
        try:
            birth = datetime.strptime(birth_date, "%Y-%m-%d")
            today = datetime.now()
            age = today.year - birth.year - ((today.month, today.day) < (birth.month, birth.day))
            return age
        except:
            return 0

    def _extract_patient_demographics(self, patient_resource: Dict[str, Any]) -> Dict[str, Any]:
        """Extract patient demographics from FHIR Patient resource"""
        demographics = {
            "age": None,
            "gender": patient_resource.get("gender"),
            "name": None,
            "birth_date": patient_resource.get("birthDate")
        }
        
        # Calculate age
        if demographics["birth_date"]:
            demographics["age"] = self._calculate_age(demographics["birth_date"])
        
        # Extract name
        if patient_resource.get("name") and patient_resource["name"]:
            name_obj = patient_resource["name"][0]
            if isinstance(name_obj, dict):
                given = name_obj.get("given", [])
                family = name_obj.get("family", "")
                demographics["name"] = f"{' '.join(given)} {family}".strip()
        
        return demographics

    def _extract_observation_text(self, observation: Dict[str, Any]) -> str:
        """Extract readable text from observation"""
        # Try to get display text from code
        if observation.get("code", {}).get("text"):
            return observation["code"]["text"]
        elif observation.get("code", {}).get("coding"):
            codings = observation["code"]["coding"]
            for coding in codings:
                if coding.get("display"):
                    return coding["display"]
        
        return "Unknown observation"

    def _extract_significant_outcomes(self, observations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract significant outcomes from observations"""
        significant_outcomes = []
        
        # Keywords that indicate significant outcomes
        outcome_keywords = [
            'response', 'remission', 'progression', 'recurrence', 'survival',
            'tumor', 'marker', 'reduction', 'stable', 'disease-free', 'relapse',
            'complete response', 'partial response', 'stable disease', 'progressive disease'
        ]
        
        for obs in observations:
            obs_text = json.dumps(obs).lower()
            
            # Check if this observation contains outcome-related information
            if any(keyword in obs_text for keyword in outcome_keywords):
                outcome = {
                    "date": obs.get("effectiveDateTime", obs.get("issued")),
                    "type": self._extract_observation_text(obs),
                    "value": None
                }
                
                # Extract value
                if obs.get("valueQuantity"):
                    value = obs["valueQuantity"]
                    outcome["value"] = f"{value.get('value', '')} {value.get('unit', '')}"
                elif obs.get("valueString"):
                    outcome["value"] = obs["valueString"]
                elif obs.get("valueCodeableConcept"):
                    outcome["value"] = obs["valueCodeableConcept"].get("text", "")
                
                significant_outcomes.append(outcome)
        
        return significant_outcomes

    async def find_similar_patients_simple(self, clinical_data: Dict[str, Any], 
                                        servers: List[str] = None, 
                                          max_results: int = 10,
                                          age_tolerance: int = 10) -> Dict[str, Any]:
        """
        Find similar patients using only age, gender, and condition codes.
        Expects input from map_to_fhir_codes_fast tool.
        
        Args:
            clinical_data: Output from map_to_fhir_codes_fast containing age, gender, condition_codes
            servers: List of FHIR servers to search
            max_results: Maximum number of similar patients to return
            age_tolerance: Age difference tolerance (default: within 10 years)
            
        Returns:
            Dict containing:
            - search_criteria: Original search parameters
            - total_found: Total number of matches across all servers
            - returned: Number of matches returned after filtering
            - patients: List of matched patients with scores and details
        """
        # Log the start of the search process
        logger.info("Starting simplified similar patient search")
        
        # Use provided servers or default to current server
        target_servers = servers or [self.current_server]
        
        # Extract key demographic data from input
        target_age = clinical_data.get("age")
        target_gender = clinical_data.get("gender", "").lower()
        condition_codes = clinical_data.get("condition_codes", [])
        
        logger.info(f"Searching for patients: Age {target_age}, Gender {target_gender}, {len(condition_codes)} conditions")
        
        # Validate input data
        if not condition_codes:
            return {
                "error": "No condition codes found. Please run map_to_fhir_codes_fast first.",
                "input_data": clinical_data
            }
        
        # Store all matches across servers
        all_matches = []
        
        # Iterate through each FHIR server
        for server_name in target_servers:
            logger.info(f"Searching {server_name}...")
            server_matches = []
            
            # Process each condition from the input data
            for condition_mapping in condition_codes:
                condition_term = condition_mapping.get("term", "")
                matches = condition_mapping.get("matches", [])
                
                logger.debug(f"Searching for condition: {condition_term}")
                
                # Process top 10 code matches for each condition
                for match in matches[:10]:  # Limit to top 10 matches per condition for speed
                    code = match.get("code", "")
                    if not code:
                        continue
                    
                    try:
                        # Query FHIR server for patients with this condition
                        condition_result = await self.universal_fhir_query(
                            "Condition",
                            server_name=server_name,
                            code=code,
                            _count="50"  # Limit results for performance
                        )
                        
                        if condition_result["status"] == "success":
                            # Process each matching condition resource
                            for resource in condition_result.get("resources", []):
                                subject = resource.get("subject", {})
                                if subject.get("reference"):
                                    # Extract patient ID from reference
                                    patient_id = subject["reference"].split("/")[-1]
                                    
                                    # Calculate match score based on demographics
                                    score = await self._calculate_simple_match_score(
                                        patient_id, target_age, target_gender, age_tolerance, server_name
                                    )
                                    
                                    # Store match details
                                    server_matches.append({
                                        "patient_id": patient_id,
                                        "server": server_name,
                                        "matched_condition": condition_term,
                                        "matched_code": code,
                                        "score_details": score,
                                        "total_score": score["total_score"]
                                    })
                    
                    except Exception as e:
                        logger.warning(f"Error searching for code {code}: {str(e)}")
                        continue
            
            # Deduplicate patients and keep highest scoring match
            unique_patients = {}
            for match in server_matches:
                patient_key = match["patient_id"]
                if patient_key not in unique_patients or match["total_score"] > unique_patients[patient_key]["total_score"]:
                    unique_patients[patient_key] = match
            
            # Sort matches by score and take top results
            sorted_matches = sorted(unique_patients.values(), key=lambda x: x["total_score"], reverse=True)
            all_matches.extend(sorted_matches[:max_results])
            
            logger.info(f"Found {len(sorted_matches)} unique patients on {server_name}")
        
        # Final sorting of all matches across servers
        final_matches = sorted(all_matches, key=lambda x: x["total_score"], reverse=True)[:max_results]
        
        # Enhance matches with detailed patient information
        enhanced_matches = []
        for match in final_matches:
            try:
                # Fetch complete patient demographics
                patient_result = await self.universal_fhir_query(
                    "Patient",
                    server_name=match["server"],
                    _id=match["patient_id"]
                )
                
                if patient_result["status"] == "success" and patient_result.get("resources"):
                    patient_resource = patient_result["resources"][0]
                    demographics = self._extract_patient_demographics(patient_resource)
                    
                    # Create enhanced match object with demographics
                    enhanced_match = {
                        "patient_id": match["patient_id"],
                        "server": match["server"],
                        "total_score": match["total_score"],
                        "score_breakdown": match["score_details"],
                        "matched_condition": match["matched_condition"],
                        "matched_code": match["matched_code"],
                        "patient_info": {
                            "name": demographics.get("name", "Unknown"),
                            "age": demographics.get("age"),
                            "gender": demographics.get("gender"),
                            "birth_date": demographics.get("birth_date")
                        }
                    }
                    enhanced_matches.append(enhanced_match)
            
            except Exception as e:
                # Handle errors gracefully by including basic match info
                logger.warning(f"Error getting patient details for {match['patient_id']}: {str(e)}")
                enhanced_matches.append({
                    "patient_id": match["patient_id"],
                    "server": match["server"],
                    "total_score": match["total_score"],
                    "error": f"Could not retrieve patient details: {str(e)}"
                })
        
        # Prepare final result object
        result = {
            "search_criteria": {
                "age": target_age,
                "gender": target_gender,
                "age_tolerance": age_tolerance,
                "condition_count": len(condition_codes)
            },
            "total_found": len(all_matches),
            "returned": len(enhanced_matches),
            "patients": enhanced_matches
        }
        
        logger.info(f"Similar patient search completed: {len(enhanced_matches)} patients found")
        return result

    async def _calculate_simple_match_score(self, patient_id: str, target_age: int, 
                                          target_gender: str, age_tolerance: int, 
                                          server_name: str) -> Dict[str, Any]:
        """
        Calculate simple match score based on age and gender only.
        
        Scoring system:
        - Age scoring (max 20 points):
            * Exact match: 20 points
            * Within 2 years: 15 points
            * Within 5 years: 10 points
            * Within tolerance: 5 points
            * Outside tolerance: 0 points
        - Gender scoring (max 10 points):
            * Exact match: 10 points
            * No match: 0 points
            
        Total possible score: 30 points
        """
        # Initialize score details
        score_details = {
            "age_score": 0,
            "gender_score": 0,
            "total_score": 0,
            "patient_age": None,
            "patient_gender": None
        }
        
        try:
            # Fetch patient data from FHIR server
            patient_result = await self.universal_fhir_query(
                "Patient",
                server_name=server_name,
                _id=patient_id
            )
            
            if patient_result["status"] != "success" or not patient_result.get("resources"):
                return score_details
            
            patient = patient_result["resources"][0]
            
            # Calculate age score if both target and patient age are available
            if target_age and patient.get("birthDate"):
                patient_age = self._calculate_age(patient["birthDate"])
                score_details["patient_age"] = patient_age
                
                age_diff = abs(patient_age - target_age)
                
                # Apply age scoring rules
                if age_diff == 0:
                    score_details["age_score"] = 20  # Perfect match
                elif age_diff <= 2:
                    score_details["age_score"] = 15  # Very close
                elif age_diff <= 5:
                    score_details["age_score"] = 10  # Close
                elif age_diff <= age_tolerance:
                    score_details["age_score"] = 5   # Within tolerance
                # else: 0 points (outside tolerance)
            
            # Calculate gender score if both target and patient gender are available
            if target_gender and patient.get("gender"):
                patient_gender = patient["gender"].lower()
                score_details["patient_gender"] = patient_gender
                
                if target_gender == patient_gender:
                    score_details["gender_score"] = 10  # Gender match
            
            # Calculate total score (sum of age and gender scores)
            score_details["total_score"] = score_details["age_score"] + score_details["gender_score"]
            
        except Exception as e:
            logger.warning(f"Error calculating match score for patient {patient_id}: {str(e)}")
        
        return score_details

    def list_available_servers(self) -> Dict[str, Any]:
        """
        List all available servers with their details.
        
        Returns:
            Dict containing:
            - current_server: Name of the currently active server
            - servers: Dict of server configurations with details like:
                * name: Server name
                * vendor: Server vendor/provider
                * type: Server type
                * base_url: Server endpoint
                * auth_type: Authentication method
                * supported_resources: List of supported FHIR resources
                * description: Server description
        """
        return {
            "current_server": self.current_server,
            "servers": {
                name: {
                    "name": config["name"],
                    "vendor": config.get("vendor"),
                    "type": config.get("type"),
                    "base_url": config["base_url"],
                    "auth_type": config.get("auth_type"),
                    "supported_resources": config.get("supported_resources", []),
                    "description": config.get("description", "")
                }
                for name, config in self.registry.servers.items()
            }
        }

    def clear_condition_cache(self, server_name: str = None):
        """Clear condition codes cache for specific server or all servers"""
        if server_name:
            if server_name in self.condition_codes_cache:
                del self.condition_codes_cache[server_name]
                del self.cache_expiry[server_name]
                logger.info(f"Cleared condition cache for {server_name}")
        else:
            self.condition_codes_cache.clear()
            self.cache_expiry.clear()
            logger.info("Cleared all condition caches")
    
    def get_condition_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about condition codes cache"""
        stats = {
            "cached_servers": list(self.condition_codes_cache.keys()),
            "cache_details": {},
            "total_cached_conditions": 0
        }
        
        current_time = datetime.now().timestamp()
        
        for server_name, cache_data in self.condition_codes_cache.items():
            expiry_time = self.cache_expiry.get(server_name, 0)
            is_expired = current_time > expiry_time
            time_remaining = max(0, expiry_time - current_time)
            
            server_stats = {
                "conditions_count": cache_data.get("total_conditions", 0),
                "extraction_time": cache_data.get("extraction_time_seconds", 0),
                "conditions_per_second": cache_data.get("conditions_per_second", 0),
                "pages_fetched": cache_data.get("pages_fetched", 0),
                "is_expired": is_expired,
                "time_remaining_seconds": round(time_remaining, 0),
                "cached_at": datetime.fromtimestamp(expiry_time - self.cache_duration).isoformat() if expiry_time > 0 else None
            }
            
            stats["cache_details"][server_name] = server_stats
            stats["total_cached_conditions"] += server_stats["conditions_count"]
        
        return stats

    async def get_comprehensive_patient_info(self, patient_id: str, server_name: str = None) -> Dict[str, Any]:
        """
        Comprehensive patient information gathering tool that collects all patient data from a single server.
        
        Gathers:
        - Patient demographics (name, gender, age, conditions)
        - Procedures (display name and date)
        - Encounters (display name and date)
        - Medications (date and dosage)
        - Practitioner information
        - Diagnostic reports (date, display name, conclusion)
        - Care plans (date and display name)
        """
        target_server = server_name or self.current_server
        logger.info(f"Gathering comprehensive patient information for patient {patient_id} from {target_server}")
        
        start_time = datetime.now()
        
        # Initialize result structure
        patient_summary = {
            "patient_id": patient_id,
            "server": target_server,
            "demographics": {},
            "conditions": [],
            "procedures": [],
            "encounters": [],
            "medications": [],
            "practitioners": [],
            "diagnostic_reports": [],
            "care_plans": [],
            "errors": [],
            "summary": {}
        }
        
        try:
            # 1. Get Patient Demographics
            logger.info("Fetching patient demographics...")
            try:
                patient_result = await self.universal_fhir_query(
                    "Patient",
                    patient_id=None,  # Don't filter by patient reference for Patient resource
                    server_name=target_server,
                    _id=patient_id
                )
                
                if patient_result["status"] == "success" and patient_result.get("resources"):
                    patient_resource = patient_result["resources"][0]
                    
                    # Extract demographics
                    demographics = {
                        "id": patient_resource.get("id"),
                        "name": self._extract_patient_name(patient_resource),
                        "gender": patient_resource.get("gender"),
                        "birth_date": patient_resource.get("birthDate"),
                        "age": None
                    }
                    
                    # Calculate age
                    if demographics["birth_date"]:
                        demographics["age"] = self._calculate_age(demographics["birth_date"])
                    
                    patient_summary["demographics"] = demographics
                    logger.info(f"Patient: {demographics['name']}, Age: {demographics['age']}, Gender: {demographics['gender']}")
                else:
                    patient_summary["errors"].append("Failed to retrieve patient demographics")
            except Exception as e:
                patient_summary["errors"].append(f"Error fetching demographics: {str(e)}")
                logger.error(f"Error fetching patient demographics: {str(e)}")
            
            # 2. Get Patient Conditions
            logger.info("Fetching patient conditions...")
            try:
                conditions_result = await self.get_patient_conditions(patient_id, servers=[target_server])
                
                if target_server in conditions_result and conditions_result[target_server]["status"] == "success":
                    for condition_resource in conditions_result[target_server].get("resources", []):
                        condition_info = self._extract_condition_info(condition_resource)
                        if condition_info:
                            patient_summary["conditions"].append(condition_info)
                    logger.info(f"Found {len(patient_summary['conditions'])} conditions")
                else:
                    patient_summary["errors"].append("Failed to retrieve patient conditions")
            except Exception as e:
                patient_summary["errors"].append(f"Error fetching conditions: {str(e)}")
                logger.error(f"Error fetching patient conditions: {str(e)}")
            
            # 3. Get Patient Procedures
            logger.info("Fetching patient procedures...")
            try:
                procedures_result = await self.get_patient_procedures(patient_id, servers=[target_server])
                
                if target_server in procedures_result and procedures_result[target_server]["status"] == "success":
                    for procedure_resource in procedures_result[target_server].get("resources", []):
                        procedure_info = self._extract_procedure_info(procedure_resource)
                        if procedure_info:
                            patient_summary["procedures"].append(procedure_info)
                    logger.info(f"Found {len(patient_summary['procedures'])} procedures")
                else:
                    patient_summary["errors"].append("Failed to retrieve patient procedures")
            except Exception as e:
                patient_summary["errors"].append(f"Error fetching procedures: {str(e)}")
                logger.error(f"Error fetching patient procedures: {str(e)}")
            
            # 4. Get Patient Encounters
            logger.info("Fetching patient encounters...")
            try:
                encounters_result = await self.get_patient_encounters(patient_id, servers=[target_server])
                
                if target_server in encounters_result and encounters_result[target_server]["status"] == "success":
                    for encounter_resource in encounters_result[target_server].get("resources", []):
                        encounter_info = self._extract_encounter_info(encounter_resource)
                        if encounter_info:
                            patient_summary["encounters"].append(encounter_info)
                    logger.info(f"Found {len(patient_summary['encounters'])} encounters")
                else:
                    patient_summary["errors"].append("Failed to retrieve patient encounters")
            except Exception as e:
                patient_summary["errors"].append(f"Error fetching encounters: {str(e)}")
                logger.error(f"Error fetching patient encounters: {str(e)}")
            
            # 5. Get Patient Medications
            logger.info("Fetching patient medications...")
            try:
                medications_result = await self.get_patient_medications(patient_id, servers=[target_server])
                
                if target_server in medications_result and medications_result[target_server]["status"] == "success":
                    for medication_resource in medications_result[target_server].get("resources", []):
                        medication_info = self._extract_medication_info(medication_resource)
                        if medication_info:
                            patient_summary["medications"].append(medication_info)
                    logger.info(f"Found {len(patient_summary['medications'])} medications")
                else:
                    patient_summary["errors"].append("Failed to retrieve patient medications")
            except Exception as e:
                patient_summary["errors"].append(f"Error fetching medications: {str(e)}")
                logger.error(f"Error fetching patient medications: {str(e)}")
            
            # 6. Get Practitioners (from encounters and care team)
            logger.info("Fetching practitioners...")
            try:
                # Get practitioners from CareTeam
                careteam_result = await self.get_patient_careteam(patient_id, servers=[target_server])
                
                if target_server in careteam_result and careteam_result[target_server]["status"] == "success":
                    for careteam_resource in careteam_result[target_server].get("resources", []):
                        practitioners = self._extract_practitioners_from_careteam(careteam_resource)
                        patient_summary["practitioners"].extend(practitioners)
                
                # Also extract practitioners from encounters
                for encounter in patient_summary["encounters"]:
                    if encounter.get("practitioner"):
                        practitioner_info = {
                            "name": encounter["practitioner"],
                            "role": "Encounter Provider",
                            "source": "Encounter"
                        }
                        # Avoid duplicates
                        if not any(p["name"] == practitioner_info["name"] for p in patient_summary["practitioners"]):
                            patient_summary["practitioners"].append(practitioner_info)
                
                logger.info(f"Found {len(patient_summary['practitioners'])} practitioners")
            except Exception as e:
                patient_summary["errors"].append(f"Error fetching practitioners: {str(e)}")
                logger.error(f"Error fetching practitioners: {str(e)}")
            
            # 7. Get Diagnostic Reports
            logger.info("Fetching diagnostic reports...")
            try:
                diagnostics_result = await self.universal_fhir_query(
                    "DiagnosticReport",
                    patient_id=patient_id,
                    server_name=target_server,
                    _count="50"
                )
                
                if diagnostics_result["status"] == "success":
                    for diagnostic_resource in diagnostics_result.get("resources", []):
                        diagnostic_info = self._extract_diagnostic_report_info(diagnostic_resource)
                        if diagnostic_info:
                            patient_summary["diagnostic_reports"].append(diagnostic_info)
                    logger.info(f"Found {len(patient_summary['diagnostic_reports'])} diagnostic reports")
                else:
                    patient_summary["errors"].append("Failed to retrieve diagnostic reports")
            except Exception as e:
                patient_summary["errors"].append(f"Error fetching diagnostic reports: {str(e)}")
                logger.error(f"Error fetching diagnostic reports: {str(e)}")
            
            # 8. Get Care Plans
            logger.info("Fetching care plans...")
            try:
                careplans_result = await self.get_patient_careplans(patient_id, servers=[target_server])
                
                if target_server in careplans_result and careplans_result[target_server]["status"] == "success":
                    for careplan_resource in careplans_result[target_server].get("resources", []):
                        careplan_info = self._extract_careplan_info(careplan_resource)
                        if careplan_info:
                            patient_summary["care_plans"].append(careplan_info)
                    logger.info(f"Found {len(patient_summary['care_plans'])} care plans")
                else:
                    patient_summary["errors"].append("Failed to retrieve care plans")
            except Exception as e:
                patient_summary["errors"].append(f"Error fetching care plans: {str(e)}")
                logger.error(f"Error fetching care plans: {str(e)}")
            
            # Generate summary statistics
            processing_time = (datetime.now() - start_time).total_seconds()
            
            patient_summary["summary"] = {
                "processing_time_seconds": round(processing_time, 2),
                "total_conditions": len(patient_summary["conditions"]),
                "total_procedures": len(patient_summary["procedures"]),
                "total_encounters": len(patient_summary["encounters"]),
                "total_medications": len(patient_summary["medications"]),
                "total_practitioners": len(patient_summary["practitioners"]),
                "total_diagnostic_reports": len(patient_summary["diagnostic_reports"]),
                "total_care_plans": len(patient_summary["care_plans"]),
                "total_errors": len(patient_summary["errors"]),
                "data_completeness": self._calculate_data_completeness(patient_summary)
            }
            
            logger.info(f"Comprehensive patient info gathered in {processing_time:.2f}s - "
                       f"Completeness: {patient_summary['summary']['data_completeness']:.1f}%")
            
            return patient_summary
            
        except Exception as e:
            logger.error(f"Critical error in comprehensive patient info gathering: {str(e)}", exc_info=True)
            patient_summary["errors"].append(f"Critical error: {str(e)}")
            return patient_summary

    def _extract_patient_name(self, patient_resource: Dict[str, Any]) -> str:
        """Extract patient name from FHIR Patient resource"""
        if patient_resource.get("name") and patient_resource["name"]:
            name_obj = patient_resource["name"][0]
            if isinstance(name_obj, dict):
                given = name_obj.get("given", [])
                family = name_obj.get("family", "")
                prefix = name_obj.get("prefix", [])
                
                # Combine name parts
                name_parts = []
                if prefix:
                    name_parts.extend(prefix)
                if given:
                    name_parts.extend(given)
                if family:
                    name_parts.append(family)
                
                return " ".join(name_parts) if name_parts else "Unknown"
        return "Unknown"

    def _extract_condition_info(self, condition_resource: Dict[str, Any]) -> Dict[str, Any]:
        """Extract condition information from FHIR Condition resource"""
        condition_info = {
            "display_name": "Unknown Condition",
            "clinical_status": condition_resource.get("clinicalStatus", {}).get("text"),
            "verification_status": condition_resource.get("verificationStatus", {}).get("text"),
            "onset_date": condition_resource.get("onsetDateTime"),
            "recorded_date": condition_resource.get("recordedDate"),
            "codes": []
        }
        
        # Extract condition name
        if condition_resource.get("code"):
            code_obj = condition_resource["code"]
            if code_obj.get("text"):
                condition_info["display_name"] = code_obj["text"]
            elif code_obj.get("coding"):
                for coding in code_obj["coding"]:
                    if coding.get("display"):
                        condition_info["display_name"] = coding["display"]
                        break
                    
            # Extract all codes
            if code_obj.get("coding"):
                for coding in code_obj["coding"]:
                    if coding.get("code"):
                        condition_info["codes"].append({
                            "code": coding.get("code"),
                            "system": coding.get("system"),
                            "display": coding.get("display")
                        })
        
        return condition_info

    def _extract_procedure_info(self, procedure_resource: Dict[str, Any]) -> Dict[str, Any]:
        """Extract procedure information from FHIR Procedure resource"""
        procedure_info = {
            "display_name": "Unknown Procedure",
            "date": None,
            "status": procedure_resource.get("status"),
            "category": None
        }
        
        # Extract procedure name
        if procedure_resource.get("code"):
            code_obj = procedure_resource["code"]
            if code_obj.get("text"):
                procedure_info["display_name"] = code_obj["text"]
            elif code_obj.get("coding"):
                for coding in code_obj["coding"]:
                    if coding.get("display"):
                        procedure_info["display_name"] = coding["display"]
                        break
        
        # Extract date
        if procedure_resource.get("performedDateTime"):
            procedure_info["date"] = procedure_resource["performedDateTime"]
        elif procedure_resource.get("performedPeriod", {}).get("start"):
            procedure_info["date"] = procedure_resource["performedPeriod"]["start"]
        
        # Extract category
        if procedure_resource.get("category", {}).get("text"):
            procedure_info["category"] = procedure_resource["category"]["text"]
        elif procedure_resource.get("category", {}).get("coding"):
            for coding in procedure_resource["category"]["coding"]:
                if coding.get("display"):
                    procedure_info["category"] = coding["display"]
                    break
        
        return procedure_info

    def _extract_encounter_info(self, encounter_resource: Dict[str, Any]) -> Dict[str, Any]:
        """Extract encounter information from FHIR Encounter resource"""
        encounter_info = {
            "display_name": "Unknown Encounter",
            "date": None,
            "status": encounter_resource.get("status"),
            "class": None,
            "practitioner": None
        }
        
        # Extract encounter type/name
        if encounter_resource.get("type"):
            types = encounter_resource["type"]
            if types and isinstance(types, list):
                type_obj = types[0]
                if type_obj.get("text"):
                    encounter_info["display_name"] = type_obj["text"]
                elif type_obj.get("coding"):
                    for coding in type_obj["coding"]:
                        if coding.get("display"):
                            encounter_info["display_name"] = coding["display"]
                            break
        
        # Extract encounter class
        if encounter_resource.get("class", {}).get("display"):
            encounter_info["class"] = encounter_resource["class"]["display"]
        elif encounter_resource.get("class", {}).get("code"):
            encounter_info["class"] = encounter_resource["class"]["code"]
        
        # Extract date
        if encounter_resource.get("period", {}).get("start"):
            encounter_info["date"] = encounter_resource["period"]["start"]
        
        # Extract practitioner
        if encounter_resource.get("participant"):
            for participant in encounter_resource["participant"]:
                if participant.get("individual", {}).get("display"):
                    encounter_info["practitioner"] = participant["individual"]["display"]
                    break
        
        return encounter_info

    def _extract_medication_info(self, medication_resource: Dict[str, Any]) -> Dict[str, Any]:
        """Extract medication information from FHIR MedicationRequest resource"""
        medication_info = {
            "display_name": "Unknown Medication",
            "date": None,
            "dosage": None,
            "status": medication_resource.get("status"),
            "intent": medication_resource.get("intent")
        }
        
        # Extract medication name
        if medication_resource.get("medicationCodeableConcept"):
            med_obj = medication_resource["medicationCodeableConcept"]
            if med_obj.get("text"):
                medication_info["display_name"] = med_obj["text"]
            elif med_obj.get("coding"):
                for coding in med_obj["coding"]:
                    if coding.get("display"):
                        medication_info["display_name"] = coding["display"]
                        break
        
        # Extract date
        if medication_resource.get("authoredOn"):
            medication_info["date"] = medication_resource["authoredOn"]
        elif medication_resource.get("effectivePeriod", {}).get("start"):
            medication_info["date"] = medication_resource["effectivePeriod"]["start"]
        
        # Extract dosage
        if medication_resource.get("dosageInstruction"):
            dosage_instructions = medication_resource["dosageInstruction"]
            if dosage_instructions and isinstance(dosage_instructions, list):
                dosage_obj = dosage_instructions[0]
                
                dosage_parts = []
                
                # Extract dose quantity
                if dosage_obj.get("doseAndRate"):
                    dose_and_rate = dosage_obj["doseAndRate"]
                    if dose_and_rate and isinstance(dose_and_rate, list):
                        dose_info = dose_and_rate[0]
                        if dose_info.get("doseQuantity"):
                            dose_qty = dose_info["doseQuantity"]
                            value = dose_qty.get("value", "")
                            unit = dose_qty.get("unit", "")
                            dosage_parts.append(f"{value} {unit}".strip())
                
                # Extract frequency
                if dosage_obj.get("timing", {}).get("repeat", {}).get("frequency"):
                    frequency = dosage_obj["timing"]["repeat"]["frequency"]
                    period = dosage_obj["timing"]["repeat"].get("period", 1)
                    period_unit = dosage_obj["timing"]["repeat"].get("periodUnit", "day")
                    dosage_parts.append(f"{frequency} times per {period} {period_unit}")
                
                # Extract text instruction
                if dosage_obj.get("text"):
                    dosage_parts.append(dosage_obj["text"])
                
                medication_info["dosage"] = "; ".join(dosage_parts) if dosage_parts else None
        
        return medication_info

    def _extract_practitioners_from_careteam(self, careteam_resource: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract practitioners from FHIR CareTeam resource"""
        practitioners = []
        
        if careteam_resource.get("participant"):
            for participant in careteam_resource["participant"]:
                practitioner_info = {
                    "name": "Unknown Practitioner",
                    "role": None,
                    "source": "CareTeam"
                }
                
                # Extract practitioner name
                if participant.get("member", {}).get("display"):
                    practitioner_info["name"] = participant["member"]["display"]
                
                # Extract role
                if participant.get("role"):
                    roles = participant["role"]
                    if roles and isinstance(roles, list):
                        role_obj = roles[0]
                        if role_obj.get("text"):
                            practitioner_info["role"] = role_obj["text"]
                        elif role_obj.get("coding"):
                            for coding in role_obj["coding"]:
                                if coding.get("display"):
                                    practitioner_info["role"] = coding["display"]
                                    break
                
                practitioners.append(practitioner_info)
        
        return practitioners

    def _extract_diagnostic_report_info(self, diagnostic_resource: Dict[str, Any]) -> Dict[str, Any]:
        """Extract diagnostic report information from FHIR DiagnosticReport resource"""
        diagnostic_info = {
            "display_name": "Unknown Diagnostic Report",
            "date": None,
            "conclusion": None,
            "status": diagnostic_resource.get("status"),
            "category": None
        }
        
        # Extract report name
        if diagnostic_resource.get("code"):
            code_obj = diagnostic_resource["code"]
            if code_obj.get("text"):
                diagnostic_info["display_name"] = code_obj["text"]
            elif code_obj.get("coding"):
                for coding in code_obj["coding"]:
                    if coding.get("display"):
                        diagnostic_info["display_name"] = coding["display"]
                        break
        
        # Extract date
        if diagnostic_resource.get("effectiveDateTime"):
            diagnostic_info["date"] = diagnostic_resource["effectiveDateTime"]
        elif diagnostic_resource.get("effectivePeriod", {}).get("start"):
            diagnostic_info["date"] = diagnostic_resource["effectivePeriod"]["start"]
        elif diagnostic_resource.get("issued"):
            diagnostic_info["date"] = diagnostic_resource["issued"]
        
        # Extract conclusion
        if diagnostic_resource.get("conclusion"):
            diagnostic_info["conclusion"] = diagnostic_resource["conclusion"]
        elif diagnostic_resource.get("conclusionCode"):
            conclusion_codes = diagnostic_resource["conclusionCode"]
            if conclusion_codes and isinstance(conclusion_codes, list):
                code_obj = conclusion_codes[0]
                if code_obj.get("text"):
                    diagnostic_info["conclusion"] = code_obj["text"]
                elif code_obj.get("coding"):
                    for coding in code_obj["coding"]:
                        if coding.get("display"):
                            diagnostic_info["conclusion"] = coding["display"]
                            break
        
        # Extract category
        if diagnostic_resource.get("category"):
            categories = diagnostic_resource["category"]
            if categories and isinstance(categories, list):
                category_obj = categories[0]
                if category_obj.get("text"):
                    diagnostic_info["category"] = category_obj["text"]
                elif category_obj.get("coding"):
                    for coding in category_obj["coding"]:
                        if coding.get("display"):
                            diagnostic_info["category"] = coding["display"]
                            break
        
        return diagnostic_info

    def _extract_careplan_info(self, careplan_resource: Dict[str, Any]) -> Dict[str, Any]:
        """Extract care plan information from FHIR CarePlan resource"""
        careplan_info = {
            "display_name": "Unknown Care Plan",
            "date": None,
            "status": careplan_resource.get("status"),
            "intent": careplan_resource.get("intent"),
            "category": None,
            "description": None
        }
        
        # Extract care plan name/title
        if careplan_resource.get("title"):
            careplan_info["display_name"] = careplan_resource["title"]
        elif careplan_resource.get("description"):
            careplan_info["display_name"] = careplan_resource["description"][:100] + "..." if len(careplan_resource["description"]) > 100 else careplan_resource["description"]
        
        # Extract date
        if careplan_resource.get("created"):
            careplan_info["date"] = careplan_resource["created"]
        elif careplan_resource.get("period", {}).get("start"):
            careplan_info["date"] = careplan_resource["period"]["start"]
        
        # Extract category
        if careplan_resource.get("category"):
            categories = careplan_resource["category"]
            if categories and isinstance(categories, list):
                category_obj = categories[0]
                if category_obj.get("text"):
                    careplan_info["category"] = category_obj["text"]
                elif category_obj.get("coding"):
                    for coding in category_obj["coding"]:
                        if coding.get("display"):
                            careplan_info["category"] = coding["display"]
                            break
        
        # Extract description
        if careplan_resource.get("description"):
            careplan_info["description"] = careplan_resource["description"]
        
        return careplan_info

    def _calculate_data_completeness(self, patient_summary: Dict[str, Any]) -> float:
        """Calculate data completeness percentage"""
        total_sections = 8  # demographics, conditions, procedures, encounters, medications, practitioners, diagnostics, careplans
        completed_sections = 0
        
        if patient_summary["demographics"]:
            completed_sections += 1
        if patient_summary["conditions"]:
            completed_sections += 1
        if patient_summary["procedures"]:
            completed_sections += 1
        if patient_summary["encounters"]:
            completed_sections += 1
        if patient_summary["medications"]:
            completed_sections += 1
        if patient_summary["practitioners"]:
            completed_sections += 1
        if patient_summary["diagnostic_reports"]:
            completed_sections += 1
        if patient_summary["care_plans"]:
            completed_sections += 1
        
        return (completed_sections / total_sections) * 100


# Alias for backward compatibility
PublicFhirMcpServer = UniversalFhirMcpServer 