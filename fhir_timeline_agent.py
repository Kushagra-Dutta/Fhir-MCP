import asyncio
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from dotenv import load_dotenv
import os
from openai import AsyncOpenAI
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Prompt
from rich import print as rprint

# Load environment variables
load_dotenv()

# Import your FHIR tools
from fhir_tools import UniversalFhirMcpServer

# Configure logging to hide verbose tool logs
logging.getLogger("fhir_tools").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

class FHIRTimelineAgent:
    """Agent that creates treatment timelines from FHIR data - configured for firely_local server"""
    
    def __init__(self):
        self.fhir_server = UniversalFhirMcpServer()
        self.target_server = "firely_local"  # Only use firely_local server
        
        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("openai_api_key")
        if not api_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY or openai_api_key in your .env file")
        self.openai_client = AsyncOpenAI(api_key=api_key)
        
    async def initialize(self):
        """Initialize the FHIR server"""
        await self.fhir_server.initialize()
    
    async def close(self):
        """Close the FHIR server connection"""
        await self.fhir_server.close()
    
    async def process_patient_query(self, user_query: str) -> Dict[str, Any]:
        """
        Main agent flow:
        1. Extract clinical keywords from user query
        2. Map to FHIR codes (using firely_local)
        3. Find similar patients (using find_similar_patients_simple)
        4. Get comprehensive info for top 5 patients
        5. Generate timeline
        """
        
        # Step 1: Extract clinical keywords
        console = Console()
        console.print("[cyan]🔍 Step 1:[/cyan] Extracting clinical keywords...")
        clinical_data = await self.fhir_server.extract_clinical_keywords(user_query)
        
        # Check for required fields
        missing_fields = []
        if not clinical_data.get("age"):
            missing_fields.append("age")
        if not clinical_data.get("gender"):
            missing_fields.append("gender")
        if not clinical_data.get("conditions"):
            missing_fields.append("condition/diagnosis")
            
        if missing_fields:
            return {
                "status": "error",
                "error": "Missing required information",
                "missing_fields": missing_fields,
                "message": f"Please provide the following information: {', '.join(missing_fields)}",
                "example": "e.g., '45-year-old female with pancreatic adenocarcinoma'",
                "extracted_data": clinical_data
            }
        
        console.print(f"   [green]✅[/green] Age={clinical_data.get('age')}, Gender={clinical_data.get('gender')}")
        console.print(f"   [green]✅[/green] Conditions: {clinical_data.get('conditions', [])}")
        
        # Step 2: Map to FHIR codes (only firely_local)
        console.print(f"\n[cyan]🏥 Step 2:[/cyan] Mapping to FHIR codes in {self.target_server}...")
        try:
            # Use map_to_fhir_codes_fast with only firely_local server
            coded_data = await self.fhir_server.map_to_fhir_codes_fast(
                clinical_data=clinical_data,
                servers=[self.target_server],  # Only firely_local
                similarity_threshold=0.6,
                max_matches=3
            )
            console.print(f"   [green]✅[/green] Mapped {len(coded_data.get('condition_codes', []))} conditions")
            if coded_data.get('condition_codes'):
                for condition_code in coded_data['condition_codes']:
                    term = condition_code.get('term', 'Unknown')
                    matches = len(condition_code.get('matches', []))
                    console.print(f"      [blue]📋[/blue] {term}: {matches} matches")
        except Exception as e:
            console.print(f"   [red]❌ Error:[/red] {e}")
            # Continue with original clinical data if mapping fails
            coded_data = clinical_data
        
        # Step 3: Find similar patients using find_similar_patients_simple
        console.print(f"\n[cyan]👥 Step 3:[/cyan] Finding similar patients in {self.target_server}...")
        try:
            similar_patients = await self.fhir_server.find_similar_patients_simple(
                clinical_data=coded_data,
                servers=[self.target_server],  # Only firely_local
                max_results=5,
                age_tolerance=10
            )
            
            # find_similar_patients_simple returns patients in "patients" key, not "similar_patients"
            if not similar_patients.get("patients"):
                return {
                    "status": "error",
                    "error": f"No similar patients found in {self.target_server}",
                    "clinical_data": coded_data,
                    "suggestion": f"Check that {self.target_server} server has patient data",
                    "search_result": similar_patients
                }
            
            console.print(f"   [green]✅[/green] Found {len(similar_patients.get('patients', []))} similar patients")
            console.print(f"   [blue]📊[/blue] Total found: {similar_patients.get('total_found', 0)}")
            
        except Exception as e:
            console.print(f"   [red]❌ Error:[/red] {e}")
            return {
                "status": "error",
                "error": f"Failed to find similar patients: {str(e)}",
                "clinical_data": coded_data
            }
        
        # Step 4: Get comprehensive info for each similar patient
        console.print(f"\n[cyan]📋 Step 4:[/cyan] Getting comprehensive medical data...")
        patient_medical_data = []
        patient_ids = []
        
        # find_similar_patients_simple returns patients in "patients" key
        for i, patient_match in enumerate(similar_patients["patients"]):
            patient_id = patient_match.get("patient_id")
            if not patient_id:
                continue
                
            patient_ids.append(patient_id)
            total_score = patient_match.get("total_score", 0)
            matched_condition = patient_match.get("matched_condition", "Unknown")
            
            console.print(f"   [blue]📝[/blue] Patient {i+1}: {patient_id[:8]}... (Score: {total_score})")
            
            try:
                comprehensive_info = await self.fhir_server.get_comprehensive_patient_info(
                    patient_id=patient_id,
                    server_name=self.target_server
                )
                
                if comprehensive_info:
                    # Add match info to the data
                    comprehensive_info["match_score"] = patient_match.get("total_score", 0)
                    comprehensive_info["matched_condition"] = matched_condition
                    comprehensive_info["matched_code"] = patient_match.get("matched_code", "")
                    comprehensive_info["patient_id"] = patient_id
                    patient_medical_data.append(comprehensive_info)
                    
                    # Print summary of collected data
                    conditions_count = len(comprehensive_info.get('conditions', []))
                    procedures_count = len(comprehensive_info.get('procedures', []))
                    medications_count = len(comprehensive_info.get('medications', []))
                    encounters_count = len(comprehensive_info.get('encounters', []))
                    
                    console.print(f"      [green]✅[/green] {conditions_count} conditions, {procedures_count} procedures, {medications_count} medications, {encounters_count} encounters")
                else:
                    console.print(f"      [yellow]⚠️[/yellow] No data returned")
                
            except Exception as e:
                console.print(f"      [red]❌ Error:[/red] {str(e)}")
                continue
        
        if not patient_medical_data:
            return {
                "status": "error",
                "error": "No comprehensive patient data could be retrieved",
                "patient_ids_attempted": patient_ids
            }
        
        console.print(f"   [green]✅[/green] Retrieved data for {len(patient_medical_data)} patients")
        
        # Step 5: Generate separate timelines for each patient using OpenAI
        console.print(f"\n[cyan]⏰ Step 5:[/cyan] Generating individual care plans with AI...")
        individual_care_plans = []
        
        for i, patient_data in enumerate(patient_medical_data):
            patient_id = patient_data.get("patient_id", f"Patient_{i+1}")
            console.print(f"   [blue]📋[/blue] Creating care plan for Patient {i+1}: {patient_id[:8]}...")
            
            try:
                timeline_data = await self._generate_individual_timeline_with_openai(
                    original_query=user_query,
                    clinical_data=coded_data,
                    individual_patient_data=patient_data
                )
                
                # Add patient-specific metadata
                care_plan = {
                    "patient_id": patient_id,
                    "patient_sequence": i + 1,
                    "match_score": patient_data.get("match_score", 0),
                    "matched_condition": patient_data.get("matched_condition", "Unknown"),
                    "patient_profile": timeline_data.get("patient_profile", {}),
                    "timeline": timeline_data.get("timeline", []),
                    "clinical_outcomes": timeline_data.get("clinical_outcomes", {}),
                    "data_sources": {
                        "server": self.target_server,
                        "patient_id": patient_id,
                        "ai_generated": True,
                        "ai_model": "OpenAI GPT-4",
                        "conditions_count": len(patient_data.get('conditions', [])),
                        "procedures_count": len(patient_data.get('procedures', [])),
                        "medications_count": len(patient_data.get('medications', [])),
                        "encounters_count": len(patient_data.get('encounters', []))
                    }
                }
                
                individual_care_plans.append(care_plan)
                console.print(f"      [green]✅[/green] Generated {len(timeline_data.get('timeline', []))} timeline events")
                
            except Exception as e:
                console.print(f"      [red]❌ Error:[/red] {str(e)}")
                # Add error care plan
                individual_care_plans.append({
                    "patient_id": patient_id,
                    "patient_sequence": i + 1,
                    "match_score": patient_data.get("match_score", 0),
                    "error": f"Failed to generate timeline: {str(e)}",
                    "data_sources": {
                        "server": self.target_server,
                        "patient_id": patient_id,
                        "ai_generated": False
                    }
                })
        
        # Final output - return all individual care plans
        result = {
            "status": "success",
            "query": user_query,
            "target_patient_profile": {
                "age": clinical_data.get("age"),
                "gender": clinical_data.get("gender"),
                "diagnosis": clinical_data.get("conditions", []),
                "stage": clinical_data.get("stage"),
                "biomarkers": clinical_data.get("biomarkers", [])
            },
            "individual_care_plans": individual_care_plans,
            "summary": {
                "total_care_plans_generated": len(individual_care_plans),
                "successful_plans": len([cp for cp in individual_care_plans if "error" not in cp]),
                "failed_plans": len([cp for cp in individual_care_plans if "error" in cp])
            },
            "data_sources": {
                "server": self.target_server,
                "similar_patients_analyzed": len(patient_medical_data),
                "patient_ids": patient_ids,
                "clinical_keywords": clinical_data,
                "fhir_codes": coded_data.get('condition_codes', []),
                "ai_generated": True,
                "ai_model": "OpenAI GPT-4"
            }
        }
        
        console.print(f"[green]✅ Individual care plan generation completed![/green]")
        return result
    
    async def _generate_individual_timeline_with_openai(self, original_query: str, 
                                                       clinical_data: Dict[str, Any],
                                                       individual_patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a detailed clinical treatment timeline for ONE specific patient using OpenAI
        """
        
        console = Console()
        
        # Extract data for this specific patient only
        patient_procedures = individual_patient_data.get("procedures", [])
        patient_medications = individual_patient_data.get("medications", [])
        patient_encounters = individual_patient_data.get("encounters", [])
        patient_conditions = individual_patient_data.get("conditions", [])
        patient_demographics = individual_patient_data.get("demographics", {})
        
        console.print(f"      [blue]📊[/blue] Patient data: {len(patient_procedures)} procedures, {len(patient_encounters)} encounters")
        
        # Create detailed prompt for this specific patient
        prompt = self._create_individual_timeline_prompt(
            original_query, clinical_data, individual_patient_data,
            patient_procedures, patient_medications, patient_encounters, patient_conditions, patient_demographics
        )
        
        try:
            console.print(f"      [magenta]🤖[/magenta] Sending to OpenAI...")
            
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a senior oncologist and medical timeline specialist. Generate detailed, clinically accurate treatment timelines based on individual patient data. Use proper medical terminology and realistic treatment schedules. ALWAYS return only valid JSON without any markdown formatting, code blocks, or explanatory text."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=8000
            )
            
            # Parse OpenAI response
            ai_response = response.choices[0].message.content
            
            # Try to parse as JSON (handle various formats OpenAI might return)
            try:
                # First try direct parsing
                timeline_data = json.loads(ai_response)
                return timeline_data
                
            except json.JSONDecodeError:
                # Try to extract JSON from code blocks or markdown
                import re
                json_matches = re.findall(r'```(?:json)?\s*(\{.*?\})\s*```', ai_response, re.DOTALL)
                if not json_matches:
                    # Try to find JSON without code blocks
                    json_matches = re.findall(r'(\{.*\})', ai_response, re.DOTALL)
                
                for json_text in json_matches:
                    try:
                        timeline_data = json.loads(json_text)
                        return timeline_data
                    except json.JSONDecodeError:
                        continue
                
                console.print(f"      [yellow]⚠️[/yellow] Using enhanced fallback parsing...")
                # Fallback to structured parsing with the AI response
                return self._parse_ai_timeline_response(ai_response, original_query, clinical_data)
                
        except Exception as e:
            console.print(f"      [red]❌ Error:[/red] {str(e)}")
            # Fallback to basic timeline for this patient
            return self._generate_fallback_individual_timeline(original_query, clinical_data, individual_patient_data)
    
    async def _generate_clinical_timeline_with_openai(self, original_query: str, 
                                                     clinical_data: Dict[str, Any],
                                                     similar_patients_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a detailed clinical treatment timeline using OpenAI based on similar patients' data
        """
        
        # Aggregate all data from similar patients
        all_procedures = []
        all_medications = []
        all_encounters = []
        all_conditions = []
        
        console = Console()
        
        for i, patient_data in enumerate(similar_patients_data):
            if patient_data.get("procedures"):
                procedures = patient_data["procedures"]
                all_procedures.extend(procedures)
                
            if patient_data.get("medications"):
                medications = patient_data["medications"]
                all_medications.extend(medications)
                
            if patient_data.get("encounters"):
                encounters = patient_data["encounters"]
                all_encounters.extend(encounters)
                
            if patient_data.get("conditions"):
                conditions = patient_data["conditions"]
                all_conditions.extend(conditions)
        
        console.print(f"   [blue]📊[/blue] Aggregated: {len(all_procedures)} procedures, {len(all_encounters)} encounters")
        
        # Create detailed prompt for OpenAI
        prompt = self._create_timeline_prompt(
            original_query, clinical_data, similar_patients_data,
            all_procedures, all_medications, all_encounters, all_conditions
        )
        
        try:
            console.print(f"   [magenta]🤖[/magenta] Sending to OpenAI...")
            
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a senior oncologist and medical timeline specialist. Generate detailed, clinically accurate treatment timelines based on real patient data. Use proper medical terminology and realistic treatment schedules. ALWAYS return only valid JSON without any markdown formatting, code blocks, or explanatory text."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=8000
            )
            
            # Parse OpenAI response
            ai_response = response.choices[0].message.content
            
            # Try to parse as JSON (handle various formats OpenAI might return)
            try:
                # First try direct parsing
                timeline_data = json.loads(ai_response)
                console.print(f"   [green]✅[/green] Generated {len(timeline_data.get('timeline', []))} timeline events")
                return timeline_data
                
            except json.JSONDecodeError:
                # Try to extract JSON from code blocks or markdown
                import re
                json_matches = re.findall(r'```(?:json)?\s*(\{.*?\})\s*```', ai_response, re.DOTALL)
                if not json_matches:
                    # Try to find JSON without code blocks
                    json_matches = re.findall(r'(\{.*\})', ai_response, re.DOTALL)
                
                for json_text in json_matches:
                    try:
                        timeline_data = json.loads(json_text)
                        console.print(f"   [green]✅[/green] Generated {len(timeline_data.get('timeline', []))} timeline events")
                        return timeline_data
                    except json.JSONDecodeError:
                        continue
                
                console.print(f"   [yellow]⚠️[/yellow] Using enhanced fallback parsing...")
                # Fallback to structured parsing with the AI response
                return self._parse_ai_timeline_response(ai_response, original_query, clinical_data)
                
        except Exception as e:
            console.print(f"   [red]❌ Error:[/red] {str(e)}")
            # Fallback to basic timeline
            return self._generate_fallback_timeline(original_query, clinical_data, similar_patients_data)
    
    def _create_timeline_prompt(self, original_query: str, clinical_data: Dict[str, Any],
                               similar_patients_data: List[Dict[str, Any]],
                               all_procedures: List[Dict], all_medications: List[Dict],
                               all_encounters: List[Dict], all_conditions: List[Dict]) -> str:
        """Create a detailed prompt for OpenAI timeline generation"""
        
        # Get detailed procedures with dates (all of them, not just names)
        detailed_procedures = []
        for i, proc in enumerate(all_procedures):
            proc_info = {
                "name": proc.get('display_name', f'Procedure {i+1}'),
                "date": proc.get('date', proc.get('performed_date', 'Date not specified')),
                "status": proc.get('status', 'Unknown'),
                "category": proc.get('category', 'Unknown category')
            }
            detailed_procedures.append(proc_info)
        
        # Get detailed encounters with dates
        detailed_encounters = []
        for i, enc in enumerate(all_encounters):
            enc_info = {
                "type": enc.get('display_name', f'Encounter {i+1}'),
                "date": enc.get('date', 'Date not specified'),
                "class": enc.get('class', 'Unknown'),
                "status": enc.get('status', 'Unknown'),
                "practitioner": enc.get('practitioner', 'Unknown practitioner')
            }
            detailed_encounters.append(enc_info)
        
        # Get detailed medications with dates
        detailed_medications = []
        for i, med in enumerate(all_medications):
            med_info = {
                "name": med.get('display_name', f'Medication {i+1}'),
                "date": med.get('date', 'Date not specified'),
                "dosage": med.get('dosage', 'Dosage not specified'),
                "status": med.get('status', 'Unknown'),
                "intent": med.get('intent', 'Unknown')
            }
            detailed_medications.append(med_info)
        
        # Get condition details
        condition_names = [c.get('display_name', 'Unknown') for c in all_conditions[:5]]
        
        prompt = f"""
Based on the following real patient data from FHIR servers, generate a detailed clinical treatment timeline in JSON format similar to the example provided.

PATIENT QUERY: {original_query}

EXTRACTED CLINICAL DATA:
- Age: {clinical_data.get('age')}
- Gender: {clinical_data.get('gender')}
- Conditions: {clinical_data.get('conditions', [])}
- Stage: {clinical_data.get('stage', 'To be determined')}
- Biomarkers: {clinical_data.get('biomarkers', [])}

REAL PATIENT DATA FROM FHIR SERVER ({len(similar_patients_data)} patients analyzed):
- Conditions found: {condition_names}

DETAILED PROCEDURES ({len(detailed_procedures)} procedures with dates):
{json.dumps(detailed_procedures, indent=2)}

DETAILED ENCOUNTERS ({len(detailed_encounters)} encounters with dates):
{json.dumps(detailed_encounters, indent=2)}

DETAILED MEDICATIONS ({len(detailed_medications)} medications with dates):
{json.dumps(detailed_medications, indent=2)}

REQUIRED JSON OUTPUT FORMAT (return only valid JSON, no markdown or explanatory text):
{{
  "patient_profile": {{
    "age_sex": "45-year-old Female",
    "diagnosis": "Invasive ductal carcinoma, left breast",
    "stage": "IIIA (T2N2M0)",
    "biomarkers": ["HER2 Positive", "EGFR Negative"]
  }},
  "timeline": [
    {{
      "step": 1,
      "title": "Initial Diagnosis",
      "day": "Day 0",
      "status": "completed",
      "description": "Diagnosed with HER2+ invasive ductal carcinoma",
      "details": "Tumor size: 3.2cm, 2 positive lymph nodes detected"
    }},
    {{
      "step": 2,
      "title": "Neoadjuvant Chemotherapy - AC",
      "day": "Day 14",
      "status": "completed",
      "description": "Started 4 cycles of Adriamycin + Cyclophosphamide",
      "details": "Well tolerated, mild nausea managed with antiemetics"
    }},
    {{
      "step": 3,
      "title": "Paclitaxel + Trastuzumab",
      "day": "Day 84",
      "status": "completed",
      "description": "Switched to weekly Paclitaxel with Trastuzumab",
      "details": "12 weeks planned, cardiac function monitored"
    }}
  ],
  "clinical_outcomes": {{
    "treatment_response": "Partial pathological response",
    "survival_status": "Disease-free at 18 months",
    "toxicity": "Grade 2 neuropathy, resolved",
    "quality_of_life": "ECOG 0, excellent functional status"
  }}
}}

INSTRUCTIONS:
1. Create a COMPREHENSIVE timeline using ALL the procedures and encounters listed above with their actual dates
2. Sort all procedures and encounters chronologically by date 
3. Create timeline events for each major procedure and significant encounter
4. Use the actual procedure names, dates, and details provided
5. Include specific drug names, dosages, and treatment durations where available
6. Group similar procedures (like radiation therapy sessions) into logical treatment phases
7. Use proper medical terminology and the actual data provided
8. Make the timeline clinically accurate for pancreatic adenocarcinoma
9. Include staging information if applicable to the condition
10. Provide realistic clinical outcomes based on the data
11. Ensure timeline progression is logical (diagnosis → workup → treatment → follow-up)
12. Use "completed" status for past treatments, "ongoing" for current, "planned" for future
13. AIM for 15-25 timeline events to cover the comprehensive treatment journey
14. Use the actual dates from the procedures/encounters when available

Generate the JSON output only, no additional text.
"""
        
        return prompt
    
    def _create_individual_timeline_prompt(self, original_query: str, clinical_data: Dict[str, Any],
                                          individual_patient_data: Dict[str, Any],
                                          patient_procedures: List[Dict], patient_medications: List[Dict],
                                          patient_encounters: List[Dict], patient_conditions: List[Dict],
                                          patient_demographics: Dict[str, Any]) -> str:
        """Create a detailed prompt for OpenAI timeline generation for ONE specific patient"""
        
        # Get patient age and gender from demographics
        patient_age = patient_demographics.get('age', 'Unknown age')
        patient_gender = patient_demographics.get('gender', 'Unknown gender')
        patient_birth_date = patient_demographics.get('birth_date', 'Unknown birth date')
        
        # Get detailed procedures with dates for this patient only
        detailed_procedures = []
        for i, proc in enumerate(patient_procedures):
            proc_info = {
                "name": proc.get('display_name', f'Procedure {i+1}'),
                "date": proc.get('date', proc.get('performed_date', 'Date not specified')),
                "status": proc.get('status', 'Unknown'),
                "category": proc.get('category', 'Unknown category')
            }
            detailed_procedures.append(proc_info)
        
        # Get detailed encounters with dates for this patient only
        detailed_encounters = []
        for i, enc in enumerate(patient_encounters):
            enc_info = {
                "type": enc.get('display_name', f'Encounter {i+1}'),
                "date": enc.get('date', 'Date not specified'),
                "class": enc.get('class', 'Unknown'),
                "status": enc.get('status', 'Unknown'),
                "practitioner": enc.get('practitioner', 'Unknown practitioner')
            }
            detailed_encounters.append(enc_info)
        
        # Get detailed medications with dates for this patient only
        detailed_medications = []
        for i, med in enumerate(patient_medications):
            med_info = {
                "name": med.get('display_name', f'Medication {i+1}'),
                "date": med.get('date', 'Date not specified'),
                "dosage": med.get('dosage', 'Dosage not specified'),
                "status": med.get('status', 'Unknown'),
                "intent": med.get('intent', 'Unknown')
            }
            detailed_medications.append(med_info)
        
        # Get condition details for this patient only
        condition_names = [c.get('display_name', 'Unknown') for c in patient_conditions[:5]]
        
        # Get match information
        match_score = individual_patient_data.get("match_score", 0)
        matched_condition = individual_patient_data.get("matched_condition", "Unknown")
        patient_id = individual_patient_data.get("patient_id", "Unknown")
        
        prompt = f"""
Generate a detailed clinical treatment timeline for ONE SPECIFIC PATIENT based on their real FHIR data.

ORIGINAL PATIENT QUERY: {original_query}

TARGET PATIENT PROFILE (what we're looking for):
- Age: {clinical_data.get('age')}
- Gender: {clinical_data.get('gender')}
- Conditions: {clinical_data.get('conditions', [])}
- Stage: {clinical_data.get('stage', 'To be determined')}
- Biomarkers: {clinical_data.get('biomarkers', [])}

ACTUAL PATIENT DATA FROM FHIR (Patient ID: {patient_id}):
- Patient Age: {patient_age}
- Patient Gender: {patient_gender}
- Birth Date: {patient_birth_date}
- Match Score: {match_score}/100
- Matched Condition: {matched_condition}
- Conditions Found: {condition_names}

THIS PATIENT'S SPECIFIC MEDICAL HISTORY:

PROCEDURES ({len(detailed_procedures)} total):
{json.dumps(detailed_procedures, indent=2)}

ENCOUNTERS ({len(detailed_encounters)} total):
{json.dumps(detailed_encounters, indent=2)}

MEDICATIONS ({len(detailed_medications)} total):
{json.dumps(detailed_medications, indent=2)}

Generate a personalized care plan timeline for THIS SPECIFIC PATIENT based on their actual medical history above.

REQUIRED JSON OUTPUT FORMAT (return only valid JSON, no markdown or explanatory text):
{{
  "patient_profile": {{
    "patient_id": "{patient_id}",
    "age_sex": "{patient_age} {patient_gender}",
    "actual_diagnosis": "Primary diagnosis from patient conditions",
    "stage": "Based on patient data or 'To be determined'",
    "biomarkers": ["Based on actual patient data"],
    "match_score": {match_score}
  }},
  "timeline": [
    {{
      "step": 1,
      "title": "Based on actual patient encounters/procedures",
      "day": "Based on actual dates from patient data",
      "status": "completed/planned based on patient history",
      "description": "Real description from patient's medical history",
      "details": "Specific details from this patient's actual data"
    }}
  ],
  "clinical_outcomes": {{
    "treatment_response": "Based on this patient's actual treatment history",
    "survival_status": "Based on this patient's current status",
    "toxicity": "Based on this patient's medication history",
    "quality_of_life": "Projected from this patient's encounter patterns"
  }}
}}

Focus on this ONE patient's actual medical journey, not a generic timeline. Use the real dates, procedures, and medications from their FHIR data.
"""
        
        return prompt
    
    def _generate_fallback_individual_timeline(self, original_query: str, clinical_data: Dict[str, Any], 
                                             individual_patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a basic timeline for one specific patient when OpenAI fails"""
        
        patient_id = individual_patient_data.get("patient_id", "Unknown")
        patient_procedures = individual_patient_data.get("procedures", [])
        patient_medications = individual_patient_data.get("medications", [])
        patient_encounters = individual_patient_data.get("encounters", [])
        patient_conditions = individual_patient_data.get("conditions", [])
        patient_demographics = individual_patient_data.get("demographics", {})
        
        # Create basic timeline from actual patient data
        timeline_events = []
        event_step = 1
        
        # Add diagnosis events from conditions
        for condition in patient_conditions[:3]:
            timeline_events.append({
                "step": event_step,
                "title": f"Diagnosis: {condition.get('display_name', 'Unknown condition')}",
                "day": f"Day 0",
                "status": "completed",
                "description": f"Patient diagnosed with {condition.get('display_name', 'condition')}",
                "details": f"Condition onset: {condition.get('onset_date', 'Date not specified')}"
            })
            event_step += 1
        
        # Add procedure events
        for i, procedure in enumerate(patient_procedures[:5]):
            timeline_events.append({
                "step": event_step,
                "title": f"Procedure: {procedure.get('display_name', f'Procedure {i+1}')}",
                "day": f"Day {event_step * 7}",
                "status": procedure.get('status', 'completed'),
                "description": f"Patient underwent {procedure.get('display_name', 'procedure')}",
                "details": f"Performed on: {procedure.get('date', 'Date not specified')}"
            })
            event_step += 1
        
        # Add medication events
        for i, medication in enumerate(patient_medications[:3]):
            timeline_events.append({
                "step": event_step,
                "title": f"Medication: {medication.get('display_name', f'Medication {i+1}')}",
                "day": f"Day {event_step * 7}",
                "status": medication.get('status', 'active'),
                "description": f"Started on {medication.get('display_name', 'medication')}",
                "details": f"Dosage: {medication.get('dosage', 'Not specified')}"
            })
            event_step += 1
        
        return {
            "patient_profile": {
                "patient_id": patient_id,
                "age_sex": f"{patient_demographics.get('age', 'Unknown')} {patient_demographics.get('gender', 'Unknown')}",
                "actual_diagnosis": patient_conditions[0].get('display_name', 'Unknown') if patient_conditions else 'Unknown',
                "stage": "Based on patient data",
                "biomarkers": [],
                "match_score": individual_patient_data.get("match_score", 0)
            },
            "timeline": timeline_events,
            "clinical_outcomes": {
                "treatment_response": f"Based on {len(patient_procedures)} procedures performed",
                "survival_status": f"Patient has {len(patient_encounters)} recorded encounters",
                "toxicity": f"On {len(patient_medications)} medications",
                "quality_of_life": "Based on encounter frequency and medication regimen"
            }
        }
    
    def _parse_ai_timeline_response(self, ai_response: str, original_query: str, clinical_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse AI response when JSON parsing fails - try to extract timeline info"""
        
        # Try to extract timeline events from text
        timeline_events = []
        
        # Look for timeline-like patterns in the response
        import re
        
        # Pattern to find step-like entries
        step_patterns = [
            r'(?:Step\s*)?(\d+)[\.\:\-\s]*([^\n]+?)(?:Day\s*(\d+))?[^\n]*?(?:\n([^\n]+))?',
            r'(\d+)[\.\:\-\s]*([^\n]+?)(?:\(Day\s*(\d+)\))?[^\n]*?(?:\n([^\n]+))?'
        ]
        
        for pattern in step_patterns:
            matches = re.findall(pattern, ai_response, re.MULTILINE | re.IGNORECASE)
            if matches:
                for i, match in enumerate(matches[:7]):  # Limit to 7 events
                    step_num = match[0] if match[0] else str(i + 1)
                    title = match[1].strip() if match[1] else f"Treatment Step {step_num}"
                    day = f"Day {match[2]}" if match[2] else f"Day {i * 14}"
                    details = match[3].strip() if len(match) > 3 and match[3] else "Based on clinical analysis"
                    
                    timeline_events.append({
                        "step": int(step_num) if step_num.isdigit() else i + 1,
                        "title": title[:100],  # Limit title length
                        "day": day,
                        "status": "completed" if i < 4 else ("ongoing" if i == 4 else "planned"),
                        "description": title[:200],  # Use title as description
                        "details": details[:300] if details else "Clinical timeline event"
                    })
                break
        
        # If no patterns found, create basic timeline
        if not timeline_events:
            timeline_events = [
                {
                    "step": 1,
                    "title": "Initial Diagnosis",
                    "day": "Day 0",
                    "status": "completed",
                    "description": f"Diagnosed with {', '.join(clinical_data.get('conditions', ['condition']))}",
                    "details": f"Based on query: {original_query}"
                },
                {
                    "step": 2,
                    "title": "Treatment Planning",
                    "day": "Day 7-14",
                    "status": "completed",
                    "description": "Comprehensive evaluation and treatment planning",
                    "details": "Clinical workup and multidisciplinary consultation"
                }
            ]
        
        # Try to extract patient profile information
        age_sex = f"{clinical_data.get('age', 'Unknown')}-year-old {clinical_data.get('gender', 'Unknown')}"
        diagnosis = ', '.join(clinical_data.get('conditions', ['Unknown condition']))
        
        # Look for stage information in AI response
        stage_match = re.search(r'Stage\s*([IV]+|[1-4]|T\d+N\d+M\d+)', ai_response, re.IGNORECASE)
        stage = stage_match.group(1) if stage_match else clinical_data.get('stage', 'To be determined')
        
        # Look for biomarkers in AI response
        biomarker_patterns = [
            r'HER2[\+\-]?',
            r'ER[\+\-]?',
            r'PR[\+\-]?', 
            r'EGFR[\+\-]?',
            r'PD-L1[\+\-]?'
        ]
        biomarkers = clinical_data.get('biomarkers', [])
        for pattern in biomarker_patterns:
            matches = re.findall(pattern, ai_response, re.IGNORECASE)
            biomarkers.extend([m for m in matches if m not in biomarkers])
        
        fallback_data = {
            "patient_profile": {
                "age_sex": age_sex,
                "diagnosis": diagnosis,
                "stage": stage,
                "biomarkers": biomarkers[:5]  # Limit biomarkers
            },
            "timeline": timeline_events,
            "clinical_outcomes": {
                "treatment_response": "Analysis based on similar patients",
                "survival_status": "Projected from similar patient outcomes", 
                "toxicity": "Common side effects from similar cases",
                "quality_of_life": "Expected outcomes based on real patient data"
            }
        }
        
        console = Console()
        console.print(f"   [blue]🔄[/blue] Extracted {len(timeline_events)} events from AI response")
        return fallback_data
    
    def display_timeline_table(self, result: Dict[str, Any]) -> None:
        """Display the timeline in a beautiful table format"""
        
        console = Console()
        
        # Display patient profile
        if result.get('patient_profile'):
            profile = result['patient_profile']
            
            # Patient Profile Panel
            profile_text = f"""[bold cyan]Age/Sex:[/bold cyan] {profile.get('age_sex', 'Unknown')}
[bold cyan]Diagnosis:[/bold cyan] {profile.get('diagnosis', 'Unknown')}
[bold cyan]Stage:[/bold cyan] {profile.get('stage', 'Not specified')}
[bold cyan]Biomarkers:[/bold cyan] {', '.join(profile.get('biomarkers', [])) or 'None specified'}"""
            
            console.print(Panel(profile_text, title="👤 Patient Profile", border_style="blue"))
        
        # Timeline Table
        if result.get('timeline'):
            timeline = result['timeline']
            
            table = Table(title="🏥 Treatment Timeline", show_header=True, header_style="bold magenta")
            table.add_column("Step", style="cyan", no_wrap=True, width=6)
            table.add_column("Day", style="green", no_wrap=True, width=12)
            table.add_column("Title", style="bold yellow", width=30)
            table.add_column("Status", style="bold", width=12)
            table.add_column("Description", width=40)
            table.add_column("Details", style="dim", width=50)
            
            for event in timeline:
                # Color-code status
                status = event.get('status', 'unknown')
                if status == 'completed':
                    status_color = "[green]✅ Completed[/green]"
                elif status == 'ongoing':
                    status_color = "[yellow]🔄 Ongoing[/yellow]"
                elif status == 'planned':
                    status_color = "[blue]📅 Planned[/blue]"
                else:
                    status_color = f"[dim]{status}[/dim]"
                
                table.add_row(
                    str(event.get('step', '')),
                    event.get('day', ''),
                    event.get('title', ''),
                    status_color,
                    event.get('description', ''),
                    event.get('details', '')
                )
            
            console.print(table)
        
        # Clinical Outcomes
        if result.get('clinical_outcomes'):
            outcomes = result['clinical_outcomes']
            
            outcomes_text = f"""[bold green]Treatment Response:[/bold green] {outcomes.get('treatment_response', 'Unknown')}
[bold green]Survival Status:[/bold green] {outcomes.get('survival_status', 'Unknown')}
[bold green]Toxicity:[/bold green] {outcomes.get('toxicity', 'Unknown')}
[bold green]Quality of Life:[/bold green] {outcomes.get('quality_of_life', 'Unknown')}"""
            
            console.print(Panel(outcomes_text, title="📊 Clinical Outcomes", border_style="green"))
        
        # Data Sources
        if result.get('data_sources'):
            sources = result['data_sources']
            
            sources_text = f"""[bold blue]FHIR Server:[/bold blue] {sources.get('server', 'Unknown')}
[bold blue]Similar Patients:[/bold blue] {sources.get('similar_patients_analyzed', 0)}
[bold blue]AI Model:[/bold blue] {sources.get('ai_model', 'Unknown')}
[bold blue]Patient IDs:[/bold blue] {', '.join(sources.get('patient_ids', []))}"""
            
            console.print(Panel(sources_text, title="🔗 Data Sources", border_style="cyan"))
    
    def display_error_table(self, result: Dict[str, Any]) -> None:
        """Display error information in a formatted way"""
        
        console = Console()
        
        error_text = f"""[bold red]Error:[/bold red] {result.get('error', 'Unknown error')}
[bold red]Status:[/bold red] {result.get('status', 'Failed')}"""
        
        if result.get('missing_fields'):
            error_text += f"\n[bold red]Missing Fields:[/bold red] {', '.join(result['missing_fields'])}"
        
        if result.get('suggestion'):
            error_text += f"\n[bold yellow]Suggestion:[/bold yellow] {result['suggestion']}"
        
        console.print(Panel(error_text, title="❌ Error", border_style="red"))
    
    def _generate_fallback_timeline(self, original_query: str, clinical_data: Dict[str, Any], 
                                  similar_patients_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate fallback timeline when OpenAI fails"""
        
        console = Console()
        console.print(f"   [blue]🔄[/blue] Generating fallback timeline...")
        
        return {
            "patient_profile": {
                "age_sex": f"{clinical_data.get('age', 'Unknown')}-year-old {clinical_data.get('gender', 'Unknown')}",
                "diagnosis": ', '.join(clinical_data.get('conditions', ['Unknown condition'])),
                "stage": clinical_data.get('stage', 'To be determined'),
                "biomarkers": clinical_data.get('biomarkers', [])
            },
            "timeline": [
                {
                    "step": 1,
                    "title": "Initial Diagnosis",
                    "day": "Day 0",
                    "status": "completed",
                    "description": f"Patient diagnosed with {', '.join(clinical_data.get('conditions', ['condition']))}",
                    "details": f"Based on query: {original_query}"
                },
                {
                    "step": 2,
                    "title": "Treatment Planning",
                    "day": "Day 7-14",
                    "status": "completed",
                    "description": "Comprehensive evaluation and treatment planning",
                    "details": f"Based on {len(similar_patients_data)} similar patients from {self.target_server}"
                }
            ],
            "clinical_outcomes": {
                "treatment_response": f"Analysis based on {len(similar_patients_data)} similar patients",
                "survival_status": "Projected from similar patient outcomes",
                "toxicity": "Common side effects from similar cases",
                "quality_of_life": "Expected outcomes based on real patient data"
            }
        }


async def interactive_main():
    """Interactive CLI for FHIR Timeline Agent"""
    
    console = Console()
    agent = FHIRTimelineAgent()
    
    # Welcome message
    console.print(Panel.fit(
        "[bold blue]🏥 FHIR Clinical Timeline Generator[/bold blue]\n"
        "[cyan]Generate detailed treatment timelines from patient queries using real FHIR data[/cyan]\n\n"
        "[yellow]Powered by:[/yellow] OpenAI GPT-4 + firely_local FHIR server",
        border_style="blue"
    ))
    
    try:
        # Initialize agent
        console.print("[yellow]🔧 Initializing FHIR agent...[/yellow]")
        await agent.initialize()
        console.print("[green]✅ Agent initialized successfully![/green]\n")
        
        while True:
            # Get user input
            console.print("[bold cyan]Enter a patient query (or 'quit' to exit):[/bold cyan]")
            console.print("[dim]Examples:[/dim]")
            console.print("[dim]  • 45-year-old male with pancreatic adenocarcinoma[/dim]")
            console.print("[dim]  • 62-year-old female with HER2+ breast cancer[/dim]")
            console.print("[dim]  • 58-year-old male with stage IIIA lung adenocarcinoma[/dim]\n")
            
            user_query = Prompt.ask("[bold green]Patient Query")
            
            if user_query.lower() in ['quit', 'exit', 'q']:
                console.print("[yellow]👋 Goodbye![/yellow]")
                break
            
            if not user_query.strip():
                console.print("[red]❌ Please enter a valid query[/red]\n")
                continue
            
            # Process query
            console.print(f"\n[bold blue]🔍 Processing:[/bold blue] {user_query}")
            console.print("=" * 80)
            
            try:
                result = await agent.process_patient_query(user_query)
                
                console.print("\n")
                
                if result["status"] == "success":
                    # Display beautiful table
                    agent.display_timeline_table(result)
                    
                    # Summary
                    console.print(f"\n[bold green]✅ SUCCESS![/bold green] Generated {len(result['timeline'])} timeline events")
                    console.print(f"[blue]📊 Based on {result['data_sources']['similar_patients_analyzed']} similar patients from {result['data_sources']['server']}[/blue]")
                    console.print(f"[blue]🤖 Generated using {result['data_sources']['ai_model']}[/blue]")
                    
                else:
                    # Display error
                    agent.display_error_table(result)
                
            except Exception as e:
                console.print(f"[red]❌ Error processing query: {str(e)}[/red]")
                console.print("[dim]Check logs for details[/dim]")
            
            # Ask if user wants to continue
            console.print("\n" + "="*80)
            continue_choice = Prompt.ask("[cyan]Generate another timeline?[/cyan]", choices=["y", "n"], default="y")
            
            if continue_choice.lower() == 'n':
                console.print("[yellow]👋 Thank you for using FHIR Timeline Generator![/yellow]")
                break
            
            console.print("\n")
            
    except KeyboardInterrupt:
        console.print("\n[yellow]👋 Interrupted by user. Goodbye![/yellow]")
    except Exception as e:
        console.print(f"\n[red]❌ Fatal error: {str(e)}[/red]")
        import traceback
        traceback.print_exc()
    finally:
        await agent.close()

# Example usage for testing
async def example_demo():
    """Demo with example queries"""
    # Example user queries
    test_queries = [
        "45-year-old male with pancreatic adenocarcinoma",
        "Female patient with breast cancer", 
        "52 years old with lung adenocarcinoma",
    ]
    
    console = Console()
    agent = FHIRTimelineAgent()
    
    try:
        await agent.initialize()
        
        for i, query in enumerate(test_queries[:1]):  # Test with first query only
            console.print(f"\n[bold blue]🏥 Demo Query {i+1}:[/bold blue] {query}")
            console.print("="*80)
            
            result = await agent.process_patient_query(query)
            
            if result["status"] == "success":
                agent.display_timeline_table(result)
                console.print(f"\n[green]✅ SUCCESS: {len(result['timeline'])} events generated[/green]")
            else:
                agent.display_error_table(result)
            
    except Exception as e:
        console.print(f"[red]❌ Error: {str(e)}[/red]")
        import traceback
        traceback.print_exc()
        
    finally:
        await agent.close()


if __name__ == "__main__":
    import sys
    
    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        # Run demo with example queries
        asyncio.run(example_demo())
    else:
        # Run interactive CLI
        asyncio.run(interactive_main()) 