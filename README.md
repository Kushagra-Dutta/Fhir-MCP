# FHIR Careplan - Universal FHIR Server and Tools

A comprehensive FHIR (Fast Healthcare Interoperability Resources) server and toolkit designed for healthcare data integration, patient care planning, and clinical decision support. This project provides a universal interface to multiple FHIR servers with advanced AI-powered clinical analysis capabilities.

## 🏥 Overview

The FHIR Careplan project consists of two main components:

1. **Universal FHIR MCP Server** (`fhir_server.py`) - A Model Context Protocol (MCP) server that provides standardized access to multiple FHIR servers
2. **FHIR Tools Library** (`fhir_tools.py`) - A comprehensive Python library for FHIR data manipulation, analysis, and AI-powered clinical insights

## ✨ Key Features

### 🔗 Multi-Server FHIR Integration
- **Universal FHIR Interface**: Connect to multiple FHIR servers (Epic, Cerner, HAPI, Firely, etc.)
- **Vendor-Agnostic**: Standardized API regardless of underlying FHIR server implementation
- **Real-time Connectivity Testing**: Automatic server health checks and diagnostics
- **Intelligent Failover**: Automatic switching between servers for optimal performance

### 🤖 AI-Powered Clinical Analysis
- **OpenAI Integration**: Extract clinical keywords and concepts from free text
- **Semantic Mapping**: Map clinical terms to standardized FHIR codes
- **Similar Patient Matching**: Find patients with similar clinical profiles
- **Predictive Analytics**: Generate care recommendations based on historical data

### 📊 Comprehensive Patient Data Access
- **Complete Patient Records**: Demographics, conditions, medications, procedures, encounters
- **Vital Signs & Lab Results**: Categorized observations with time-series data
- **Care Plans & Teams**: Treatment plans and healthcare provider information
- **Allergies & Procedures**: Complete medical history tracking

### 🚀 Performance Optimization
- **Async Operations**: Non-blocking I/O for high-performance data access
- **Intelligent Caching**: Condition codes and frequently accessed data caching
- **Batch Processing**: Efficient handling of multiple patient records
- **Connection Pooling**: Optimized HTTP connections for multiple servers

## 🛠 Installation

### Prerequisites
- Python 3.11 or higher
- OpenAI API key (for AI features)
- Access to FHIR servers (local or remote)

### Setting Up MCP Server

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Kushagra-Dutta/Fhir-MCP.git
   cd FHIR-MCP
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   # or using uv
   uv sync
   ```

3. **Configure Environment Variables**
   ```bash
   # Create .env file
   echo "OPENAI_API_KEY=your_openai_api_key" > .env
   ```

4. **Run the FHIR MCP Server**
   ```bash
   python fhir_server.py
   ```

### Setting Up Firely Test Database Locally

1. **Obtain a Firely Server License Key**
   - Visit [Firely Server Trial Page](https://fire.ly/firely-server-trial/)
   - Fill out the form to receive a license key via email
   - You'll receive the license key and download files (license valid for 7 days)
   - Save the license file as `firelyserver-license.json`

2. **Set Up Using Docker**
   ```bash
   # Pull the Firely server image
   docker pull firely/server

   # Run the container
   # For Windows CMD:
   docker run -d -p 9090:4080 --name firely.server -v %CD%/firelyserver-license.json:/app/firelyserver-license.json firely/server

   # For PowerShell or macOS/Linux:
   docker run -d -p 9090:4080 --name firely.server -v ${PWD}/firelyserver-license.json:/app/firelyserver-license.json firely/server

   # Verify container is running
   docker ps
   ```

3. **Load Test Data**
   - Use Postman to load test data bundles
   - Create a new PUT request in Postman
   - Set request type to raw JSON
   - Copy content from test data bundles
   - Send requests to base URL (http://localhost:9090)
   - Repeat for each data bundle

After completing these steps, you'll have a working test database accessible at http://localhost:9090.

### Setting Up MCP Chatbot Client

You can set up the MCP chatbot client either by forking the repository or using Docker. The chatbot will serve as the frontend interface for interacting with the FHIR server.

1. **Get the Chatbot setup**
   ```bash
   # Clone the repository
   git clone https://github.com/cgoinglove/mcp-client-chatbot.git
   cd mcp-client-chatbot
   ```

2. **Install PNPM (if not installed)**
   ```bash
   npm install -g pnpm
   ```

3. **Choose Setup Method:**

   #### Docker Compose Setup 🐳
   ```bash
   # Install dependencies
   pnpm i

   # Configure environment variables
   # Edit .env file with your API keys (created automatically after pnpm i)

   # Start all services including PostgreSQL
   pnpm docker-compose:up
   ```

   #### Local Setup 🚀
   ```bash
   # Install dependencies
   pnpm i

   # Setup environment variables
   pnpm initial:env

   # Start PostgreSQL (skip if already running)
   pnpm docker:pg

   # Run database migrations
   pnpm db:migrate

   # Start development server
   pnpm dev

   # Optional: Build & start for production-like testing
   pnpm build:local && pnpm start
   ```

4. **Configure Environment Variables**
   Create/edit `.env` file with required API keys:
   ```env
   # LLM Provider API Keys (add the ones you plan to use)
   OPENAI_API_KEY=****
   GOOGLE_GENERATIVE_AI_API_KEY=****
   ANTHROPIC_API_KEY=****
   XAI_API_KEY=****
   OPENROUTER_API_KEY=****
   OLLAMA_BASE_URL=http://localhost:11434/api

   # Auth Configuration
   BETTER_AUTH_SECRET=**** # Generate with: npx @better-auth/cli@latest secret
   BETTER_AUTH_URL= # Optional: URL you access the app from

   # Database Configuration
   POSTGRES_URL=postgres://your_username:your_password@localhost:5432/your_database_name

   # MCP Configuration
   FILE_BASED_MCP_CONFIG=false

   # Optional OAuth Settings (for Google/GitHub login)
   GOOGLE_CLIENT_ID=
   GOOGLE_CLIENT_SECRET=
   GITHUB_CLIENT_ID=
   GITHUB_CLIENT_SECRET=
   ```

5. **Connect MCP Server to Chatbot**
   - Access the chatbot at http://localhost:3000
   - Create an account and login
   - Go to MCP configuration
   - Click "Add Server"
   - Copy-paste your `.chatbot-config.json` configuration

6. **Configure System Prompt**
   - Navigate to `src/app/prompts.ts` in the chatbot repository
   - Replace the default system prompt with your custom prompt from `system_prompt.txt`
   - Example system prompt structure:
     ```typescript
     export const defaultSystemPrompt = `You are a powerful agentic AI coding assistant.
     You have access to a set of tools you can use to answer the user's question.
     
     <communication>
     1. Be conversational but professional.
     2. Refer to the USER in the second person and yourself in the first person.
     3. Format your responses in markdown.
     4. NEVER lie or make things up.
     </communication>

     <tool_calling>
     Follow these rules regarding tool calls:
     1. ALWAYS follow the tool call schema exactly as specified.
     2. NEVER call tools that are not explicitly provided.
     3. Only calls tools when they are necessary.
     4. Before calling each tool, first explain to the USER why you are calling it.
     </tool_calling>

     <making_code_changes>
     When making code changes:
     1. Add all necessary import statements and dependencies
     2. NEVER output code to the USER, unless requested
     3. Use appropriate code edit tools for implementation
     </making_code_changes>
     `;
     ```
   - Customize the prompt sections based on your needs:
     - `<communication>`: Define how the AI should interact
     - `<tool_calling>`: Specify rules for using tools
     - `<making_code_changes>`: Set guidelines for code modifications
   - Save the changes and restart the development server

After completing these steps, your chatbot will be connected to the MCP server and ready to interact with the FHIR database.

## 🏗 Architecture

### Core Components

```mermaid
graph TD
    A[FHIR Server] --> B[Universal FHIR MCP Server]
    B --> C[FHIR Tools Library]
    C --> D[Multiple FHIR Servers]
    C --> E[OpenAI Integration]
    C --> F[Caching Layer]
    
    D --> G[Firely Local]
    D --> H[HAPI R4]
    D --> I[Epic Systems]
    D --> J[Cerner]
    
    E --> K[Clinical Text Analysis]
    E --> L[Code Mapping]
    E --> M[Patient Matching]
```

### Server Registry
The system maintains a comprehensive registry of FHIR servers:

- **Firely Local** (`http://localhost:9090`) - Local development server
- **HAPI R4** (`http://hapi.fhir.org/baseR4`) - Public test server
- **Epic, Cerner, Azure** - Enterprise healthcare systems (configurable)

## 📖 Usage Guide

### Basic FHIR Operations

#### 1. Server Management
```python
from fhir_tools import UniversalFhirMcpServer

# Initialize the server
fhir_server = UniversalFhirMcpServer()
await fhir_server.initialize()

# Test server connectivity
result = await fhir_server.test_server_connectivity("firely_local")
print(result)

# Switch to different server
await fhir_server.switch_server("hapi_r4")
```

#### 2. Patient Search
```python
# Search for patients
search_criteria = {
    "family": "Smith",
    "given": "John",
    "birthdate": "1990-01-01"
}
patients = await fhir_server.find_patient(search_criteria)
```

#### 3. Patient Data Retrieval
```python
# Get comprehensive patient information
patient_info = await fhir_server.get_comprehensive_patient_info("patient-123")

# Get specific data types
conditions = await fhir_server.get_patient_conditions("patient-123")
medications = await fhir_server.get_patient_medications("patient-123")
vital_signs = await fhir_server.get_vital_signs("patient-123")
lab_results = await fhir_server.get_lab_results("patient-123")
```

### AI-Powered Clinical Analysis

#### 1. Clinical Text Extraction
```python
# Extract clinical keywords from free text
clinical_note = "45-year-old female with HER2+ invasive ductal carcinoma, stage IIIA"
keywords = await fhir_server.extract_clinical_keywords(clinical_note)
```

#### 2. FHIR Code Mapping
```python
# Map clinical terms to FHIR codes
clinical_data = {
    "conditions": ["breast cancer", "diabetes"],
    "age": 45,
    "gender": "female"
}
fhir_codes = await fhir_server.map_to_fhir_codes_fast(clinical_data)
```

#### 3. Similar Patient Matching
```python
# Find patients with similar clinical profiles
criteria = {
    "age": 45,
    "gender": "female",
    "conditions": ["breast cancer"]
}
similar_patients = await fhir_server.find_similar_patients_simple(criteria)
```

### MCP Server Integration

The FHIR server can be used as an MCP server for integration with AI assistants:

```json
{
  "mcpServers": {
    "fhir-server": {
      "command": "python",
      "args": ["fhir_server.py"],
      "env": {
        "OPENAI_API_KEY": "your-key-here"
      }
    }
  }
}
```

## 🔧 Configuration

### Server Configuration
The system supports multiple FHIR servers configured in `fhir_tools.py`:

```python
servers = {
    "firely_local": {
        "name": "Firely Server Local",
        "base_url": "http://localhost:9090",
        "version": "R4",
        "vendor": "Firely",
        "auth_type": "none"
    },
    "hapi_r4": {
        "name": "HAPI FHIR R4 Public",
        "base_url": "http://hapi.fhir.org/baseR4",
        "version": "R4",
        "vendor": "HAPI",
        "auth_type": "none"
    }
}
```

### Environment Variables
```bash
# Required
OPENAI_API_KEY=your_openai_api_key

# Optional
FHIR_SERVER_URL=http://localhost:9090
FHIR_SERVER_AUTH_TOKEN=your_auth_token
```

## 🔍 Available Tools

### Core FHIR Operations
- `switch_server(server_name)` - Switch between FHIR servers
- `test_server_connectivity(server_name)` - Test server connectivity
- `find_patient(search_criteria)` - Search for patients
- `get_comprehensive_patient_info(patient_id)` - Get complete patient data

### Clinical Data Access
- `get_patient_observations(patient_id)` - Get patient observations
- `get_patient_conditions(patient_id)` - Get patient conditions
- `get_patient_medications(patient_id)` - Get patient medications
- `get_vital_signs(patient_id)` - Get vital signs
- `get_lab_results(patient_id)` - Get laboratory results
- `get_patient_encounters(patient_id)` - Get patient encounters
- `get_patient_allergies(patient_id)` - Get patient allergies
- `get_patient_procedures(patient_id)` - Get patient procedures

### AI-Powered Analysis
- `extract_clinical_keywords(text)` - Extract clinical information from text
- `map_to_fhir_codes_fast(clinical_data)` - Map terms to FHIR codes
- `find_similar_patients_simple(criteria)` - Find similar patients
- `extract_condition_codes_from_fhir()` - Extract all condition codes

### System Management
- `list_available_servers()` - List all configured servers
- `get_server_registry()` - Get complete server registry
- `diagnose_fhir_server(server_name)` - Diagnose server capabilities
- `clear_condition_cache()` - Clear condition codes cache
- `get_condition_cache_stats()` - Get cache performance statistics

## 🧪 Testing Tools

### MCP Inspector
The MCP (Model Context Protocol) Inspector is a powerful development and testing tool that helps debug and validate FHIR server interactions. It provides real-time inspection of server behavior and API responses.

#### Using MCP Inspector
```bash
# Run MCP Inspector on a specific file
mcp dev fhir_server.py

# This will:
# 1. Start the inspector in development mode
# 2. Monitor all FHIR server interactions
# 3. Log detailed information to logs/mcp_dev_inspector.log
```

#### Features
- **Real-time Monitoring**: Watch FHIR server interactions as they happen
- **Request/Response Logging**: Detailed logs of all API calls and responses
- **Error Detection**: Immediate feedback on API errors or misconfigurations
- **Performance Metrics**: Track response times and server performance
- **Debug Mode**: Enhanced logging for development troubleshooting

#### Log File Location
The inspector writes detailed logs to:
```
logs/mcp_dev_inspector.log
```

#### Best Practices
- Use MCP Inspector during development to validate server behavior
- Monitor the log file for unexpected errors or performance issues
- Run the inspector when implementing new FHIR endpoints
- Use it to debug connection issues with external FHIR servers

## 🏥 Business Applications

### Hospital Engagement Platform
This toolkit can power automated hospital engagement platforms:

1. **Patient Care Coordination**
   - Automated care plan generation
   - Treatment timeline management
   - Multi-disciplinary team coordination

2. **Clinical Decision Support**
   - Evidence-based treatment recommendations
   - Risk assessment and early warnings
   - Outcome prediction based on similar cases

3. **Resource Optimization**
   - Predictive analytics for resource allocation
   - Automated scheduling and capacity management
   - Cost optimization through data-driven insights

4. **Quality Improvement**
   - Automated quality metrics tracking
   - Compliance monitoring and reporting
   - Performance analytics and benchmarking

## 🔐 Security & Compliance

- **HIPAA Compliance**: Designed with healthcare data privacy in mind
- **Secure Communication**: HTTPS/TLS encryption for all server communications
- **Authentication Support**: Multiple authentication methods (OAuth, API keys, etc.)
- **Audit Logging**: Comprehensive logging for compliance and debugging

## 📊 Performance Features

- **Async Operations**: Non-blocking I/O for high throughput
- **Connection Pooling**: Efficient HTTP connection management
- **Intelligent Caching**: Condition codes and metadata caching
- **Batch Processing**: Efficient handling of multiple records
- **Error Handling**: Robust error handling and retry mechanisms

## 🧪 Testing

### Running Tests
```bash
# Run basic connectivity tests
python -c "
import asyncio
from fhir_tools import UniversalFhirMcpServer

async def test():
    server = UniversalFhirMcpServer()
    await server.initialize()
    result = await server.test_server_connectivity('firely_local')
    print(result)

asyncio.run(test())
"
```

### Server Diagnostics
```python
# Comprehensive server diagnostics
diagnostics = await fhir_server.diagnose_fhir_server("firely_local")
print(diagnostics)
```

## 📝 Logging

The system provides comprehensive logging:

```python
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Logs are automatically written to logs/mcp_dev_inspector.log
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Create an issue in the GitHub repository
- Check the logs directory for detailed error information
- Use the diagnostic tools for server troubleshooting

## 🔄 Version History

- **v0.1.0** - Initial release with core FHIR functionality
- **v0.2.0** - Added AI-powered clinical analysis
- **v0.3.0** - Enhanced multi-server support and caching

## 🎯 Roadmap

- [ ] Advanced analytics dashboard
- [ ] Real-time data streaming
- [ ] Machine learning model integration
- [ ] Enhanced security features
- [ ] Mobile application support
- [ ] Cloud deployment templates

## 🕒 FHIR Timeline Agent

The FHIR Timeline Agent (`fhir_timeline_agent.py`) is a specialized agent designed to generate detailed clinical treatment timelines from patient queries using real FHIR data. It's specifically configured to work with the Firely Local FHIR server.

### Features

- **Natural Language Processing**: Converts free-text patient queries into structured clinical data
- **Real Patient Data Analysis**: Uses actual FHIR patient records for timeline generation
- **AI-Powered Timeline Generation**: Leverages OpenAI GPT-4 for accurate clinical timelines
- **Interactive CLI Interface**: User-friendly command-line interface with rich formatting
- **Comprehensive Patient Matching**: Finds similar patients based on age, gender, and conditions

### Usage

```bash
# Run in interactive mode with chat interface
python fhir_timeline_agent.py

# Run in demo mode with example queries
python fhir_timeline_agent.py --demo
```

### Example Queries
```
• "45-year-old male with pancreatic adenocarcinoma"
• "62-year-old female with HER2+ breast cancer"
• "58-year-old male with stage IIIA lung adenocarcinoma"
```

### Timeline Generation Process

1. **Query Processing**
   - Extracts clinical keywords from natural language
   - Identifies age, gender, conditions, stage, and biomarkers

2. **FHIR Code Mapping**
   - Maps clinical terms to standardized FHIR codes
   - Uses Firely Local server's code systems

3. **Similar Patient Search**
   - Finds matching patients in the FHIR database
   - Scores matches based on age, gender, and conditions

4. **Data Aggregation**
   - Collects comprehensive medical history
   - Includes procedures, medications, encounters

5. **Timeline Generation**
   - Creates detailed treatment timeline using AI
   - Organizes events chronologically with clinical context

### Output Format

The agent generates rich, formatted output including:

- **Patient Profile**: Demographics, diagnosis, stage, biomarkers
- **Treatment Timeline**: Step-by-step clinical events with dates
- **Clinical Outcomes**: Treatment response, survival status, toxicity
- **Data Sources**: Server information and analysis metrics

### Configuration

The agent is hardcoded to use:
- Firely Local FHIR server (`http://localhost:9090`)
- OpenAI GPT-4 for timeline generation
- Rich console output for formatted display

### Requirements

- OpenAI API key (set in `.env` file)
- Running Firely Local FHIR server
- Python packages:
  ```
  openai>=1.86.0
  rich>=13.7.0
  python-dotenv>=1.0.0
  aiohttp>=3.8.0
  ```

### Best Practices

- Provide complete patient information in queries
- Include age, gender, and primary diagnosis
- Add stage and biomarker information when available
- Use specific clinical terms for better matching

### Error Handling

The agent includes robust error handling for:
- Missing patient information
- FHIR server connectivity issues
- AI generation failures
- Data parsing errors

Each error is displayed with helpful suggestions for resolution.

---

*Built with ❤️ for healthcare interoperability and patient care improvement* 
=======

