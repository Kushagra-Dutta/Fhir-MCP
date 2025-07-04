from setuptools import setup, find_packages

setup(
    name="fhir-careplan",
    version="0.1.0",
    description="FHIR Care Plan Tools and MCP Server",
    author="Your Name",
    packages=find_packages(exclude=["logs", "templates", "logs.*", "templates.*"]),
    install_requires=[
        "fastmcp",
        "python-dotenv",
        "aiohttp",
        "mcp-core[all]",
        "mcp-cli[all]"
    ],
    entry_points={
        "console_scripts": [
            "fhir-mcp=mcp_inspector:main",
        ],
    },
    python_requires=">=3.7",
) 