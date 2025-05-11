# Graph-Concept Adapter

This module acts as a translation and integration layer between the Crawler's ML output and the downstream services (Graph API, Concept Dictionary, and optionally Neo4j).

## Features
- Accepts raw concept data from the Crawler or other ML services
- Translates and validates data for Graph API and Concept Dictionary
- Handles async HTTP requests and error logging
- Easily extensible to support Neo4j or other backends

## Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Use the adapter in your code:
   ```python
   from adapter import GraphConceptAdapter
   import asyncio

   adapter = GraphConceptAdapter()
   raw_concept = { ... }  # Your ML output
   asyncio.run(adapter.process_and_send(raw_concept))
   ```

3. Configure service URLs via environment variables if needed:
   - `GRAPH_API_URL` (default: http://graph-api:8200)
   - `CONCEPT_DICT_URL` (default: http://concept-dictionary:8000)
   - `NEO4J_URL` (default: bolt://neo4j:7687)

## Extending
- To add direct Neo4j integration, install the `neo4j` Python package and add logic in `process_and_send`.

## License
MIT 