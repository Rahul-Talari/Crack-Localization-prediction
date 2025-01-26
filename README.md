
# Disease Prediction and Recommendation System using Knowledge Graphs & RAG 🤖🧠

This repository contains a **Disease Prediction and Recommendation System** built using **Neo4j Knowledge Graph** and **Retrieval Augmented Generation (RAG)** techniques with **Groq** and **DuckDuckGo** search. The system is designed to predict and recommend diseases based on user-provided symptoms, while also suggesting diets, natural remedies, and medical consultations.

## Key Features 🌟
- **Disease Prediction**: Predict diseases based on symptoms using a Neo4j Knowledge Graph.
- **Real-time Web Search**: Augment predictions with real-time web search results from DuckDuckGo.
- **RAG Workflow**: Combine graph-based knowledge with external web search results to provide comprehensive disease recommendations.
- **Diet and Natural Remedies**: Suggest a healthy diet and natural techniques to manage or alleviate symptoms.
- **Doctor Recommendations**: Suggest the type of doctor to consult based on symptoms.

## Technologies Used ⚙️
- **Neo4j**: Graph database for storing disease-symptom relationships.
- **Groq LLM**: Large language model for generating Cypher queries and refining recommendations.
- **DuckDuckGo Search**: Search tool for retrieving disease-related information from the web.
- **LangChain**: Framework for integrating Neo4j, Groq, and DuckDuckGo with an LLM.

## Requirements 🔧

1. **Neo4j Database**: Set up a Neo4j database with disease and symptom data.
2. **Python Libraries**:
    ```bash
    pip install pandas langchain_neo4j neo4j langchain_core langchain_ollama langchain_groq duckduckgo-search langchain_tools
    ```
3. **Groq API Key**: Get an API key from Groq and set it in your environment:
    ```bash
    export GROQ_API_KEY="your-groq-api-key"
    ```

## Setup and Usage 🚀

1. **Neo4j Database Setup**:
    - Download and install Neo4j from [neo4j.com](https://neo4j.com/download/).
    - Set up a new database and import your disease-symptom data.
    - Ensure you have the Neo4j URI, username, and password ready.

2. **Clone the Repository**:
    ```bash
    git clone https://github.com/your-username/disease-prediction-system.git
    cd disease-prediction-system
    ```

3. **Connecting to Neo4j**:
    - In the `Neo4jGraph` initialization code, replace `NEO4J_URI`, `NEO4J_USERNAME`, and `NEO4J_PASSWORD` with your database details.

4. **Running the Knowledge Graph Query**:
    - After loading your data into Neo4j, use the provided code to query the graph for diseases related to symptoms. You can modify the symptoms list based on user input.

5. **Running the Agentic RAG System**:
    - Integrate the DuckDuckGo search tool and Groq LLM to refine the disease predictions and merge them with web-based results. The system can recommend diseases, diets, natural techniques, and doctors to consult based on symptoms.

6. **Example**:
    ```python
    symptoms = "itching, skin rash, nodal skin eruptions"
    Agent_response = agent.invoke(f"Recommend diseases based on the following symptoms: {symptoms}")
    print(Agent_response['output'])
    ```

## Folder Structure 📂
```
disease-prediction-system/
│
├── data/
│   ├── preprocessed_data.csv      # Sample data for diseases and symptoms
│
├── scripts/
│   ├── kg_construction.py         # Knowledge graph construction
│   ├── rag_system.py              # Agentic RAG system
│   ├── query_execution.py         # Querying the Neo4j graph
│
├── README.md                     # Project documentation
```

## License 📜
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements 🙏
- **Neo4j**: A powerful graph database used for storing and querying disease-symptom relationships.
- **Groq**: LLM used to generate Cypher queries and refine predictions.
- **DuckDuckGo**: Search tool used to augment the disease recommendation process.

---

Feel free to contribute to the project, report issues, or suggest improvements. For any questions, open an issue or contact the project maintainers.
