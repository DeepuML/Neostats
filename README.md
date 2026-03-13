
Project Title: AI Knowledge Chatbot with RAG and Live Web Search

Objective:
Build an intelligent chatbot using Streamlit that can answer user queries using both local knowledge sources and real-time web search. The chatbot should support multiple LLM providers and allow users to control response detail levels.

Core Features:

1. Retrieval-Augmented Generation (RAG)
- The chatbot must retrieve relevant information from local documents such as PDFs or text files.
- Documents are embedded using vector embeddings.
- Relevant document chunks are retrieved using similarity search before generating responses.
- Embedding models must be implemented inside models/embeddings.py.
- Retrieval logic should be implemented in the utils/ folder.

2. Live Web Search Integration
- If relevant information is not found in the local knowledge base, the chatbot should perform a live web search.
- The search results should be summarized and used as context for generating responses.
- Web search logic must be implemented inside utils/.
- API keys must be stored inside config/config.py.

3. Response Modes
The user interface should allow switching between two response styles:
- Concise Mode: Short summarized answers
- Detailed Mode: Longer explanations with structured insights

4. Modular Architecture
The project must follow this folder structure.


5. LLM Integration
The chatbot should support at least one LLM provider such as:
- OpenAI
- Groq
- Google Gemini

LLM loading logic should be implemented in models/llm.py.

6. Error Handling
All major functions should include try/except blocks to prevent application crashes and log errors.

7. Streamlit Interface
The UI should include:
- Chat input box
- Response display area
- Sidebar configuration panel
- Response mode toggle (Concise / Detailed)

Optional Enhancements:
- Document upload
- Chat history
- Source citations
- Multiple model selection

Deployment:
The application will be deployed on Streamlit Cloud and connected to a public GitHub repository.

Final Deliverables:
1. GitHub repository containing the source code
2. Deployed Streamlit application link
3. PowerPoint presentation explaining:
   - Use case
   - Architecture
   - Implementation approach
   - Challenges faced
   - Deployment link
