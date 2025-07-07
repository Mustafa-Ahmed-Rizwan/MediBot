# MediBot

🩺 **MediBot** is a medical question-answering chatbot built with Chainlit, LangChain, and Groq. It answers medical queries based on a knowledge base of PDF documents, using a FAISS vector store for efficient retrieval and a Groq-powered large language model (LLM) for generating concise, accurate responses. The UI is clean and user-friendly, with Grok-like source formatting for transparency.

## Features
- **Medical Q&A**: Answers medical questions using information from a preprocessed PDF knowledge base.
- **Chainlit UI**: A modern, chat-based interface for seamless user interaction.
- **Source Attribution**: Displays sources in a clear, numbered list with metadata (source, title, page) and content snippets, styled like Grok's source presentation.
- **Optimized Performance**: Fast startup with global embeddings caching, asynchronous FAISS loading, and lazy LLM initialization.
- **Error Handling**: Robust retry logic for handling API rate limits and parsing errors.

## Project Structure
```
Medical-Chatbot/
├── data/                   # Directory for input PDF documents
├── vectorstore/            # FAISS vector store (generated)
│   └── db_faiss/
├── medibot.py             # Main Chainlit application
├── ui.py                  # UI formatting functions
├── create_memory_for_llm.py # Script to process PDFs and create FAISS index
├── connect_memory_with_llm.py # CLI script for testing QA chain
├── .env                   # Environment file for GROQ_API_KEY
└── README.md              # Project documentation
```

## Prerequisites
- Python 3.8+
- A Groq API key (sign up at [x.ai](https://x.ai/api) to obtain one)
- PDF documents in the `data/` directory for the knowledge base

## Installation
1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd Medical-Chatbot
   ```

2. **Set Up a Virtual Environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install chainlit langchain-groq langchain-huggingface langchain-community python-dotenv faiss-cpu
   ```

4. **Set Up Environment Variables**:
   Create a `.env` file in the project root with your Groq API key:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

## Setup Knowledge Base
1. **Prepare PDFs**:
   Place medical PDF documents in the `data/` directory. These will form the knowledge base for MediBot.

2. **Create FAISS Vector Store**:
   Run the script to process PDFs and generate the FAISS index:
   ```bash
   python create_memory_for_llm.py
   ```
   This creates a `vectorstore/db_faiss` directory with the indexed data.

## Running the Application
1. **Start the Chainlit Server**:
   ```bash
   chainlit run medibot.py
   ```
2. **Access the UI**:
   Open your browser to `http://localhost:8000` to interact with MediBot.

3. **Usage**:
   - Type a medical question (e.g., "What are the symptoms of diabetes?").
   - MediBot will respond with an answer based on the PDF knowledge base.
   - Sources are displayed below the answer in a numbered list with metadata and content snippets.
   - Use the "Clear Chat" button to reset the conversation.

## Testing the QA Chain (Optional)
To test the question-answering chain via CLI:
```bash
python connect_memory_with_llm.py
```
Enter a query when prompted to see the raw response and source documents.

## Performance Optimizations
- **Global Embeddings**: The `sentence-transformers/all-MiniLM-L6-v2` model is loaded once globally to reduce startup time.
- **Asynchronous FAISS Loading**: The FAISS vector store is loaded non-blocking to improve startup performance.
- **Lazy LLM Initialization**: The Groq LLM is initialized only on the first query to minimize initial boot time.
- **Manual Caching**: The vector store is cached globally to prevent reloading across sessions.

## Troubleshooting
- **Slow Startup**:
  - Ensure the FAISS index (`vectorstore/db_faiss`) is not too large. Reduce `chunk_size` in `create_memory_for_llm.py` (e.g., from 500 to 300) if needed.
  - Verify the HuggingFace model cache (`~/.cache/huggingface`) is not cleared frequently.
  - Add timing logs to `medibot.py` to profile loading:
    ```python
    start = time.time()
    vectorstore = await get_vectorstore()
    print(f"Vector store loaded in {time.time() - start:.2f} seconds")
    ```
- **RuntimeError: cannot reuse already awaited coroutine**:
  - Ensure you're using the latest `medibot.py`, which uses manual caching instead of `@cl.cache`.
- **API Errors**:
  - Verify your `GROQ_API_KEY` is valid and the `llama-3.3-70b-versatile` model is available.
  - Check for rate limit errors and retry after a delay.
- **Missing Sources**:
  - Ensure PDFs in `data/` have proper metadata (source, title, page) for source attribution.

## Example Output
When you ask, "What are the symptoms of diabetes?":
```
**Answer:** Symptoms of diabetes include frequent urination, increased thirst, and fatigue.

**Sources:**

**1. Source: medical.pdf | Page: 5**
> Symptoms of diabetes include frequent urination, increased thirst...

**2. Source: health_guide.pdf | Title: Diabetes Overview | Page: 12**
> Common signs are fatigue, blurred vision, and slow-healing sores...
```

## Contributing
Contributions are welcome! Please:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Commit changes (`git commit -m 'Add your feature'`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.

## License
This project is licensed under the MIT License.

## Acknowledgments
- Powered by [LangChain](https://langchain.com), [Chainlit](https://chainlit.io), and [Groq](https://x.ai).
- Uses `sentence-transformers/all-MiniLM-L6-v2` for embeddings and `llama-3.3-70b-versatile` for LLM.
- Inspired by Grok's source formatting for clear, transparent responses.

© 2025 MediBot