# Metso sales assistant

Answer product questions from brochures, estimates service costs from past visits, and compares plant readings with historical data.

## Run locally

```bash
uv sync --extra embeddings
cp .env.example .env
```

In `.env`, set `HF_TOKEN` to a Hugging Face token with Inference Providers access. `CHAT_MODEL` is the model and provider which should accept images alongside text.

Prepare the supplied data, then start the API:

```bash
uv run --no-sync python prepare.py --data /path/to/data
uv run --no-sync uvicorn app:app --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000/docs** and use `POST /chat`. Include the returned `session_id` in follow-up requests.

## Folder structure

| File | Purpose |
| --- | --- |
| `prepare.py` | Prepare documents, page images and search indexes. |
| `retrieval.py` | Combine keyword and semantic search. |
| `evidence.py` | Tools. |
| `calculations.py` | Calculate estimates and plant statistics. |
| `conversation.py` | The prompt, conversation history and tool loop. |
| `model.py` | Connect to Hugging Face Inference Providers. |
| `app.py` | Serve chat, sessions and source references. |
