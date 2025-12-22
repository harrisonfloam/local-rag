# Development

### Development environment

Proceed to create an isolated environment with [uv](#with-uv), VSCode [Dev Containers](#with-vscode-dev-containers-preferred), or [conda](#with-conda).

##### With uv (fast)

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) here, then run:

```bash
uv sync
```

##### With VSCode *Dev Containers* (isolated)

Install [Docker Desktop](https://marketplace.visualstudio.com/items/?itemName=ms-vscode-remote.remote-containers) and the VSCode extension [here](https://marketplace.visualstudio.com/items/?itemName=ms-vscode-remote.remote-containers). Once installed, VSCode will prompt to re-open the workspace in the Dev Container defined in [*.devcontainer*](/.devcontainer).

>**Hotkey**: <kbd>Ctrl</kbd>+<kbd>Shift</kbd>+<kbd>P</kbd> + <kbd>*Dev Containers: Rebuild and Reopen in Container*</kbd>

##### With conda (safe)

```bash
conda create -n local-rag python=3.12
conda activate local-rag
pip install -r requirements.txt
```

### Running the app

To attach your terminal to just the app container, not the ollama and chromadb containers:

```bash
docker compose up --attach app
```

To override settings in [settings.py](/app/settings.py), launch the app container with environment variables with the prefix `APP_` via [docker-compose.yml](docker-compose.yml)

```yml
services:
  app:
    environment:
      - APP_OLLAMA_HOST=ollama
      - APP_CHROMA_HOST=chromadb
      
      # Dev settings
      - APP_DEV_MODE=true
      - APP_DEBUG=true
      - APP_LOG_LEVEL=DEBUG
      - APP_MOCK_LLM=true
      - APP_USE_RAG=false
      - APP_RELOAD=true
      - APP_HTTPX_TIMEOUT=60
      ...
  ollama:
    ...
```