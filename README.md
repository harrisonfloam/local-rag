# local-rag

A Python project for simple, fully customizable local retrieval-augmented generation (RAG).

# Getting started

### Get the code

First, create a local clone of the repository:
```bash
git clone https://github.com/harrisonfloam/local-rag
```

To recreate the development environment, follow on in [DEV.md](DEV.md).

### Install external dependencies

The only hard requirements for normal execution is [Docker Desktop](https://docs.docker.com/desktop/), [Ollama](https://ollama.com/download), and an internet connection for image building.

### Download models

After installing ollama, pull an embedding model and an llm. We recommend [nomic-embed-text](https://ollama.com/library/nomic-embed-text) for an embedding model.

```bash
ollama pull gpt-oss:latest
```

### Run with Make

Start the application, build the Docker image, and run tests quickly using commands from the [Makefile](Makefile). Run the following from the project root to start the application:

```bash
make up
```

### Run with Docker Compose

If you don't want to install [Make](https://gnuwin32.sourceforge.net/packages/make.htm) or you're more familiar with the [Docker CLI](https://docs.docker.com/reference/cli/docker/), the same functionality and more are exposed via [docker-compose.yml](docker-compose.yml). To run the application:

```bash
docker compose up
```
