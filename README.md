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

The only hard requirements for normal execution is [Docker Desktop](https://docs.docker.com/desktop/) and an internet connection for image building. If you plan on using local LLMs for other tasks, you may want to install [Ollama](https://ollama.com/download).

### Run with Make

Start the application, build the Docker image, and run tests quickly using commands from the [Makefile](Makefile). Run the following from the project root to start the application:

```bash
make up
```

### Run with Docker Compose

If you're more familiar with the [Docker CLI](https://docs.docker.com/reference/cli/docker/), the same functionality and more are exposed via [docker-compose.yml](docker-compose.yml). To run the application:

```bash
docker compose up
```