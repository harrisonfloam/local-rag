# local-rag

A Python project for simple, no strings attached local RAG.

# Getting started

### Get the code

First, create a local clone of the repository:
```bash
git clone https://github.com/harrisonfloam/natural-language-policy-learning
```

To recreate the development environment, follow on in [DEV.md](DEV.md).

### Install external dependencies

TODO: Docker, Make

### Run with Make

Start the application, build the Docker image, and run tests quickly using commands from the [Makefile](Makefile). Run the following from the project root to start the application:

```bash
make up
```

#### Run with Docker Compose

If you're more familiar with the [Docker CLI](https://docs.docker.com/reference/cli/docker/), the same functionality and more exposed via [docker-compose.yml](docker-compose.yml). To run the application:

```bash
docker compose up
```