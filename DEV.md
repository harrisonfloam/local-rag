# Development

### Development environment

Proceed to create an isolated environment with either VSCode [Dev Containers](#with-vscode-dev-containers-preferred) or [conda](#with-conda).

##### With VSCode *Dev Containers* (preferred)

Activate the development environment by installing [Docker Desktop](https://marketplace.visualstudio.com/items/?itemName=ms-vscode-remote.remote-containers) and the VSCode extension [here](https://marketplace.visualstudio.com/items/?itemName=ms-vscode-remote.remote-containers). Once installed, VSCode will prompt to re-open the workspace in the Dev Container defined in [*.devcontainer*](/.devcontainer).

>**Hotkey**: <kbd>Ctrl</kbd>+<kbd>Shift</kbd>+<kbd>P</kbd> + <kbd>*Dev Containers: Rebuild and Reopen in Container*</kbd>

##### With conda

```bash
conda create -n nlpl python=3.12
conda activate nlpl
pip install -r requirements.txt
```