# AnkiGen Development Notes

## Dependency Management

This project uses `pyproject.toml` for direct dependencies (floor pins like `>=`) and `requirements.txt` for the full frozen lockfile (exact pins with `==`). Both must stay in sync.

### Gradio + HuggingFace Spaces

The app is deployed as a Gradio Space on HuggingFace. The README.md YAML frontmatter contains `sdk_version:` which **must** match the pinned gradio version in `requirements.txt`. If these drift, the Space will either fail to build or run a different gradio version than what was tested locally.

When bumping gradio:
1. Update `requirements.txt` (`gradio==X.Y.Z`)
2. Update `pyproject.toml` (`gradio>=X.Y.Z`)
3. Update `README.md` frontmatter (`sdk_version: X.Y.Z`)
4. Also bump `gradio-client` in `requirements.txt` to the matching release
