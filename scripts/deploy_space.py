"""Deploy to a Hugging Face Docker Space.

Needs HF_WRITE_TOKEN (a Hugging Face token with write access) in backend/.env. FAL_KEY, APP_PASSWORD and HF_TOKEN
are set as Space secrets (never committed). Usage: python scripts/deploy_space.py [space-name]
"""
import secrets
import sys
from pathlib import Path

from dotenv import dotenv_values
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
env = dotenv_values(ROOT / "backend" / ".env")
token = env.get("HF_WRITE_TOKEN")
if not token:
    sys.exit("Add HF_WRITE_TOKEN=<a write token from https://huggingface.co/settings/tokens> to backend/.env")
api = HfApi(token=token)
user = api.whoami()["name"]
name = sys.argv[1] if len(sys.argv) > 1 else "jewellery-tryon"
space = f"{user}/{name}"

api.create_repo(space, repo_type="space", space_sdk="docker", private=False, exist_ok=True)

# secrets live in the Space settings, not in the code
if env.get("FAL_KEY"):
    api.add_space_secret(space, "FAL_KEY", env["FAL_KEY"])
if env.get("HF_TOKEN"):
    api.add_space_secret(space, "HF_TOKEN", env["HF_TOKEN"])
password = env.get("APP_PASSWORD") or secrets.token_urlsafe(9)
api.add_space_secret(space, "APP_PASSWORD", password)

api.upload_folder(
    repo_id=space, repo_type="space", folder_path=str(ROOT),
    ignore_patterns=[".venv/*", "data/*", "frontend/node_modules/*", "frontend/dist/*", "backend/.env", "**/__pycache__/*",
                     "test-results/*", "test-photos/*", "scratch/*", "output/*", ".playwright-cli/*", ".git/*", "README.md"],
    commit_message="Deploy jewellery try-on",
)
readme = ("---\ntitle: Jewellery Try-On\nemoji: 💍\ncolorFrom: yellow\ncolorTo: purple\nsdk: docker\napp_port: 7860\n"
          "pinned: false\n---\n\n" + (ROOT / "README.md").read_text())
api.upload_file(path_or_fileobj=readme.encode(), path_in_repo="README.md", repo_id=space, repo_type="space",
                commit_message="Space README")
host = f"{user}-{name}".lower().replace("_", "-")
print(f"Space: https://huggingface.co/spaces/{space}")
print(f"App:   https://{host}.hf.space")
if not env.get("APP_PASSWORD"):
    with open(ROOT / "backend" / ".env", "a") as f:  # keep it so redeploys don't change it
        f.write(f"\nAPP_PASSWORD={password}\n")
print("Password saved as APP_PASSWORD in backend/.env")
