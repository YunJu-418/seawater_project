import mimetypes
import os
import urllib.parse
import uuid
from pathlib import Path

import requests


BUCKET = "exit-images"


def upload_public_image(image_path: str) -> str | None:
    supabase_url = os.getenv("SUPABASE_URL", "").strip().rstrip("/")
    secret_key = os.getenv("SUPABASE_SECRET_KEY", "").strip()

    if not supabase_url or not secret_key:
        print("[SUPABASE ERROR] URL 또는 SECRET_KEY가 없습니다.")
        return None

    path = Path(image_path)
    if not path.exists():
        print(f"[SUPABASE ERROR] 이미지가 없습니다: {path}")
        return None

    remote_name = f"exits/{path.stem}_{uuid.uuid4().hex[:8]}{path.suffix.lower()}"
    encoded_name = urllib.parse.quote(remote_name, safe="/")

    upload_url = f"{supabase_url}/storage/v1/object/{BUCKET}/{encoded_name}"
    content_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"

    response = requests.post(
        upload_url,
        headers={
            "apikey": secret_key,
            "Content-Type": content_type,
        },
        data=path.read_bytes(),
        timeout=30,
    )

    if response.status_code not in (200, 201):
        print(f"[SUPABASE UPLOAD ERROR] {response.status_code}: {response.text}")
        return None

    public_url = (
        f"{supabase_url}/storage/v1/object/public/"
        f"{BUCKET}/{encoded_name}"
    )

    print("[SUPABASE IMAGE UPLOADED]")
    return public_url