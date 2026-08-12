from __future__ import annotations
from exit_system.supabase_storage import upload_public_image

import json
from pathlib import Path
from typing import Any

import requests


class KakaoNotifier:
    def __init__(self, cfg: dict) -> None:
        self.enabled = bool(cfg["enabled"])
        self.endpoint = str(cfg["endpoint"])
        self.link_url = str(cfg["link_url"])
        self.upload_endpoint = str(
            cfg.get(
                "upload_endpoint",
                "https://kapi.kakao.com/v2/storage/image/upload",
            )
        )
        self.token_endpoint = str(
            cfg.get(
                "token_endpoint",
                "https://kauth.kakao.com/oauth/token",
            )
        )
        self.token_file = Path(
            str(cfg.get("token_file", "kakao_token.json"))
        )

        self.token_data = self._load_token_data()
        self.access_token = str(
            self.token_data.get("access_token")
            or cfg.get("access_token", "")
        ).strip()

    def _load_token_data(self) -> dict[str, Any]:
        if not self.token_file.exists():
            return {}

        try:
            data = json.loads(
                self.token_file.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[KAKAO TOKEN LOAD ERROR] {exc}")
            return {}

        return data if isinstance(data, dict) else {}

    def _save_token_data(self) -> bool:
        try:
            temp_file = self.token_file.with_suffix(
                self.token_file.suffix + ".tmp"
            )
            temp_file.write_text(
                json.dumps(
                    self.token_data,
                    ensure_ascii=False,
                    indent=4,
                ),
                encoding="utf-8",
            )
            temp_file.replace(self.token_file)
            return True
        except OSError as exc:
            print(f"[KAKAO TOKEN SAVE ERROR] {exc}")
            return False

    def _refresh_access_token(self) -> bool:
        refresh_token = str(
            self.token_data.get("refresh_token", "")
        ).strip()
        rest_api_key = str(
            self.token_data.get("rest_api_key", "")
        ).strip()
        client_secret = str(
            self.token_data.get("client_secret", "")
        ).strip()

        if not refresh_token or not rest_api_key:
            print(
                "[KAKAO TOKEN REFRESH ERROR] "
                "refresh_token 또는 rest_api_key가 비어 있습니다."
            )
            return False

        request_data = {
            "grant_type": "refresh_token",
            "client_id": rest_api_key,
            "refresh_token": refresh_token,
        }

        if client_secret:
            request_data["client_secret"] = client_secret

        try:
            response = requests.post(
                self.token_endpoint,
                headers={
                    "Content-Type":
                    "application/x-www-form-urlencoded;charset=utf-8"
                },
                data=request_data,
                timeout=10,
            )
        except requests.RequestException as exc:
            print(f"[KAKAO TOKEN REFRESH NETWORK ERROR] {exc}")
            return False

        if response.status_code != 200:
            print(
                f"[KAKAO TOKEN REFRESH ERROR] "
                f"{response.status_code}: {response.text}"
            )
            return False

        try:
            payload = response.json()
        except ValueError:
            print("[KAKAO TOKEN REFRESH ERROR] 응답이 JSON이 아닙니다.")
            return False

        new_access_token = str(
            payload.get("access_token", "")
        ).strip()

        if not new_access_token:
            print("[KAKAO TOKEN REFRESH ERROR] 새 access_token이 없습니다.")
            return False

        self.access_token = new_access_token
        self.token_data["access_token"] = new_access_token

        if payload.get("refresh_token"):
            self.token_data["refresh_token"] = payload["refresh_token"]

        for key in (
            "expires_in",
            "refresh_token_expires_in",
            "scope",
            "token_type",
        ):
            if key in payload:
                self.token_data[key] = payload[key]

        if not self._save_token_data():
            return False

        print("[KAKAO TOKEN REFRESHED]")
        return True

    def upload_image(self, image_path: str) -> str | None:
        return upload_public_image(image_path)

    def _post_template(self, template: dict) -> bool:
        if not self.access_token:
            print("[KAKAO ERROR] access_token이 비어 있습니다.")
            return False

        def request_send():
            return requests.post(
                self.endpoint,
                headers={
                    "Authorization":
                    f"Bearer {self.access_token}",
                    "Content-Type":
                    "application/x-www-form-urlencoded;charset=utf-8",
                },
                data={
                    "template_object": json.dumps(
                        template,
                        ensure_ascii=False,
                    )
                },
                timeout=10,
            )

        try:
            response = request_send()
        except requests.RequestException as exc:
            print(f"[KAKAO NETWORK ERROR] {exc}")
            return False

        if response.status_code == 401:
            if not self._refresh_access_token():
                return False

            try:
                response = request_send()
            except requests.RequestException as exc:
                print(f"[KAKAO RETRY ERROR] {exc}")
                return False

        if response.status_code != 200:
            print(
                f"[KAKAO ERROR] "
                f"{response.status_code}: {response.text}"
            )
            return False

        try:
            return response.json().get("result_code") == 0
        except ValueError:
            print("[KAKAO ERROR] 응답이 JSON이 아닙니다.")
            return False

    def send_to_me(
        self,
        text: str,
        image_path: str | None = None,
    ) -> bool:
        if not self.enabled:
            print("[KAKAO DISABLED]")
            return True

        image_url = (
            self.upload_image(image_path)
            if image_path
            else None
        )

        if image_url:
            image_template = {
                "object_type": "feed",
                "content": {
                    "title": "외출 감지 사진",
                    "description": "외출 당시 촬영된 사진입니다.",
                    "image_url": image_url,
                    "link": {
                        "web_url": self.link_url,
                        "mobile_web_url": self.link_url,
                    },
                },
                "button_title": "확인",
            }

            if not self._post_template(image_template):
                return False

        text_template = {
            "object_type": "text",
            "text": text,
            "link": {
                "web_url": self.link_url,
                "mobile_web_url": self.link_url,
            },
            "button_title": "확인",
        }

        return self._post_template(text_template)
