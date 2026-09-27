from __future__ import annotations
import json
from pathlib import Path
from .ui_constants import PROJECT_ROOT
SETTINGS_PATH = PROJECT_ROOT / "config" / "ui_settings.json"
DEFAULT_SETTINGS = {"language":"fa","theme":"dark","last_tab":0}
class UISettings:
    def __init__(self,path:Path=SETTINGS_PATH): self.path=Path(path); self.data=dict(DEFAULT_SETTINGS); self.load()
    def load(self):
        try:
            if self.path.exists():
                data=json.loads(self.path.read_text(encoding="utf-8"));
                if isinstance(data,dict): self.data.update({k:v for k,v in data.items() if k in DEFAULT_SETTINGS})
        except Exception: self.data=dict(DEFAULT_SETTINGS)
        return self.data
    def save(self,**updates):
        self.data.update(updates); self.path.parent.mkdir(parents=True,exist_ok=True); self.path.write_text(json.dumps(self.data,ensure_ascii=False,indent=2),encoding="utf-8")
