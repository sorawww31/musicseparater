"""backend/src/model_catalog.py: 実行可能モデルと公開ステム契約を一元管理する。"""

from __future__ import annotations

from dataclasses import dataclass

from .config import SEPACAP, SINGER_INFORMED, UNMIXX


@dataclass(frozen=True)
class ModelDefinition:
    """API が受理するモデルの最小契約。"""

    model_id: str
    stems: tuple[str, ...]
    allowed_num_vocals: tuple[int | None, ...]
    multi_singer: bool = False
    requires_enrollment: bool = False


MODEL_CATALOG = {
    "bs-polarformer": ModelDefinition(
        "bs-polarformer", ("vocals", "instrumental"), (None, 2),
    ),
    "unmixx": ModelDefinition(
        "unmixx", (*UNMIXX.stems, "instrumental"), (None, 2), multi_singer=True,
    ),
    "sepacap": ModelDefinition(
        "sepacap", (*SEPACAP.stems, "instrumental"), (None,), multi_singer=True,
    ),
    "singer-informed": ModelDefinition(
        "singer-informed",
        SINGER_INFORMED.stems,
        (None,),
        multi_singer=True,
        requires_enrollment=True,
    ),
}


def validate_model_request(model_id: str, num_vocals: int | None) -> ModelDefinition:
    """未実装モデルとモデルに合わない人数指定を推論前に拒否する。"""
    definition = MODEL_CATALOG.get(model_id)
    if definition is None:
        raise ValueError(f"未対応のモデルです: {model_id}")
    if num_vocals not in definition.allowed_num_vocals:
        raise ValueError(f"{model_id} では num_vocals={num_vocals} を指定できません")
    return definition
