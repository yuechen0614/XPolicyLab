"""Weight-free structural checks for the Wan backbone migration.

阶段0 基线版:只断言"可 import + 注册 + 类型契约",在当前(未重构)树上即应通过。
随重构推进逐步加严——`no-pipeline` / variant 元数据 / rope / mask 等断言留到阶段 3
(届时 `wan_backbone.py` 与 `wan/backbone` 工具就位后启用)。

bit-level 数值验收(vs 基线 golden)需真实权重,在 GPU 服务器上经 run_backbone.py +
compare.py 完成,不在此处。

    PYTHONPATH=. pytest tests/wan_migration/test_local.py -q
"""

from openwam.model.video_backbone import (
    _VIDEO_BACKBONE_REGISTRY,
    VideoBackbone,
    Wan21,
    Wan22Ti2v,
)

WAN_NAME_TO_CLS = {
    "wan22_ti2v_5b": Wan22Ti2v,
    "wan21_vace_1_3b": Wan21,
    "wan21_i2v_14b_480p": Wan21,
}


def test_three_wan_names_registered():
    for name, cls in WAN_NAME_TO_CLS.items():
        assert _VIDEO_BACKBONE_REGISTRY[name] is cls
    assert issubclass(Wan22Ti2v, VideoBackbone)
    assert issubclass(Wan21, VideoBackbone)


def test_wan_backbone_is_concrete():
    assert not getattr(Wan22Ti2v, "__abstractmethods__", frozenset())
    assert not getattr(Wan21, "__abstractmethods__", frozenset())
