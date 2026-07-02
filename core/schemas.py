"""
Pydantic 结构化输出模型 —— 系统的数据契约。

原则：
1. 所有模型继承 BaseModel，LangGraph 状态中自动序列化为 dict。
2. 字段使用严格类型标注，不允许多余字段（model_config 禁止 extra）。
3. 裁判输出（RefereeJudgment）是核心契约，不得被其他模块随意扩展。
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

# =============================================================================
# 基类
# =============================================================================


class _StrictModel(BaseModel):
    """项目基类：禁止额外字段，所有模型统一继承。"""

    model_config = ConfigDict(extra="forbid")


# =============================================================================
# 裁判输出（核心契约）
# =============================================================================


class RefereeJudgment(_StrictModel):
    """裁判对一轮论题演化的结构化判定。

    这是系统的核心契约 —— agents/referee.py 的 LLM 调用
    必须返回符合此 schema 的 JSON 输出。

    注意：round 由工作流状态和 RoundRecord 归档管理，
    LLM 不应输出该字段。
    """

    continue_debate: bool = Field(description="是否继续下一轮辩论。True = 论题仍需打磨，False = 论题已足够完善。")

    new_thesis: str = Field(
        min_length=1,
        description="裁判将当前论题、草稿和确认版拼合后的新 current_thesis。",
    )

    reasoning: str = Field(min_length=1, description="判定理由，说明为何继续或结束。")

    improvement_hint: str = Field(
        default="",
        description="下一轮的改进方向，或对最终论题的肯定评价。",
    )


# =============================================================================
# 轮次归档
# =============================================================================


class RoundRecord(_StrictModel):
    """单轮论题演化的完整归档（存入 history 列表）。"""

    round_number: int = Field(ge=1)
    thesis_before: str = Field(min_length=1, description="本轮开始时的 current_thesis")
    critique: str = Field(min_length=1, description="Opponent 的批判文本")
    user_response: str = Field(min_length=1, description="用户对批判的回应")
    draft_thesis: str = Field(min_length=1, description="Presenter 精确化后的草稿")
    confirmed_thesis: str = Field(min_length=1, description="用户确认后的论题")
    thesis_after: str = Field(min_length=1, description="Referee 拼合后的新 current_thesis")

    continue_debate: bool = Field(description="裁判是否决定继续下一轮")
    referee_reasoning: str = Field(min_length=1, description="裁判的判定理由")

    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
