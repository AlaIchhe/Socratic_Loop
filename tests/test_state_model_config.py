"""core/state.py —— model_config 字段与 make_initial_state 测试。"""

from core.state import make_initial_state, validate_state_shape


class TestModelConfigField:
    def test_make_initial_state_includes_model_config_none(self):
        state = make_initial_state("test thesis")
        assert "model_config" in state
        assert state["model_config"] is None

    def test_validate_state_shape_accepts_model_config(self):
        state = make_initial_state("test thesis")
        # 不应抛错
        result = validate_state_shape(state)
        assert result is state

    def test_validate_state_shape_accepts_model_config_dict(self):
        state = make_initial_state("test thesis")
        state["model_config"] = {
            "provider": "openai",
            "base_url": "https://api.openai.com/v1",
            "api_key": "sk-test",
            "model": "gpt-4o",
            "temperature": 0.7,
            "max_tokens": 4096,
            "json_mode": False,
        }
        result = validate_state_shape(state)
        assert result["model_config"] == state["model_config"]

    def test_make_initial_state_other_fields_unaffected(self):
        state = make_initial_state("thesis", agent_temperature=0.5, max_rounds=5)
        assert state["current_thesis"] == "thesis"
        assert state["round"] == 1
        assert state["agent_temperature"] == 0.5
        assert state["max_rounds"] == 5
        assert state["status"] == "idle"
        assert state["messages"] == []
        assert state["history"] == []
        assert state["final_result"] == ""
