from langchain_core.messages import ToolsUnitTests, ParrotMultiplyTool

class TestParrotMultiplyToolUnit(ToolsUnitTests):
    @property
    def tool_constructor(self):
        return ParrotMultiplyTool

    def tool_invoke_params_example(self):
        return {"a": 2, "b": 3}
    