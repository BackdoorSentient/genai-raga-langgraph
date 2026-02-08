from app.actions.base_action import BaseAction
class ExampleAction(BaseAction):
    def execute(self, input_data):
        return f"Action executed with input: {input_data}"
