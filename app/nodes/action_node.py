from langgraph import Node
from app.actions.example_action import ExampleAction


class ActionNode(Node):
    def __init__(self, name: str):
        super().__init__(name)
        self.action = ExampleAction()

    def run(self, input_data: str):
        return self.action.execute(input_data)
