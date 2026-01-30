class BaseAction:
    def execute(self, input_data):
        raise NotImplementedError("Action must implement execute()")
