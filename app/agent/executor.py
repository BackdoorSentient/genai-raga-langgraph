# def executor_node(state):
#     step = state["plan"][state["current_step"]]

#     tool = step.tool
#     action = step.action

#     if tool == "rag":
#         result = rag_tool(action)

#     elif tool == "web_search":
#         result = web_search_tool(action)

#     else:  # llm
#         result = llm.invoke(action).content

#     return {
#         "observations": state.get("observations", []) + [result],
#         "current_step": state["current_step"] + 1
#     }

