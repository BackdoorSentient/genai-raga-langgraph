from fastapi import APIRouter
from app.workflows.raga_graph import build_raga_graph

router = APIRouter()
graph = build_raga_graph()


@router.post("/raga")
async def run_raga(query: str):
    result = graph.run({"raga_node": query})
    return {"response": result}
