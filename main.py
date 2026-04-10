from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from agent.agentic_wrokflow_api import GraphBuilder
import os
app=FastAPI()

class QueryRequest(BaseModel):
    question:str

@app.post("/query")
async def queryTravelAgent(query:QueryRequest):
    try:
        graph_build=GraphBuilder(model_provider="groq")
        react_app=graph_build()
        png_graph=react_app.get_graph().draw_mermaid_png()
        with open("myGraph.png","wb") as f:
            f.write(png_graph)
        print(f"Graph saved as 'myGraph.png' in {os.getcwd()}")

        messages={"messages": [query.question]}
        output= react_app.invoke(messages)

        if isinstance(output, dict) and "messages" in output:
            final_output=output["messages"][-1].content  #AI Last Response
        else:
            final_output=str(output)
        return {"answer": final_output}          
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
        