from fastapi import FastAPI
import modules
from pydantic import BaseModel


app = FastAPI()

class Answer(BaseModel):
    question: str


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/query/{q_id}")
async def questionAnswer(q_id: int, ans: Answer):
    answer=modules.get_response(ans.question)
    # answer = obj.start_dialog(ans.question)
    return {"item_id": q_id,"Question": ans.question,"Answer":answer}

# @app.post('/addknowledge/{q_id}')
# async def addKnowledge(q_id:int, knowledge:str):
#     obj=AnswerGeneration.AnswerGeneration()
#     val=obj.addKnowledge(knowledge)
#     if val:
#         return {"status": "Success! Knowledge Successfully added."}
#     else:
#         return {"status": "Error!"}

# @app.post('/addinstruction/{q_id}')
# async def addInstruction(q_id:int, instruction:int):
#     obj=AnswerGeneration.AnswerGeneration()
#     val=obj.modifyInstruction(instruction)
#     if val:
#         return {"status": "Success! Instruction Update Successful."}
#     else:
#         return {"status": "Error!"}

# @app.get("/query/{q_id}")
# async def questionAnswer(q_id: int,ans=Answer):
#     return {"ID": q_id,"Question": ans.question,"Answer":ans.answer}
    

# answer = obj.start_dialog("Tell me about Teddy AI.")
# print(answer)