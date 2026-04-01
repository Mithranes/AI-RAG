from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from llm import llm
from retriever import search, load_vectorstore
import json
import shutil
import os

app = FastAPI()

class MessageInput(BaseModel):
    text: str
    history: list = []

@app.get("/")
async def serve_index():
    return FileResponse("index.html")

# ← NEW: upload endpoint
@app.post('/upload')
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith(('.pdf', '.txt')):
        raise HTTPException(status_code=400, detail='Only PDF and TXT files allowed')
    try:
        filepath = f'uploaded_{file.filename}'
        with open(filepath, 'wb') as f:
            shutil.copyfileobj(file.file, f)
        load_vectorstore(filepath)
        return {'message': f'{file.filename} uploaded and loaded successfully'}
    except Exception as e:
        # ← proper FastAPI way for regular routes
        raise HTTPException(
            status_code=500,
            detail=f'Upload failed: {str(e)}'
        )

@app.post('/chat')
async def chat(message: MessageInput):
    if not message.text.strip():
        # main.py — don't process empty messages
        raise HTTPException(status_code=400, detail='Message cannot be empty')

    def generate():
        try:
            context = search(message.text)
            context_text = '\n'.join(context)

            system_prompt = {
                "role": "system",
                "content": f"""You are a document assistant. Your job is to help users understand the provided document.

            Answer only from the context below. Be concise and direct.
            If the answer is not in the context, say "I couldn't find that in the document."
            Never make up information.

            If its a logical question or a basic question you can answer it

            Context:
            {context_text}"""
            }

            messages = [system_prompt] + message.history + [{"role": "user", "content": message.text}]

            stream = llm.chat.completions.create(
                model=os.getenv('MODEL','llama-3.3-70b-versatile'),
                messages=messages,
                stream=True
            )

            for chunk in stream:
                token = getattr(chunk.choices[0].delta, "content", None)
                if token:
                    yield f"data: {json.dumps({'token': token})}\n\n"

        except Exception as e:
            # ← can't use HTTPException here, streaming already started
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type='text/event-stream',
        headers={'Cache-Control': 'no-cache'}
    )
