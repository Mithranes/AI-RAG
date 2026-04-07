from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from llm import llm
from retriever import search, load_vectorstore, delete_file, get_uploaded_files
import json
import shutil
import os
import uuid
from supabase_client import supabase

app = FastAPI()



class MessageInput(BaseModel):
    text: str
    history: list = []

class AuthInput(BaseModel):
    email: str
    password: str

security = HTTPBearer()

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        user = supabase.auth.get_user(credentials.credentials)
        return user.user
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

@app.get("/")
async def serve_index():
    return FileResponse("index.html")

@app.post('/register')
async def register(auth: AuthInput):
    try:
        response = supabase.auth.sign_up({"email": auth.email, "password": auth.password})
        return {"message": "Registration successful, please check your email"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post('/login')
async def login(auth: AuthInput):
    try:
        response = supabase.auth.sign_in_with_password({"email": auth.email, "password": auth.password})
        return {"access_token": response.session.access_token, "user_id": response.user.id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post('/upload')
async def upload_file(file: UploadFile = File(...), user=Depends(get_current_user)):
    if not file.filename.endswith(('.pdf', '.txt')):
        raise HTTPException(status_code=400, detail='Only PDF and TXT files allowed')
    try:
        filepath = f'uploaded_{uuid.uuid4()}_{file.filename}'
        with open(filepath, 'wb') as f:
            shutil.copyfileobj(file.file, f)
        load_vectorstore(filepath, user.id)
        return {'message': f'{file.filename} uploaded and loaded successfully'}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Upload failed: {str(e)}')

@app.get('/files')
async def list_files(user=Depends(get_current_user)):
    return {'files': get_uploaded_files(user.id)}

@app.delete('/files/{filename}')
async def remove_file(filename: str, user=Depends(get_current_user)):
    success = delete_file(filename, user.id)
    if not success:
        raise HTTPException(status_code=404, detail=f'{filename} not found')
    return {'message': f'{filename} removed successfully'}

# ── CHAT HISTORY ──
@app.get('/history')
async def get_history(user=Depends(get_current_user)):
    try:
        response = supabase.table('messages') \
            .select('role, content') \
            .eq('user_id', user.id) \
            .order('created_at') \
            .execute()
        return {'history': response.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete('/history')
async def clear_history(user=Depends(get_current_user)):
    try:
        supabase.table('messages').delete().eq('user_id', user.id).execute()
        return {'message': 'History cleared'}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/chat')
async def chat(message: MessageInput, user=Depends(get_current_user)):
    if not message.text.strip():
        raise HTTPException(status_code=400, detail='Message cannot be empty')

    def generate():
        try:
            results = search(message.text, user.id)
            context_text = '\n'.join([r["content"] for r in results])
            citations = [
                {"page": r["page"], "source": os.path.basename(r["source"])}
                for r in results
            ]

            system_prompt = {
                "role": "system",
                "content": f"""You are a helpful AI assistant that answers questions based on provided documents.

            ## Rules
            - Answer ONLY based on the context provided below
            - If the answer is not in the context, say: "I couldn't find that information in the uploaded document."
            - Never make up information
            - Be concise and clear
            - Use bullet points when listing multiple items
            - Answer in the same language the user writes in

            ## Context
            {context_text}
            """
            }

            messages = [system_prompt] + message.history + [{"role": "user", "content": message.text}]

            stream = llm.chat.completions.create(
                model=os.getenv('MODEL', 'llama-3.3-70b-versatile'),
                messages=messages,
                stream=True
            )

            yield f"data: {json.dumps({'citations': citations})}\n\n"

            full_response = ""
            for chunk in stream:
                token = getattr(chunk.choices[0].delta, "content", None)
                if token:
                    full_response += token
                    yield f"data: {json.dumps({'token': token})}\n\n"

            # save both messages to Supabase after streaming is done
            supabase.table('messages').insert([
                {"user_id": user.id, "role": "user", "content": message.text},
                {"user_id": user.id, "role": "assistant", "content": full_response}
            ]).execute()

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type='text/event-stream', headers={'Cache-Control': 'no-cache'})