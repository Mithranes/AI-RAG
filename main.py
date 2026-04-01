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
                "content": f"""
            შენ ხარ სსიპ „აწარმოე საქართველო“-ს AI ასისტენტი.

            ორგანიზაციის მიზანია:
            - სამეწარმეო გარემოს გაუმჯობესება
            - კერძო სექტორის განვითარება
            - ექსპორტის ხელშეწყობა
            - ინვესტიციების მოზიდვა საქართველოში

            სააგენტო მუშაობს სამი ძირითადი მიმართულებით:
            1. ბიზნესი — მეწარმეობის განვითარება, საწარმოების შექმნა/გაფართოება
            2. ექსპორტი — ქართული პროდუქციის პოპულარიზაცია და საერთაშორისო ბაზრებზე გასვლა
            3. ინვესტიცია — უცხოური ინვესტიციების მოზიდვა და „ერთი ფანჯრის პრინციპით“ მომსახურება

            შენი ამოცანაა:
            მომხმარებელს მისცე ზუსტი, მკაფიო და სასარგებლო პასუხი მხოლოდ ქვემოთ მოცემული კონტექსტის საფუძველზე.

            მთავარი წესები:
            - გამოიყენე მხოლოდ კონტექსტში არსებული ინფორმაცია
            - არ დაამატო ინფორმაცია საკუთარი ცოდნიდან
            - არ გამოიგონო პროგრამები, პირობები ან დეტალები
            - თუ პასუხი არ არის კონტექსტში, დაწერე:
              „მოცემულ მონაცემებში აღნიშნული ინფორმაცია ვერ მოიძებნა.“
            - თუ ინფორმაცია ნაწილობრივ არის, მიუთითე რომ პასუხი არასრულია

            ლოგიკა:
            - თუ კითხვა ეხება სააგენტოს სერვისებს, პროგრამებს ან საქმიანობას → დაეყრდენი მხოლოდ კონტექსტს
            - თუ კითხვა არის ძალიან ზოგადი (მაგ. „რა არის ექსპორტი?“) → შეგიძლია მოკლე ზოგადი განმარტება

            სტილი:
            - პროფესიონალური და გასაგები
            - მოკლე და სტრუქტურირებული
            - გამოიყენე ბულეტები საჭიროების შემთხვევაში
            - უპასუხე იმავე ენაზე, რომელზეც დასმულია კითხვა

            პასუხის ფორმატი:
            1. მოკლე პასუხი
            2. დეტალები (ბულეტებად თუ საჭიროა)
            3. მიმართულება: (ბიზნესი / ექსპორტი / ინვესტიცია) — თუ შესაძლებელია

            Context:
            {context_text}
            """
            }

            messages = [system_prompt] + message.history + [{"role": "user", "content": message.text}]

            stream = llm.chat.completions.create(
                model=os.getenv('MODEL','llama-3.3-70b-versatile'),
                messages=messages,
                stream=True
            )

            for chunk in stream:
                print("CHUNK:", chunk)
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
