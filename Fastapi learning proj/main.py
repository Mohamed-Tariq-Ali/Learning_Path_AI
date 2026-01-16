# from os import name
#
 #from fastapi import FastAPI, Query, Path, Form, UploadFile, File
# import pandas as pd
# from PyPDF2 import PdfReader
# import io
#
#
# from typing import List
# from pydantic import BaseModel
# from typing import Optional
# from pydantic import Field

# class manf(BaseModel):
#     company:str
#     country:str
#
# class items(BaseModel):
#     name:str= Field(min_length=3,max_length=50,pattern="^[a-zA-Z]")
#     price:float=Field(gt=0,lt=10000)
#     available:Optional[bool]=None
#     manufacturer:manf
#
# app = FastAPI()
#
# '''Path parameter- used to give parameters in the path itself , If we give parameter in the
# path it will consider as required rather than optional'''
#
# emp=[
#     {'id':1,'name':'Tariq','place':'chennai'},
#     {'id':2,'name':'Sam','place':'Goa'},
# {'id':4,'name':'Joe','place':'Simp'},
# {'id':6,'name':'Alex','place':'USA'},
# ]
#
# @app.get("/display/{id}/{name}")
# def display(id:int,name:str):
#     for e in emp:
#         if e['id']==id and e['name']==name:
#             return e
#
#
# @app.get("/search/")
# def searchquery(id:int):
#     for e in emp:
#         if e['id'] == id:
#             return e
#
# #REQUEST BODY
#
# @app.post("/display/")
# def view(data:items):
#     return {"messgae":"item received","data":data}
#
#
# #Query parameter validation
#
# @app.get("/searchquery/")
# def searchque(id:int=Query(ge=0,le=10)):
#     for e in emp:
#         if e['id'] == id:
#             return e
#
#
# @app.get("/searchnameid/")
# def searchname(id:int=Query(ge=0,le=10),
#         name:str=Query(min_length=3,max_length=50,regex="^[a-zA-Z]")):
#     for e in emp:
#         if e['id']==id and e['name'].lower() == name.lower():
#             return e
#
#
# #PATH PARAMETER VALIDATION
#
#
# @app.get("/displayforpath/{id}")
# def displaypath(id:int=Path(ge=0,le=10,multiple_of=2)):
#     for e in emp:
#         if e['id']==id:
#             return e

#HANDLING FORMS AND FILE
#
# app = FastAPI()
#
# @app.post("/feedback/")
# def feeda(name:str=Form(...),email:str=Form(...),rating:int=Form(...)):
#      return {
#          "status":"feedback received",
#          "name":name,
#          "Rating":rating
#      }


# ===============FILE ULOAD BOTH SINGLE AND MULTI=======================

# app = FastAPI()
#
# @app.post("/file-upload/")
# async def file_upload(files: List[UploadFile] = File(...)):
#     #storing the data in the list
#     result = []
#     for file in files:
#         content = await file.read()
#
#         try:
#             text_preview = content.decode("utf-8")[:200]
#         except:
#             text_preview = "cannot be previewed as text"
#         result.append ( {
#              "filename": file.filename,
#             "content_type": file.content_type,
#             "Size in bytes": len(content),
#             "Text": text_preview
#             })
#     return result

#============= UPLOADING DIFFERENT FORMATS OF FILES

#
# app = FastAPI()
#
#
# @app.post("/different-file-upload/")
# async def file_upload(files: List[UploadFile] = File(...)):
#     results = []
#
#     for file in files:
#         content = await file.read()
#         name = file.filename.lower()
#
#         if name.endswith((".xls", ".xlsx")):
#             df = pd.read_excel(io.BytesIO(content))
#             results.append({
#                 "filename": file.filename,
#                 "Type": "EXCEL",
#                 "Data": df.head(3).to_dict()
#             })
#         elif name.endswith(".pdf"):
#             reader = PdfReader(io.BytesIO(content))
#             text = "".join([p.extract_text() or "" for p in reader.pages[:2]])
#             results.append({
#                 "filename": file.filename,
#                 "Type": "PDF",
#                 "Data": text.strip()[:500]
#             })
#         else:
#             results.append({
#                 "filename": file.filename,
#                 "Type": "ERROR",
#                 "Data": "Unsupported file type"
#             })
#
#     return {"total_files": len(results), "results": results}

#======================SESSION HANDLING=============================

# from fastapi import FastAPI, HTTPException, Cookie, Response
# from typing import Optional
# import uuid
# from datetime import datetime, timedelta
#
# app = FastAPI()
#
# couname = "admin"
# coupass = "1234"
#
# sessions = {}
# SESSION_TIMEOUT = 300  # 5 minutes in seconds
#
#
# @app.get("/login")
# def login(uname: str, pas: str, res: Response):
#     if couname == uname and coupass == pas:
#         sid = str(uuid.uuid4())
#         sessions[sid] = {
#             "username": uname,
#             "expires_at": datetime.now() + timedelta(seconds=SESSION_TIMEOUT)
#         }
#         res.set_cookie(key="sid", value=sid, httponly=True, max_age=SESSION_TIMEOUT)
#         return {"Message": "Login Successful"}
#     else:
#         raise HTTPException(status_code=401, detail="Incorrect username or password")
#
#
# @app.get("/home")
# def home(sid: Optional[str] = Cookie(None)):
#     if sid is None or sid not in sessions:
#         raise HTTPException(status_code=401, detail="NOT AUTHENTICATED")
#
#     # Check if session expired
#     if datetime.now() > sessions[sid]["expires_at"]:
#         del sessions[sid]
#         raise HTTPException(status_code=401, detail="SESSION EXPIRED")
#
#     return {"message": "Welcome", "user": sessions[sid]["username"]}
#
#
# @app.get("/logout")
# def logout(res: Response, sid: Optional[str] = Cookie(None)):
#     if sid and sid in sessions:
#         del sessions[sid]
#     res.delete_cookie(key="sid")
#     return {"Message": "Logout Successful"}


#===================== JWT AUTHENTICATION==================
from fastapi import FastAPI,HTTPException
from datetime import datetime,timedelta
from jose import JWTError,jwt

#configuration #header.payload.signature
ALGORITHM="HS256"
ACCESS_TOKEN_EXPIRE_MINUTES=10
SECRET_KEY="mysecretkey_123"


app=FastAPI()

def create_token(uname:str):
    expire=datetime.utcnow()+timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload={"user name":uname,"expire":expire.timestamp()}
    return jwt.encode(payload,SECRET_KEY,ALGORITHM)


def verify_token(token:str):
    try:
        pauload=jwt.decode(token,SECRET_KEY,algorithms=[ALGORITHM])
        return pauload["user name"]
    except JWTError:
        raise HTTPException(status_code=401,detail="Could not validate credentials")
@app.post("/login")
def login(uname:str , password:str):
    if uname=="admin" and password=="1234":
        token=create_token(uname)
        return {"access_token":token}
    return HTTPException(status_code=401,detail="Incorrect username or password")

@app.get("/secure_data")
def secure_data(token:str):
    uname=verify_token(token)
    return {"Message":f"Hello {uname},this is secure endpoint"}



