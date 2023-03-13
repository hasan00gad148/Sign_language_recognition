from fastapi import FastAPI, Response, status, HTTPException, Depends, APIRouter,Depends, File, UploadFile
from fastapi.params import Body
from .orm_models import post as Postdb, base
from . import baseModels 
from .database import get_db ,engine,session
import os
import aiofiles
from fastapi.middleware.cors import CORSMiddleware

base.metadata.create_all(bind=engine)



        
        
        
app = FastAPI()


origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.get("/")
async def root():
    return {"massege": "welcome"}



@app.get("/sqlalchemy")
async def test(db:session = Depends(get_db)):
    q = db.query(Postdb).all()
    print(q)
    return {"massege": "welcome to sqlalchemy get"}



@app.post("/sqlalchemy/file")
async def test(f1: UploadFile, f2: UploadFile, db:session = Depends(get_db)):
    
    async with aiofiles.open("./files/"+f1.filename, 'wb') as out_file:
        content = await f1.read()  # async read
        await out_file.write(content)  
        
    async with aiofiles.open("./files/"+f2.filename, 'wb') as out_file:
        content = await f2.read()  # async read
        await out_file.write(content)   
           
    return {"filename": [f1.filename,f2.filename],"file type": [f1.content_type,f2.content_type], "status":"success"}



@app.get("/posts",response_model = baseModels.PostResGet)
async def posts(db:session = Depends(get_db)):
    posts = db.query(Postdb).all()
    return {"posts": posts,"status":"success"}



@app.post("/posts",response_model = baseModels.PostResGet)
async def createPosts(post: baseModels.Post = Body(...), db:session = Depends(get_db)):
    post = Postdb(**post.dict())
    db.add(post)
    db.commit()
    db.refresh(post)
    print(post.id)
    return {"posts": [post],"status":"success"}



@app.get("/posts/{id}")
async def getPost(id: int, db:session = Depends(get_db)):
    p = db.query(Postdb).filter(Postdb.id == id)
    p = p.first()
    return {"posts": [p],"status":"success"} if p else HTTPException(status.HTTP_404_NOT_FOUND, "not found ")



@app.put("/posts/{id}")
async def updatePosts(id: int, post: baseModels.Post = Body(...), db:session = Depends(get_db)):
    p = db.query(Postdb).filter(Postdb.id == id)
    p_ = p.first()
    if p_ :
        p.update(post.dict(), synchronize_session=False)
        # db.update(Postdb).where(Postdb.id == id).values(**post.dict())
        db.commit()
        return {"posts": p_.id,"status":"success"}
    return HTTPException(status.HTTP_404_NOT_FOUND, "not found ")



@app.delete("/posts/{id}")
async def deletePosts(id: int, db:session = Depends(get_db)):
    p = db.query(Postdb).filter(Postdb.id == id)
    p_ = p.first()
    if p_ :
        db.delete(p_)#.where(Postdb.id == id)
        db.commit()
        return {"posts": p_.id,"status":"success"}
    return HTTPException(status.HTTP_404_NOT_FOUND, "not found ")