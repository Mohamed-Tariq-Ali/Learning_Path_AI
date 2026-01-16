from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

# Create FastAPI app
app = FastAPI(title="Student Management API")

# In-memory database
students_db = []

# Student Model
class Student(BaseModel):
    id: int
    name: str
    age: int
    course: str


# Home Route
@app.get("/")
def home():
    return {"message": "Welcome to FastAPI Student App"}


# Add a Student (POST)
@app.post("/students", response_model=Student)
def add_student(student: Student):
    for s in students_db:
        if s.id == student.id:
            raise HTTPException(status_code=400, detail="Student ID already exists")
    students_db.append(student)
    return student


# Get All Students (GET)
@app.get("/students", response_model=List[Student])
def get_students():
    return students_db


# Get Student by ID (GET)
@app.get("/students/{student_id}", response_model=Student)
def get_student(student_id: int):
    for student in students_db:
        if student.id == student_id:
            return student
    raise HTTPException(status_code=404, detail="Student not found")


# Update Student (PUT)
@app.put("/students/{student_id}", response_model=Student)
def update_student(student_id: int, updated_student: Student):
    for index, student in enumerate(students_db):
        if student.id == student_id:
            students_db[index] = updated_student
            return updated_student
    raise HTTPException(status_code=404, detail="Student not found")


# Delete Student (DELETE)
@app.delete("/students/{student_id}")
def delete_student(student_id: int):
    for student in students_db:
        if student.id == student_id:
            students_db.remove(student)
            return {"message": "Student deleted successfully"}
    raise HTTPException(status_code=404, detail="Student not found")