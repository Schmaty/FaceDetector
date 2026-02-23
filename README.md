# Face Recognition Auto-Enrollment System

This project detects faces using a 2D camera, automatically recognizes previously seen individuals, and creates new profiles for unknown faces.

## Features

- Detect faces in real-time
- Compare faces against stored embeddings
- Automatically create new profiles for unknown individuals
- Store multiple images per person
- Maintain a persistent JSON index of known people

## Folder Structure

faces/
  ├── Person_1/
  ├── Person_2/
  └── ...

people.json

- `faces/` contains subfolders for each identified person.
- Each person folder stores captured face images.
- `people.json` stores metadata and profile indexing.

## Setup

Run:

python3 setup.py

This will:
- Create the faces directory
- Create an empty people.json file
- Prepare the system for recognition

## Adding Faces Automatically

The recognition script should:

1. Detect face
2. Match against stored embeddings
3. If unknown:
   - Create new Person_X folder
   - Add entry in people.json
4. Save cropped face image inside that person's folder

## Example people.json Format

{
    "Person_1": {
        "id": 1,
        "image_count": 5
    },
    "Person_2": {
        "id": 2,
        "image_count": 3
    }
}

## Notes

- This is not a biometric security system.
- Lighting consistency improves accuracy.
- More stored images per person improves stability.
