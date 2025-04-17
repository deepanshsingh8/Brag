from setuptools import setup, find_packages

setup(
    name="physics-question-generator",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'pdf2image==1.16.3',
        'PyPDF2==3.0.1',
        'pytesseract==0.3.10',
        'pillow==10.0.0',
        'langchain==0.1.0',
        'pinecone-client==3.0.0',
        'python-dotenv==1.0.0',
        'gradio==4.12.0',
        'transformers==4.35.2',
        'torch==2.1.0',
        'sentence-transformers==2.2.2',
        'roboflow==1.1.9',
        'numpy==1.24.3',
        'opencv-python==4.8.0.74',
        'python-magic==0.4.27'
    ],
) 