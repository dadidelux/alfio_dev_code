Here's the README.md content in Markdown format:

markdown
Copy code
# alfio_dev_code

This repository contains the code for a FastAPI application that uses sentence transformers and FAISS for searching similar titles based on a query.

## Getting Started

### Prerequisites

- Python 3.6 or higher
- [Pip](https://pypi.org/project/pip/)

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/dadidelux/alfio_dev_code.git
   cd alfio_dev_code
Create and activate a virtual environment (optional):

bash
Copy code
python -m venv venv
source venv/bin/activate  # On Unix/macOS
venv\Scripts\activate  # On Windows
Install the required packages:

bash
Copy code
pip install -r requirements.txt
Configuration
Make sure the alfio_dev_path variable in your code is set to the correct path. For running locally, you can use:

python
Copy code
alfio_dev_path = "../alfio_dev_code/"
Running the Application
To run the program on your local machine:

bash
Copy code
uvicorn recommender:app --reload
Accessing the Application
Once the application is running, you can access it by visiting the following URL in your web browser:

perl
Copy code
http://127.0.0.1:8000/search?query_title=Fast%20Shipping&top_k=5
Replace Fast Shipping with your query title and adjust top_k to control the number of results returned.