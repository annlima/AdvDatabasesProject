# Document Management System

This Document Management System (DMS) is designed to handle and analyze text documents within a MySQL database. It provides functionalities for adding, deleting, and listing documents, alongside advanced text processing and analysis capabilities. This system integrates document management with text similarity analysis using natural language processing and machine learning techniques.

## Features

- **Database Integration**: Connect to a MySQL database to manage document data.
- **Document Handling**: Add, delete, and list documents using either textual input or URLs.
- **Text Analysis**: Process text to extract meaningful data and calculate term frequencies.
- **Similarity Measurement**: Compare documents using various metrics like cosine similarity, Jaccard index, and more.
- **Interactive Web Interface**: Manage documents through a user-friendly interface powered by Streamlit.

## Prerequisites

Before you start, ensure you have the following installed:
- Python 3.8 or higher
- MySQL Server
- Python libraries: `mysql-connector-python`, `pandas`, `numpy`, `scikit-learn`, `spacy`, `streamlit`, `beautifulsoup4`, `requests`
- After install spacy you have to install: `python -m spacy download en_core_web_sm`

## Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/annlima/AdvDatabasesProject.git
   cd document-management-system
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Initialize the database**:
   - Run the MySQL server and create a new database named `DocumentDatabase`.
   - Execute the SQL script provided in `database_setup.sql` to create the necessary tables.

4. **Configure your database connection**:
   - Modify the `connect_db` function in the `database.py` file to match your database credentials.

5. **Run the Streamlit application**:
   ```bash
   streamlit run app.py
   ```

## Usage

- **Adding Documents**: Add documents by inputting text directly or providing URLs for text extraction.
- **Deleting Documents**: Documents can be deleted by ID or title.
- **Viewing Documents**: List all documents along with their details stored in the database.
- **Document Comparison**: Compare two documents based on different similarity metrics.
- **Term Document Matrix and Similarity Matrix**: Visualize the relationship between different documents and terms.

## Contributing

Contributions to improve the Document Management System are welcome. Before contributing, please ensure to follow the coding conventions and pull request process.
