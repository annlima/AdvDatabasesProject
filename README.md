# Advanced Databases Project

## Installation

1. **Clone the repository:**
   ```bash
   git clone <(https://github.com/annlima/AdvDatabasesProject)>
   ```

2. **Install required Python packages:**
   ```bash
   pip install numpy scipy mysql-connector-python requests bs4 scikit-learn spacy
   python -m spacy download en_core_web_sm
   ```

3. **Set up the MySQL database:**
    - Execute the provided SQL script to create and configure the database.

## Usage

1. **Start the script:**
   ```bash
   python document_analysis_system.py
   ```

2. **Follow the on-screen menu to interact with the system:**
    - Add documents by text or URL.
    - Compare documents using various similarity metrics.
    - View and manage the document list and term-document matrix.

## Configurations

- **Database Configuration**: Modify `connect_db()` in `db_config.py` to change database connection settings.

- **Model Parameters**: Adjust parameters for vectorization and topic modeling within the script as needed.

