import mysql.connector
import pandas as pd


def connect_db():
    """Connects to the DocumentDatabase.

    :return: A connection object to the DocumentDatabase.
    """
    return mysql.connector.connect(
        host="localhost",
        user="root",
        passwd="root",
        database="DocumentDatabase"
    )


def insert_document(cursor, text, url=None):
    """
    Insert a document into the database with the given cursor, text, and optional URL.

    :param cursor: The cursor object for executing database queries.
    :param text: The text content of the document.
    :param url: Optional URL of the document.
    :return: The ID of the inserted document.
    """
    title = "Generated Title" if url is None else url.split("/")[-1]
    query = "INSERT INTO Documents (doc_url, doc_title, doc_text) VALUES (%s, %s, %s)"

    # Convert list to string if text is a list
    if isinstance(text, list):
        text = ' '.join(text)

    cursor.execute(query, (url if url else "No URL", title, text))

    # Get the ID of the document
    doc_id = cursor.lastrowid

    print("Document added successfully.")
    return doc_id


def eliminate_document_by_id(cursor, doc_id):
    """
    Delete a document and its related data from the database using the provided document ID.

    :param cursor: The database cursor object.
    :param doc_id: The ID of the document to be deleted.
    :return: None
    """
    cursor.execute("DELETE FROM Frequencies WHERE doc_id = %s", (doc_id,))
    cursor.execute("DELETE FROM Documents WHERE doc_id = %s", (doc_id,))
    delete_unreferenced_terms(cursor)
    print("Document and related data deleted successfully.")


def eliminate_document_by_title(cursor, title):
    """
    :param cursor: The cursor object for database operations
    :param title: The title of the document to be eliminated

    :return: None

    This method eliminates a document from the database based on its title.
    If a document with the given title exists,
    it retrieves the corresponding `doc_id` from the `Documents` table,
    and then calls another method `eliminate_document_by_id`
    to eliminate the document using the `doc_id`.
    If the document doesn't exist, it prints "Document not found."
    """
    cursor.execute("SELECT doc_id FROM Documents WHERE doc_title = %s", (title,))
    doc_id = cursor.fetchone()
    if doc_id:
        eliminate_document_by_id(cursor, doc_id[0])
    else:
        print("Document not found.")


def delete_unreferenced_terms(cursor):
    """
    :param cursor: The database cursor object.
    :return: None

    Deletes terms from the database table 'Terms' that are not referenced in the 'Frequencies' table.
    Note: This method does not return any value.
    """
    cursor.execute("DELETE FROM Terms WHERE term_id NOT IN (SELECT DISTINCT term_id FROM Frequencies)")
    print("Unreferenced terms deleted successfully.")


def insert_terms(cursor, terms):
    """
    Insert unique terms into the Terms table.

    :param cursor: The database cursor
    :param terms: A list of unique terms to insert
    :return: None
    """
    # Generate term data as a list of tuples [(term_text,), (term_text,), ...]
    term_data = [(term,) for term in terms if term]
    if term_data:
        for term in term_data:
            # Check if the term already exists
            cursor.execute("SELECT term_id FROM Terms WHERE term_text = %s", term)
            result = cursor.fetchone()
            if result:
                # If it exists, return the existing term_id
                return result[0]
            else:
                # If it doesn't exist, insert the new term
                cursor.execute("INSERT INTO Terms (term_text) VALUES (%s)", term)
                return cursor.lastrowid



def insert_frequencies(cursor, frequency_data):
    """
    Insert frequency data into the Frequencies table.

    :param cursor: Database cursor object.
    :param frequency_data: List of tuples containing doc_id, term_id, and frequency.
    :return: None.
    """
    insert_frequency = "INSERT INTO Frequencies (doc_id, term_id, frequency) VALUES (%s, %s, %s)"
    frequency_data = [(int(doc_id), int(term_id), int(freq)) for doc_id, term_id, freq in frequency_data]
    cursor.executemany(insert_frequency, frequency_data)


def document_exists(cursor, identifier, by_text=False):
    """
    Check if a document exists in the database.

    :param cursor: A database cursor object
    :param identifier: the identifier of the document, either the document text or URL
    :param by_text: a flag indicating if the identifier is a document text (default: False)
    :return: True if the document exists, False otherwise
    """
    if by_text:
        query = "SELECT COUNT(*) FROM Documents WHERE doc_text = %s"
    else:
        query = "SELECT COUNT(*) FROM Documents WHERE doc_url = %s"
    cursor.execute(query, (identifier,))
    return cursor.fetchone()[0] > 0


def fetch_all_terms(cursor):
    """
    Fetches all terms from the database.

    :param cursor: The database cursor object used to execute the query.
    :return: A list of tuples containing the term_id and term_text of all terms in the database.

    """
    cursor.execute("SELECT term_id, term_text FROM Terms")
    return cursor.fetchall()


def fetch_all_frequencies(cursor):
    """
    Fetches all frequencies from the database.

    :param cursor: The database cursor to execute the query.
    :return: A list of tuples representing the fetched frequencies.
    Each tuple contains the following elements:
             - doc_id: The document ID.
             - term_id: The term ID.
             - frequency: The frequency of the term in the document.
    """
    cursor.execute("SELECT doc_id, term_id, frequency FROM Frequencies")
    return cursor.fetchall()


def fetch_all_documents(cursor):
    """
    Fetches all the documents from the database.

    :param cursor: The database cursor.
    :return: A list of tuples with the documents' data.
    """
    cursor.execute("SELECT doc_id, doc_url, doc_text FROM Documents")
    return cursor.fetchall()


def list_documents(cursor):
    """
    Retrieve the list of documents from the database.

    :param cursor: The database cursor to execute the query.
    :type cursor: Cursor object

    :return: None
    """
    query = "SELECT doc_id, doc_title FROM Documents"
    cursor.execute(query)
    documents = cursor.fetchall()  # Fetch the documents
    if documents:
        print("List of Documents:")
        for doc_id, title in documents:
            print(f"ID: {doc_id}, Title: {title}")
    else:
        print("No documents found.")


def list_documents_for_interface(cursor):
    """
    Retrieve a list of documents and their titles from the database.

    :param cursor: The database cursor.
    :return: A pandas DataFrame containing the document ID and title for each document,
             or an empty DataFrame if no documents are found.
    """
    query = "SELECT doc_id, doc_title FROM Documents"
    cursor.execute(query)
    documents = cursor.fetchall()
    if documents:
        df = pd.DataFrame(documents, columns=['Document ID', 'Title'])
        return df
    else:
        return pd.DataFrame()  # Return an empty DataFrame if no documents are found


def compare_list_documents(cursor):
    """
    Retrieve the list of documents from the database and return the results.

    :param cursor: The database cursor object
    :return: A list of tuples containing the document ID and title
    """
    query = "SELECT doc_id, doc_title FROM Documents"
    cursor.execute(query)
    return cursor.fetchall()


def get_or_create_term_id(cursor, term):
    """
    Get the term ID for the given term from the database.
    If the term
    does not exist, create it and return the new ID.

    :param cursor: The database cursor object.
    :param term: The term to look for or create.
    :return: The term ID (int) if it exists, otherwise the new term ID (int).
    """
    # Check if the term already exists
    cursor.execute("SELECT term_id FROM Terms WHERE term_text = %s", (term,))
    result = cursor.fetchone()
    if result:
        return result[0]

    # If it doesn't exist, create it
    cursor.execute("INSERT INTO Terms (term_text) VALUES (%s)", (term,))
    return cursor.lastrowid


def insert_or_update_frequency(cursor, doc_id, term_id, frequency):
    """

    :param cursor: The cursor object for executing SQL queries.
    :param doc_id: The ID of the document.
    :param term_id: The ID of the term.
    :param frequency: The frequency value to insert or update.

    :return: None

    """
    # Check if the frequency already exists
    cursor.execute("SELECT frequency FROM Frequencies WHERE doc_id = %s AND term_id = %s", (doc_id, term_id))
    result = cursor.fetchone()

    if result:
        # If it exists, update the frequency
        new_frequency = result[0] + frequency
        cursor.execute("UPDATE Frequencies SET frequency = %s WHERE doc_id = %s AND term_id = %s",
                       (new_frequency, doc_id, term_id))
    else:
        # If it doesn't exist, insert the new frequency
        cursor.execute("INSERT INTO Frequencies (doc_id, term_id, frequency) VALUES (%s, %s, %s)",
                       (doc_id, term_id, frequency))


def check_if_documents_exist(cursor):
    """
    :param cursor: The database cursor object used to execute the SQL query.
    :return: True if there are documents in the database, False otherwise.
    """
    cursor.execute("SELECT COUNT(*) FROM Documents")
    return cursor.fetchone()[0] > 0


def close_db(db):
    """
    Closes the given database connection.

    :param db: The database connection to be closed.
    :return: None
    """
    db.close()
