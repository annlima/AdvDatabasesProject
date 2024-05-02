import mysql.connector
import pandas as pd


def connect_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        passwd="root",
        database="DocumentDatabase"
    )


def insert_document(cursor, text, url=None):
    title = "Generated Title" if url is None else url.split("/")[-1]
    query = "INSERT INTO Documents (doc_url, doc_title, doc_text) VALUES (%s, %s, %s)"
    cursor.execute(query, (url if url else "No URL", title, text))

    # Get the ID of the document
    doc_id = cursor.lastrowid

    print("Document added successfully.")
    return doc_id


def eliminate_document_by_id(cursor, doc_id):
    cursor.execute("DELETE FROM Frequencies WHERE doc_id = %s", (doc_id,))
    cursor.execute("DELETE FROM Documents WHERE doc_id = %s", (doc_id,))
    delete_unreferenced_terms(cursor)
    print("Document and related data deleted successfully.")


def eliminate_document_by_title(cursor, title):
    cursor.execute("SELECT doc_id FROM Documents WHERE doc_title = %s", (title,))
    doc_id = cursor.fetchone()
    if doc_id:
        eliminate_document_by_id(cursor, doc_id[0])
    else:
        print("Document not found.")

def delete_unreferenced_terms(cursor):
    cursor.execute("DELETE FROM Terms WHERE term_id NOT IN (SELECT DISTINCT term_id FROM Frequencies)")
    print("Unreferenced terms deleted successfully.")

def insert_terms(cursor, term_data):
    insert_term = "INSERT INTO Terms (term_id, term_text) VALUES (%s, %s)"
    cursor.executemany(insert_term, term_data)


def insert_frequencies(cursor, frequency_data):
    insert_frequency = "INSERT INTO Frequencies (doc_id, term_id, frequency) VALUES (%s, %s, %s)"
    frequency_data = [(int(doc_id), int(term_id), int(freq)) for doc_id, term_id, freq in frequency_data]
    cursor.executemany(insert_frequency, frequency_data)


def document_exists(cursor, identifier, by_text=False):
    if by_text:
        query = "SELECT COUNT(*) FROM Documents WHERE doc_text = %s"
    else:
        query = "SELECT COUNT(*) FROM Documents WHERE doc_url = %s"
    cursor.execute(query, (identifier,))
    return cursor.fetchone()[0] > 0


def fetch_all_terms(cursor):
    cursor.execute("SELECT term_id, term_text FROM Terms")
    return cursor.fetchall()


def fetch_all_frequencies(cursor):
    cursor.execute("SELECT doc_id, term_id, frequency FROM Frequencies")
    return cursor.fetchall()


def fetch_all_documents(cursor):
    cursor.execute("SELECT doc_id, doc_url, doc_text FROM Documents")
    return cursor.fetchall()


def list_documents(cursor):
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
    query = "SELECT doc_id, doc_title FROM Documents"
    cursor.execute(query)
    documents = cursor.fetchall()
    if documents:
        df = pd.DataFrame(documents, columns=['Document ID', 'Title'])
        return df
    else:
        return pd.DataFrame()  # Return an empty DataFrame if no documents are found


def compare_list_documents(cursor):
    query = "SELECT doc_id, doc_title FROM Documents"
    cursor.execute(query)
    return cursor.fetchall()


def get_or_create_term_id(cursor, term):
    # Check if the term already exists
    cursor.execute("SELECT term_id FROM Terms WHERE term_text = %s", (term,))
    result = cursor.fetchone()
    if result:
        return result[0]

    # If it doesn't exist, create it
    cursor.execute("INSERT INTO Terms (term_text) VALUES (%s)", (term,))
    return cursor.lastrowid


def insert_or_update_frequency(cursor, doc_id, term_id, frequency):
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
    cursor.execute("SELECT COUNT(*) FROM Documents")
    return cursor.fetchone()[0] > 0


def close_db(db):
    db.close()
