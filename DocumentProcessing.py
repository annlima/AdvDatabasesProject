import collections

import matplotlib.pyplot as plt
import requests
import spacy
import streamlit as st
from bs4 import BeautifulSoup
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from Formulas import *
from Queries import *

# Load the English NLP model
nlp = spacy.load("en_core_web_sm")
vectorizer = TfidfVectorizer()


def fetch_and_extract_text(url):
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, 'html.parser')
    return ' '.join(p.text for p in soup.find_all('p'))


def preprocess_text(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if token.is_alpha and not token.is_stop])


def build_corpus_from_db(cursor):
    documents = fetch_all_documents(cursor)
    texts = []
    for _, url, text in documents:
        if url and not text:
            processed_text = preprocess_text(fetch_and_extract_text(url))
        else:
            processed_text = preprocess_text(text)
        texts.append(processed_text)
    return texts


def create_lsi_model(texts, num_topics=10):
    vectorizer = TfidfVectorizer()
    x = vectorizer.fit_transform(texts)

    # Apply Truncated SVD (LSI)
    svd_model = TruncatedSVD(n_components=num_topics)
    svd_model.fit(x)

    terms = vectorizer.get_feature_names_out()
    for i, comp in enumerate(svd_model.components_):
        terms_comp = zip(terms, comp)
        sorted_terms = sorted(terms_comp, key=lambda x: x[1], reverse=True)[:10]
        print("Topic " + str(i) + ": ")
        print(sorted_terms)


# URL of the web page you want to process
urls = [
    "https://en.wikipedia.org/wiki/Public_health",
    "https://en.wikipedia.org/wiki/Environmental_health",
    "https://en.wikipedia.org/wiki/Air_pollution",
    "https://en.wikipedia.org/wiki/Water_pollution",
    "https://en.wikipedia.org/wiki/Climate_change_and_health",
    "https://en.wikipedia.org/wiki/Toxicology",
    "https://en.wikipedia.org/wiki/Global_health",
    "https://en.wikipedia.org/wiki/World_Health_Organization",
    "https://en.wikipedia.org/wiki/Epidemiology",
    "https://en.wikipedia.org/wiki/Public_health_law"
]


def plot_similarities(urls, similarities):
    # Sorting the documents by similarity score for better visualization
    urls, similarities = zip(*sorted(zip(urls, similarities), key=lambda x: x[1], reverse=True))

    plt.figure(figsize=(10, 5))
    plt.bar(urls, similarities, color='blue')
    plt.xlabel('Documents')
    plt.ylabel('Similarity Score')
    plt.title('Document Similarity')
    plt.xticks(rotation=90)  # Rotate document labels for better readability
    plt.tight_layout()  # Adjust layout to make room for label rotation
    plt.show()


def query_relevant_documents(query, texts, vectorizer, top_n=5):
    while True:
        print("\nSelect a method:")
        print("1. Cosine Similarity")
        print("2. Dice Similarity")
        print("3. Jaccard Similarity")
        print("4. Euclidean Distance")
        print("5. Manhattan Distance")
        print("6. Exit")

        method_choice = input("Enter your choice: ")
        if method_choice == "6":
            break

        method = "cosine" if method_choice == "1" else \
            "dice" if method_choice == "2" else \
                "jaccard" if method_choice == "3" else \
                    "euclidean" if method_choice == "4" else \
                        "manhattan"

        # Preprocess and vectorize the query
        similarities = calculate_similarities(query, texts, vectorizer, method)

        # Get the top N indices of the most similar documents
        top_indices = np.argsort(similarities)[-top_n:]
        results = [(urls[i], similarities[i]) for i in top_indices]
        for url, similarity in results:
            print(f"URL: {url}, Similarity: {similarity}")
        # plot_similarities(urls, similarities)


def calculate_similarities(query, texts, vectorizer, method):
    query_processed = preprocess_text(query)
    query_vector = vectorizer.transform([query_processed]).toarray().ravel()  # Flatten the vector

    # Transform texts to vectors and ensure they are flat
    text_vectors = vectorizer.transform(texts).toarray()

    similarities = []
    for text_vector in text_vectors:
        if method == "cosine":
            similarity = cosine_similarity(query_vector, text_vector.ravel())  # Flatten the vector
        elif method == "dice":
            similarity = dice_similarity(query_vector, text_vector.ravel())
        elif method == "jaccard":
            similarity = jaccard_similarity(query_vector, text_vector.ravel())
        elif method == "euclidean":
            similarity = euclidean_distance(query_vector, text_vector.ravel())
        elif method == "manhattan":
            similarity = manhattan_distance(query_vector, text_vector.ravel())

        similarities.append(similarity)

    return similarities


def main_menu(db, matrix, vectorizer):
    while True:
        print("\nDocument Base System Menu:")
        print("0. Add Document to Database (Text)")
        print("1. Add Documents to Database (Link)")
        print("2. Check List of Documents")
        print("3. Compare Document Similarity")
        print("4. Compare Document with All Documents (ALL METHODS)")
        print("5. Remove Documents from Database")
        print("6. Show Term-Document Matrix")
        print("7. Similarity Matrix")
        print("8. Query Document Relevance")
        print("9. Exit")

        choice = input("Enter your choice: ")
        if choice == "9":
            break
        handle_menu_choice(db, choice, matrix, vectorizer)


def handle_menu_choice(db, choice, matrix, vectorizer):
    cursor = db.cursor()
    try:
        if choice == "0":
            text = input("Enter document text: ")
            add_document_by_text(cursor, text)
        elif choice == "1":
            link = input("Enter document link: ")
            add_document_by_link(cursor, link)
        elif choice == "2":
            list_documents(cursor)
        elif choice == "3":
            id1 = int(input("Enter first document ID: "))
            id2 = int(input("Enter second document ID: "))
            compare_documents(cursor, id1, id2, matrix)
        elif choice == "4":
            compare_with_all_documents(db, matrix)
        elif choice == "5":
            menu_eliminate_document(cursor)
        elif choice == "6":
            show_term_document_matrix(cursor)
        elif choice == "7":
            show_similarity_matrix(cursor, vectorizer)
        elif choice == "8":
            query = input("Enter a query: ")
            results = query_relevant_documents(query, build_corpus_from_db(cursor), vectorizer)
            for url, similarity in results:
                print(f"URL: {url}, Similarity: {similarity}")
        db.commit()
    except Exception as e:
        print("An error occurred:", e)
        db.rollback()
    finally:
        cursor.close()


def show_similarity_matrix(cursor, vectorizer):
    texts = build_corpus_from_db(cursor)
    documents = fetch_all_documents(cursor)
    if not texts:
        print("No documents available to create similarity matrix.")
        return

    print("\nSelect a method for similarity measurement:")
    print("1. Cosine Similarity")
    print("2. Dice Similarity")
    print("3. Jaccard Similarity")
    print("4. Euclidean Distance")
    print("5. Manhattan Distance")
    choice = input("Enter your choice: ")
    method = "cosine" if choice == "1" else \
        "dice" if choice == "2" else \
            "jaccard" if choice == "3" else \
                "euclidean" if choice == "4" else \
                    "manhattan"

    similarity_matrix = create_similarity_matrix(texts, vectorizer, method)

    # Create headers using document IDs
    headers = [f"Doc{doc_id}" for doc_id, _, _ in documents]
    print("\nSimilarity Matrix:")
    print("\t" + "\t".join(headers))
    for i, row in enumerate(similarity_matrix):
        formatted_row = "\t".join(f"{sim:.2f}" for sim in row)
        print(f"{headers[i]}:\t{formatted_row}")


def create_similarity_matrix(texts, vectorizer, method):
    # Transform texts to vectors
    text_vectors = vectorizer.transform(texts).toarray()

    # Calculate the number of documents
    num_docs = text_vectors.shape[0]

    # Initialize an empty similarity matrix
    similarity_matrix = np.zeros((num_docs, num_docs))

    # Calculate similarity for each pair of documents
    for i in range(num_docs):
        for j in range(i, num_docs):
            if method == "cosine":
                similarity = cosine_similarity(text_vectors[i], text_vectors[j])
            elif method == "dice":
                similarity = dice_similarity(text_vectors[i], text_vectors[j])
            elif method == "jaccard":
                similarity = jaccard_similarity(text_vectors[i], text_vectors[j])
            elif method == "euclidean":
                similarity = euclidean_distance(text_vectors[i], text_vectors[j])
            elif method == "manhattan":
                similarity = manhattan_distance(text_vectors[i], text_vectors[j])
            # Fill both (i, j) and (j, i) to make the matrix symmetric
            similarity_matrix[i, j] = similarity_matrix[j, i] = similarity

    return similarity_matrix


def add_document_by_text(cursor, text):
    if not document_exists(cursor, text, by_text=True):
        doc_id = insert_document(cursor, text)
        terms = preprocess_text(text)
        term_frequencies = collections.Counter(terms)
        for term, freq in term_frequencies.items():
            term_id = get_or_create_term_id(cursor, term)
            insert_or_update_frequency(cursor, doc_id, term_id, freq)
    else:
        print("Document already exists in the database.")


def add_document_by_link(cursor, link):
    text = fetch_and_extract_text(link)
    if not document_exists(cursor, link):
        doc_id = insert_document(cursor, text, link)
        terms = preprocess_text(text)
        term_frequencies = collections.Counter(terms)
        for term, freq in term_frequencies.items():
            term_id = get_or_create_term_id(cursor, term)
            insert_or_update_frequency(cursor, doc_id, term_id, freq)
    else:
        print("Document already exists in the database.")


def show_term_document_matrix(cursor):
    documents = fetch_all_documents(cursor)
    terms = fetch_all_terms(cursor)

    doc_id_to_index = {doc_id: index for index, (doc_id, _, _) in enumerate(documents)}
    term_id_to_index = {term_id: index for index, (term_id, _) in enumerate(terms)}

    matrix = [[0 for _ in range(len(terms))] for _ in range(len(documents))]

    frequencies = fetch_all_frequencies(cursor)
    for doc_id, term_id, frequency in frequencies:
        doc_index = doc_id_to_index.get(doc_id)
        term_index = term_id_to_index.get(term_id)
        if doc_index is not None and term_index is not None:
            matrix[doc_index][term_index] = frequency

    normalized_matrix = []
    for row in matrix:
        total_terms = sum(row)
        if total_terms > 0:
            normalized_row = [round((freq / total_terms), 2) for freq in row]
        else:
            normalized_row = [0] * len(row)  # Ensure all zeros if no terms
        normalized_matrix.append(normalized_row)

    # Display headers with term IDs
    print("Term-Document Matrix:")
    headers = [f"T{term_id}" for term_id, _ in terms]
    print("\t" + "\t".join(headers))
    for (doc_id, _, _), row in zip(documents, normalized_matrix):
        print(f"Doc {doc_id}:\t" + "\t".join(f"{freq}" if freq != 0 else "0" for freq in row))


def compare_with_all_documents(db, matrix):
    cursor = db.cursor()
    documents = compare_list_documents(cursor)  # Fetch a list of all documents
    if not documents:
        print("No documents available for comparison.")
        return

    # Create a mapping of document ID to matrix index
    id_to_index = {doc_id: index for index, (doc_id, _) in enumerate(documents)}

    print("Available Documents:")
    for doc_id, title in documents:
        print(f"ID: {doc_id}, Title: {title}")

    try:
        selected_id = int(input("Enter the ID of the document to compare: "))
        selected_index = id_to_index.get(selected_id)
        if selected_index is None or selected_index >= len(matrix):
            print("Document ID not found or out of matrix range.")
            return
        selected_vector = matrix[selected_index]
    except ValueError:
        print("Invalid input. Please enter a valid document ID.")
        return

    for doc_id, title in documents:
        doc_index = id_to_index.get(doc_id)
        if doc_id == selected_id or doc_index is None or doc_index >= len(matrix):
            continue

        compare_vector = matrix[doc_index]
        print(f"\nComparing Document {selected_id} with Document {doc_id} ({title}):")
        print(f"Cosine Similarity: {cosine_similarity(selected_vector, compare_vector)}")
        print(f"Dice Similarity: {dice_similarity(selected_vector, compare_vector)}")
        print(f"Jaccard Similarity: {jaccard_similarity(selected_vector, compare_vector)}")
        print(f"Euclidean Distance: {euclidean_distance(selected_vector, compare_vector)}")
        print(f"Manhattan Distance: {manhattan_distance(selected_vector, compare_vector)}")
    cursor.close()


def menu_eliminate_document(cursor):
    while True:
        print("\nDocument Elimination Menu:")
        print("1. Eliminate Document by ID")
        print("2. Eliminate Document by Title")
        print("9. Go Back")

        choice = input("Select an elimination method: ")
        if choice == '9':
            break

        if choice == '1':
            doc_id = input("Enter the document ID: ")
            eliminate_document_by_id(cursor, doc_id)
        elif choice == '2':
            title = input("Enter the document title: ")
            eliminate_document_by_title(cursor, title)
        else:
            print("Invalid choice. Please select a valid option.")


def compare_documents(cursor, id1, id2, matrix):
    while True:
        print("\nDocument Comparison Menu:")
        print("1. Cosine Similarity")
        print("2. Dice Similarity")
        print("3. Jaccard Similarity")
        print("4. Euclidean Distance")
        print("5. Manhattan Distance")
        print("9. Go Back")

        choice = input("Select a comparison method: ")
        if choice == '9':
            break

        vector_a = matrix[id1]
        vector_b = matrix[id2]

        if choice == '1':
            print(f"Cosine Similarity: {cosine_similarity(vector_a, vector_b)}")
        elif choice == '2':
            print(f"Dice Similarity: {dice_similarity(vector_a, vector_b)}")
        elif choice == '3':
            print(f"Jaccard Similarity: {jaccard_similarity(vector_a, vector_b)}")
        elif choice == '4':
            print(f"Euclidean Distance: {euclidean_distance(vector_a, vector_b)}")
        elif choice == '5':
            print(f"Manhattan Distance: {manhattan_distance(vector_a, vector_b)}")
        else:
            print("Invalid choice. Please select a valid option.")


def add_document_with_terms_and_frequencies(cursor, text, url=None):
    doc_id = insert_document(cursor, text, url)
    terms = preprocess_text(text)
    term_frequencies = collections.Counter(terms)
    for term, freq in term_frequencies.items():
        term_id = get_or_create_term_id(cursor, term)
        insert_or_update_frequency(cursor, doc_id, term_id, freq)


def show_similarity_matrixInterface(cursor, vectorizer):
    texts = build_corpus_from_db(cursor)
    documents = fetch_all_documents(cursor)
    if not texts:
        st.write("No documents available to create similarity matrix.")
        return None

    method = st.selectbox(
        'Select a method for similarity measurement:',
        ['cosine', 'dice', 'jaccard', 'euclidean', 'manhattan']
    )

    similarity_matrix = create_similarity_matrix(texts, vectorizer, method)

    # Prepare the data for display
    headers = [f"Doc{doc_id}" for doc_id, _, _ in documents]
    df = pd.DataFrame(similarity_matrix, index=headers, columns=headers)

    # Display the similarity matrix
    st.write("Similarity Matrix:")
    st.dataframe(df.style.format("{:.2f}"))


def show_term_document_matrixInterface(cursor):
    documents = fetch_all_documents(cursor)
    terms = fetch_all_terms(cursor)

    # Create index and columns for the DataFrame
    index = [doc_id for doc_id, _, _ in documents]
    columns = [f"T{term_id}" for term_id, _ in terms]

    # Initialize the DataFrame with zeros
    matrix_df = pd.DataFrame(0, index=index, columns=columns)

    # Populate the DataFrame with frequency data
    frequencies = fetch_all_frequencies(cursor)
    for doc_id, term_id, frequency in frequencies:
        # Ensure to use the correct column identifier with 'T' prefix
        term_column = f"T{term_id}"
        if doc_id in matrix_df.index and term_column in matrix_df.columns:
            matrix_df.at[doc_id, term_column] = frequency

    # Normalize the frequencies to show proportions
    if not matrix_df.empty and matrix_df.sum(axis=1).sum() > 0:  # Only normalize if there is data to normalize
        matrix_df = matrix_df.div(matrix_df.sum(axis=1), axis=0).fillna(0)

    return matrix_df


def query_relevant_documentsInterface(query, texts, vectorizer, method, top_n=5):
    if not query:
        return []  # Return empty if no query is provided

    # Preprocess and vectorize the query
    similarities = calculate_similarities(query, texts, vectorizer, method)

    print(f"Total documents compared: {len(similarities)}")
    if method in ['euclidean', 'manhattan']:
        top_indices = np.argsort(similarities)[:top_n]  # Ascending order for distances
    else:
        top_indices = np.argsort(similarities)[-top_n:][::-1]  # Descending order for similarities

    print(f"Top indices: {top_indices}")
    print(f"URLs length: {len(urls)}")

    results = [(urls[i], similarities[i]) for i in top_indices if i < len(urls)]
    return results


def compare_documentsInterface(cursor, id1, id2, matrix, method):
    vector_a = matrix[id1]
    vector_b = matrix[id2]

    if method == 'cosine':
        similarity = cosine_similarity(vector_a, vector_b)
    elif method == 'dice':
        similarity = dice_similarity(vector_a, vector_b)
    elif method == 'jaccard':
        similarity = jaccard_similarity(vector_a, vector_b)
    elif method == 'euclidean':
        similarity = euclidean_distance(vector_a, vector_b)
    elif method == 'manhattan':
        similarity = manhattan_distance(vector_a, vector_b)

    return similarity


def initialize_database():
    db = connect_db()
    cursor = db.cursor()

    if not check_if_documents_exist(cursor):
        for url in urls:
            if not document_exists(cursor, url):
                text = fetch_and_extract_text(url)
                processed_text = preprocess_text(text)
                add_document_with_terms_and_frequencies(cursor, processed_text, url)
                print(f"Document added from URL: {url}")
            else:
                print(f"Document already in the database: {url}")
        db.commit()

    else:
        print("Database already has documents.")

    return db, cursor
