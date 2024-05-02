import collections
import requests
from bs4 import BeautifulSoup
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from Queries import *
from Formulas import *

# Load the English NLP model
nlp = spacy.load("en_core_web_sm")

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
    X = vectorizer.fit_transform(texts)

    # Apply Truncated SVD (LSI)
    svd_model = TruncatedSVD(n_components=num_topics)
    svd_model.fit(X)

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

        # Get the top N indices of the most similar documents
        top_indices = np.argsort(similarities)[-top_n:]
        results = [(urls[i], similarities[i]) for i in top_indices]
        for url, similarity in results:
            print(f"URL: {url}, Similarity: {similarity}")


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
            print("not implemented")
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


def add_document_by_text(cursor, text):
    if not document_exists(cursor, text, by_text=True):
        insert_document(cursor, text)
    else:
        print("Document already exists in the database.")


def add_document_by_link(cursor, link):
    text = fetch_and_extract_text(link)
    if not document_exists(cursor, link):
        insert_document(cursor, text, link)
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
            normalized_row = [round((freq / total_terms), 2) for freq in row]  # Use percentage normalization
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
    documents = comparate_list_documents(cursor)  # Fetch list of all documents
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
            eliminate_document_byid(cursor, doc_id)
        elif choice == '2':
            title = input("Enter the document title: ")
            eliminate_document_bytitle(cursor, title)
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

        vectorA = matrix[id1]
        vectorB = matrix[id2]

        if choice == '1':
            print(f"Cosine Similarity: {cosine_similarity(vectorA, vectorB)}")
        elif choice == '2':
            print(f"Dice Similarity: {dice_similarity(vectorA, vectorB)}")
        elif choice == '3':
            print(f"Jaccard Similarity: {jaccard_similarity(vectorA, vectorB)}")
        elif choice == '4':
            print(f"Euclidean Distance: {euclidean_distance(vectorA, vectorB)}")
        elif choice == '5':
            print(f"Manhattan Distance: {manhattan_distance(vectorA, vectorB)}")
        else:
            print("Invalid choice. Please select a valid option.")


def add_document_with_terms_and_frequencies(cursor, text, url=None):
    doc_id = insert_document(cursor, text, url)
    terms = preprocess_text(text)
    term_frequencies = collections.Counter(terms)
    for term, freq in term_frequencies.items():
        term_id = get_or_create_term_id(cursor, term)
        insert_or_update_frequency(cursor, doc_id, term_id, freq)


def main():
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

    try:
        texts = build_corpus_from_db(cursor)
        tfidf_vectorizer = TfidfVectorizer()
        if texts:
            tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
            main_menu(db, tfidf_matrix.toarray(), tfidf_vectorizer)
        else:
            print("No sufficient text data to create TF-IDF matrix.")
    finally:
        cursor.close()
        close_db(db)


if __name__ == "__main__":
    main()
