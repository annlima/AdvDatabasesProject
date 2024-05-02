from sklearn.feature_extraction.text import TfidfVectorizer

from DocumentProcessing import initialize_database, build_corpus_from_db, main_menu
from Queries import close_db


def main():
    """
    Main function to initiate the program.

    :return: None
    """
    db, cursor = initialize_database()

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
