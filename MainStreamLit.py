import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer

from DocumentProcessing import initialize_database, build_corpus_from_db, add_document_by_text, add_document_by_link, \
    compare_documentsInterface, show_term_document_matrixInterface, query_relevant_documentsInterface, \
    show_similarity_matrixInterface
from Queries import list_documents, list_documents_for_interface, close_db


def main():
    db, cursor = initialize_database()

    try:
        tfidf_vectorizer = TfidfVectorizer()
        texts = build_corpus_from_db(cursor)

        if texts:
            tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
            st.title('Document Management System')
            options = ["Add Document (Text)", "Add Document (Link)", "View Documents",
                       "Compare Documents", "Show Term-Document Matrix",
                       "Query Document Relevance", "Show Similarity Matrix", "Exit"]
            choice = st.sidebar.selectbox("Choose an option", options)

            if choice == "Add Document (Text)":
                doc_text = st.text_area("Enter document text:")
                if st.button('Add Document'):
                    add_document_by_text(cursor, doc_text)
                    db.commit()
                    st.success("Document added successfully!")
                    list_documents(cursor)

            elif choice == "Add Document (Link)":
                doc_link = st.text_input("Enter document link:")
                if st.button('Fetch and Add Document'):
                    add_document_by_link(cursor, doc_link)
                    db.commit()
                    st.success("Document fetched and added successfully!")
                    list_documents(cursor)

            elif choice == "View Documents":
                documents_df = list_documents_for_interface(cursor)
                if not documents_df.empty:
                    st.dataframe(documents_df)
                else:
                    st.write("No documents found.")

            elif choice == "Compare Documents":
                doc_id1 = st.number_input("Enter first document ID:", min_value=1, format="%d")
                doc_id2 = st.number_input("Enter second document ID:", min_value=1, format="%d")
                method = st.selectbox('Select Comparison Method',
                                      ['cosine', 'dice', 'jaccard', 'euclidean', 'manhattan'])
                if st.button('Compare'):
                    tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
                    similarity = compare_documentsInterface(cursor, doc_id1, doc_id2, tfidf_matrix.toarray(), method)
                    st.write(f"The {method} similarity between documents is {similarity:.2f}")

            if choice == "Show Term-Document Matrix":
                matrix_df = show_term_document_matrixInterface(cursor)
                if not matrix_df.empty:
                    st.dataframe(matrix_df.style.format("{:.2%}"), height=600)
                else:
                    st.write("No term-document matrix available.")

            if choice == "Query Document Relevance":
                query = st.text_input("Enter a query:")
                method = st.selectbox(
                    'Select a method:',
                    ['cosine', 'dice', 'jaccard', 'euclidean', 'manhattan']
                )
                if st.button('Search') and query:
                    results = query_relevant_documentsInterface(query, texts, tfidf_vectorizer, method)
                    if results:
                        df = pd.DataFrame(results, columns=['URL', 'Similarity'])
                        st.dataframe(df)
                    else:
                        st.write("No relevant documents found or no query entered.")

            elif choice == "Show Similarity Matrix":
                show_similarity_matrixInterface(cursor, tfidf_vectorizer)

            elif choice == "Exit":
                st.write("Exiting the application.")
                close_db(db)
                st.stop()
        else:
            print("No sufficient text data to create TF-IDF matrix.")
    finally:
        cursor.close()
        close_db(db)


if __name__ == "__main__":
    main()
