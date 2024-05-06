import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer

from DocumentProcessing import initialize_database, build_corpus_from_db, add_document_by_text, add_document_by_link, \
    compare_documents_interface, show_term_document_matrixInterface, query_relevant_documentsInterface, \
    show_similarity_matrixInterface
from Queries import *


def main():
    """
    Main method for Document Management System application.

    :return: None
    """
    db, cursor = initialize_database()

    try:
        tfidf_vectorizer = TfidfVectorizer()
        texts = build_corpus_from_db(cursor)

        if texts:
            tfidf_vectorizer.fit_transform(texts)
            st.title('Document Management System')
            options = ["Add Document (Text)", "Add Document (Link)", "Eliminate document", "View Documents",
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

            elif choice == "Eliminate document":
                elimination_method = st.radio("Select elimination method:", ('By ID', 'By Title'))
                if elimination_method == 'By ID':
                    doc_id = st.text_input("Enter Document ID to eliminate:")
                    if st.button('Eliminate Document'):
                        eliminate_document_by_id(cursor, doc_id)
                        st.success(f"Document ID {doc_id} eliminated successfully!")
                elif elimination_method == 'By Title':
                    title = st.text_input("Enter Document Title to eliminate:")
                    if st.button('Eliminate Document'):
                        eliminate_document_by_title(cursor, title)
                        st.success(f"Document titled '{title}' eliminated successfully!")
                db.commit()

            elif choice == "View Documents":
                documents_df = list_documents_for_interface(cursor)
                if not documents_df.empty:
                    st.dataframe(documents_df)
                else:
                    st.write("No documents found.")

            elif choice == "Compare Documents":
                st.subheader("For Cosine, Dice, and Jaccard, a value of 0 means no similarity, and 1 means a complete match.")
                st.subheader("For Euclidean and Manhattan, the closer the value is to 0, the better the similarity.")
                tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
                compare_documents_interface(cursor, tfidf_matrix.toarray())


            if choice == "Show Term-Document Matrix":
                matrix_df = show_term_document_matrixInterface(cursor)
                if not matrix_df.empty:
                    st.dataframe(matrix_df.style.format("{:.2%}"), height=600)
                else:
                    st.write("No term-document matrix available.")

            if choice == "Query Document Relevance":
                st.subheader("For Cosine, Dice, and Jaccard, a value of 0 means no similarity, and 1 means a complete match.")
                st.subheader("For Euclidean and Manhattan, the closer the value is to 0, the better the similarity.")
                query = st.text_input("Enter a query:")
                method = st.selectbox(
                    'Select a method:',
                    ['cosine', 'dice', 'jaccard', 'euclidean', 'manhattan']
                )
                if st.button('Search') and query:
                    results = query_relevant_documentsInterface(cursor, query, texts, tfidf_vectorizer, method)
                    if results:
                        df = pd.DataFrame(results, columns=['URL', 'Similarity'])
                        st.dataframe(df)
                    else:
                        st.write("No relevant documents found or no query entered.")

            elif choice == "Show Similarity Matrix":
                st.subheader("For Cosine, Dice, and Jaccard, a value of 0 means no similarity, and 1 means a complete match.")
                st.subheader("For Euclidean and Manhattan, the closer the value is to 0, the better the similarity.")
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
