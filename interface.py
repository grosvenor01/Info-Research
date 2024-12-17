import streamlit as st
from Final import *
import matplotlib.pyplot as plt
def main():
    st.title("Information Retrieval System")

    # Sidebar for user inputs
    st.sidebar.header("Query Parameters")
    query = st.sidebar.text_input("Enter your query:", "")
    stemming_method = st.sidebar.selectbox("Stemming Method:", ["No Stemming", "Porter", "Lancaster"])
    preprocessing_method = st.sidebar.selectbox("Preprocessing Method:", ["Split", "Regex"])
    retrieval_method = st.sidebar.selectbox("Retrieval Method:", ["RSV", "Cosine", "Jaccard", "BM25", "Boolean"])

    # BM25 parameters (only show if BM25 is selected)
    if retrieval_method == "BM25":
        k = st.sidebar.number_input("K:", value=1.2, step=0.1)
        b = st.sidebar.number_input("B:", value=0.75, step=0.05)


    if st.sidebar.button("Submit Query"):
        if query:
            try:
                if retrieval_method == "RSV":
                    results = rsv(query, stemming_method, preprocessing_method)
                elif retrieval_method == "Cosine":
                    results = cosine(query, stemming_method, preprocessing_method)
                elif retrieval_method == "Jaccard":
                    results = jaccard(query, stemming_method, preprocessing_method)
                elif retrieval_method == "BM25":
                    results = bm25(query, stemming_method, preprocessing_method, k, b)
                elif retrieval_method == "Boolean":
                    results = boolean_model(query, stemming_method, preprocessing_method)
                else:
                    st.error("Invalid retrieval method selected.")
                    return

                # Display results (replace with your desired output format)
                st.subheader("Search Results:")
                if isinstance(results, dict):
                    st.write("Ranked Results (Document ID : Score):")
                    for doc_id, score in results.items():
                        st.write(f"{doc_id}: {score}")
                elif isinstance(results, pd.DataFrame):
                    st.dataframe(results)
                else:
                    st.write(results)  # Handle other result types as needed

            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.warning("Please enter a query.")
        p , p5 , p10 , r ,  F_score , precesions , recalls , precisions2 , recalls2 = model_evaluation(query,results)
        show_plot(recalls2, precisions2)
        st.title("Metrics and Performance Evaluation")
        # Display metrics
        st.subheader("Key Metrics")
        st.metric(label="Precision (P)", value=f"{p:.2f}")
        st.metric(label="Precision@5 (P@5)", value=f"{p5:.2f}")
        st.metric(label="Precision@10 (P@10)", value=f"{p10:.2f}")
        st.metric(label="Recall (R)", value=f"{r:.2f}")
        st.metric(label="F-Score", value=f"{F_score:.2f}")

if __name__ == "__main__":
    main()