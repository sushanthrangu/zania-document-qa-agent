import requests
import streamlit as st


API_URL = "http://127.0.0.1:8000/api/v1/qa"


st.set_page_config(
    page_title="Document QA Agent",
    page_icon="📄",
    layout="wide",
)


st.title("📄 Document QA Agent")

st.write(
    "Upload a questions JSON file and a PDF or JSON document "
    "to generate grounded answers using retrieval-augmented generation."
)

st.divider()

st.subheader("Upload Files")

questions_file = st.file_uploader(
    "Questions JSON",
    type=["json"],
    help="Upload a JSON file containing the questions to answer.",
)

document_file = st.file_uploader(
    "Document",
    type=["pdf", "json"],
    help="Upload the PDF or JSON document used as the answer source.",
)

analyze_button = st.button(
    "Analyze Document",
    type="primary",
    use_container_width=True,
)


if analyze_button:
    if questions_file is None or document_file is None:
        st.warning(
            "Please upload both the questions JSON file "
            "and the document before analyzing."
        )

    else:
        files = {
            "questions_file": (
                questions_file.name,
                questions_file.getvalue(),
                questions_file.type or "application/json",
            ),
            "document_file": (
                document_file.name,
                document_file.getvalue(),
                document_file.type or "application/octet-stream",
            ),
        }

        try:
            with st.spinner("Analyzing document..."):
                response = requests.post(
                    API_URL,
                    files=files,
                    timeout=300,
                )

        except requests.exceptions.ConnectionError:
            st.error(
                "Unable to connect to the API. "
                "Make sure the FastAPI server is running."
            )
            st.stop()

        except requests.exceptions.Timeout:
            st.error(
                "The analysis took too long and timed out. "
                "Please try again."
            )
            st.stop()

        except requests.exceptions.RequestException as exc:
            st.error(
                f"Unable to complete the request: {exc}"
            )
            st.stop()

        if response.status_code == 200:
            data = response.json()
            results = data.get("results", [])

            st.success(
                f"Analysis complete. {len(results)} questions processed."
            )

            st.subheader("Results")

            for index, result in enumerate(results, start=1):
                st.markdown(
                    f"### {index}. {result['question']}"
                )

                st.write(result["answer"])

                confidence = result.get("confidence", "low")

                if confidence == "high":
                    st.success("Confidence: HIGH")
                elif confidence == "medium":
                    st.warning("Confidence: MEDIUM")
                else:
                    st.error("Confidence: LOW")

                sources = result.get("sources", [])

                if sources:
                    st.markdown("**Sources**")

                    for source in sources:
                        source_name = source.get(
                            "source",
                            "Unknown",
                        )
                        page = source.get("page")

                        if page is not None:
                            st.write(
                                f"📄 {source_name} — Page {page}"
                            )
                        else:
                            st.write(
                                f"📄 {source_name}"
                            )

                else:
                    st.caption(
                        "No supporting source identified."
                    )

                st.divider()

        else:
            try:
                error_data = response.json()
                error_message = error_data.get(
                    "detail",
                    "The API request failed.",
                )

            except ValueError:
                error_message = "The API request failed."

            st.error(
                f"API error ({response.status_code}): "
                f"{error_message}"
            )