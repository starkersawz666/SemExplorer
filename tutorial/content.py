import streamlit as st
import os

def add_video(filename):
    video_col = st.columns([1, 1])
    with video_col[0]:
        st.video(os.path.join('./tutorial/videos', filename), format='video/mp4', start_time=0, loop=True)

def add_download(filename, label):
    with open(os.path.join('./tutorial/files', filename), "rb") as file:
        st.download_button(
            label=label,
            data=file,
            file_name=filename,
            # mime='text/csv'
        )

def section_intro():
    st.header("Introduction")
    st.write("""
    SemanticSearch is an advanced tool designed to empower researchers with semantic search technology to access customized datasets. Our platform provides an intuitive web interface for extracting relevant information, searching through extensive datasets, and submitting your data for analysis. Access the application by visiting http://ns2.nilou.top, and no prior installation is required.
    """)

def section_search():
    st.header("Semantic Search")
    st.subheader("Step 1: Requirements Extraction")
    st.write("""
    - Select a schema, enter your requirements in natural language in the text area, and click 'Extract Filters' to extract the filters from the text.
    """)
    st.write("""You can try the following example on schema 'Networks': 
             
             An undirected network about the relationship between users on social media, with more than 10000 nodes, volume >= 100000, a max degree between 500 and 2000, and an average degree smaller than 50.""")
    # add_video("Extraction.mp4")
    st.subheader("Step 2: Filters Adjustment")
    st.write("""
    - After the filters have been extracted from your input, they will appear in a manageable list below. This list allows you to review and adjust the filters to refine your search criteria.
    """)
    st.write("You can try to cancel all the 'accept missing' checkboxes.")

    # add_video("Adjustment.mp4")

    st.subheader("Step 3: Searching")

    st.write("""
    - Click the Search button below to perform the search. The system will return a list of relevant datasets that match your criteria. You can click on the dataset name to view its detailed information.
    """)

def section_create():
    st.header("Create Schema")
    st.write("""
    - **Name of Schema**: the displayed name for schema selection.
    - **Title and Link**: the primary displayed result for searching.
    - **Add Item**: add more attributes of different types to help you search within the dataset.
    """)
    st.write("You can also upload a csv, excel spreadsheet or specific json file for this system to create a schema quickly.")
    st.subheader("Example: How to Create a Schema")
    st.write("""
    Let's assume we want to create a schema called **'Network'** with the following attributes:
    
    - **name**: The name of the network.
    - **url**: The URL associated with the network.
    - **directed**: A Boolean attribute indicating if the network is directed.
    - **size**: The number of nodes in the network.
    - **volume**: The number of edges in the network.
    - **description**: Details about the network.
    - **category**: A few words that describe the network's category.
    - **average_degree**: A less commonly used attribute.

    In the **Create** interface:
    - Fill **"Name of Schema"** with `Network`.
    - Set **"Title Attribute"** to `name` and **"Link Attribute"** to `url`.
    - Use **Add Items** to define the remaining attributes:
        - `directed` → Boolean type.
        - `size` and `volume` → Statistics-basic type.
        - `description` → Text-semantic type.
        - `average_degree` → Statistics-advanced type.
        - `category` → Depending on semantic search needs, choose either Text-semantic or Text-nonsemantic type.
    """)

def section_submission():
    st.header("Import Data")
    st.write("""
    - **Uploading Files**: select a schema, and click on `Upload File` button and select the file you wish to submit. Supported formats include csv and excel spreadsheet.
    - **Auto-matching**: The system will automatically try to match your data columns to our database. You can review and adjust these mappings before final submission.
    - **Submission**: Once satisfied with the mappings, click `Submit` to finalize your data submission.
    """)
    st.write("You can try data submission function on 'Networks':")
    add_download('networks.csv', "Sample network dataset")
    st.write("After submission, click 'Update Embedding' so that the new data can be searched.")

def section_faq():
    st.header("FAQ")
    st.write("""
    - **Q**: Are the search results reproducible?
    - **A**: The "Filters Extraction" process has a certain degree of randomness due to the use of LLM, but the search results are reproducible when the search text and conditions are exactly the same.
    """)
    st.header("Getting Help")
    st.write("""
    If you encounter any issues or have questions, please contact us at zixinwei1@link.cuhk.edu.cn.
    """)

def show_tutorial():
    st.title("User Guide for SemanticSearch")
    section_intro()
    section_search()
    section_create()
    section_submission()
    section_faq()





    