import streamlit as st
from llm_with_tools import process_query

st.title("🔮 Chatbot with Tools + LangGraph")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Initialize message counter for unique keys
if "message_count" not in st.session_state:
    st.session_state.message_count = 0

# User input
user_input = st.text_input("You:", key=f"user_input_{st.session_state.message_count}")

if st.button("Send", key=f"send_{st.session_state.message_count}") and user_input.strip():
    # Add user message
    st.session_state.chat_history.append({"role": "user", "content": user_input.strip()})
    
    try:
        # Get final assistant message
        new_messages = process_query(user_input.strip(), st.session_state.chat_history)
        
        # Add assistant response
        st.session_state.chat_history.extend(new_messages)
        
    except ValueError as e:
        if "Checkpointer requires" in str(e):
            # Handle LangGraph configuration error
            st.error("Configuration error: Please check your LangGraph setup. You may need to provide a thread_id in your configuration.")
            st.error(f"Error details: {str(e)}")
        else:
            st.error(f"An error occurred: {str(e)}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
    
    # Increment counter to create new widgets on rerun
    st.session_state.message_count += 1
    
    # Rerun to clear input and show new message
    st.rerun()

# Show chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])