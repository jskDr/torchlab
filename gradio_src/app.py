import gradio as gr

# Function to handle chat responses
def chat_response(user_input, history):
    history = history or []
    response = f"You said: {user_input}"
    history.append((user_input, response))
    return history, ""

# Defining the Gradio chat interface
with gr.Blocks() as demo:
    gr.Markdown("# ChatGPT-like Chatbot")
    gr.Markdown("Welcome to the ChatGPT-like chatbot demo! Type below to start chatting.")
    
    chatbot = gr.Chatbot()
    user_input = gr.Textbox(placeholder="Type your message here...", lines=1)
    submit_button = gr.Button("Send")
    
    # Capture user input and provide response, clear input after submission
    user_input.submit(fn=chat_response, inputs=[user_input, chatbot], outputs=[chatbot, user_input])
    submit_button.click(fn=chat_response, inputs=[user_input, chatbot], outputs=[chatbot, user_input])

# Launching the Gradio app
demo.launch()