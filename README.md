# EasyLaw - Supreme Courts Legal Information Chatbot

## Introduction
This is an OpenAI GPT3.5-Turbo based LLM chatbot that uses data scrapped from https://www.judiciary.gov.sg to answer a user's query using Retrieval Augmented Generation (RAG).
For this project, Langchain's Question Answer Chain was used, with conversational buffer memory to simulate a continuous convosation. (Note: langchain code used is currently not supported)
ChromaDB is used to create a vector database encoded using OpenAI's embedding vector model.

The frontend is created using HTML & CSS, using Flask RESTful API to connect the frontend and backend.

The production version of the application is served using Waitress, a python WSGI server.

A version of this application was hosted on AWS EC2, under the domain name https://easylaw.com/ (CURRENTLY INACTIVE)

## Setup

1. If you don’t have Python installed, [install it from here](https://www.python.org/downloads/)

2. Clone this repository

3. Navigate into the project directory

   ```bash
   $ cd [/path/to/directory]
   ```

4. Create a new virtual environment

   ```bash
   $ python -m venv venv
   $ . venv/Scripts/activate
   ```

5. Install the requirements

   ```bash
   $ pip install -r requirements.txt
   ```

6. Add your [API key](https://beta.openai.com/account/api-keys) to a `.env` file

8. Start a local instance

   ```bash
   $ flask run
   ```

9. Start an instance on AWS EC2

   install ubuntu on Windows using WSL:

   https://techcommunity.microsoft.com/t5/windows-11/how-to-install-the-linux-windows-subsystem-in-windows-11/td-p/2701207
   
   connect to ubuntu:
   ```bash
   ssh -i [/path/to/ssh.pem] ubuntu@3.1.181.57
   ```
   
   login to session and activate virtual env:
   ```bash
   cd A2J
   screen -r 47888.A2J
   source venv/bin/activate
   ```

   start the application:
   ```bash
   waitress-serve --host 0.0.0.0 --port 5000 app:app
   ```

   detech session so it runs independently:

   ctrl+a, ctrl+d

   KIlling the AWS EC2 session:
   ```bash
   screen -X -S 47888.A2J kill
   ```

   kill process:
   ```bash
   sudo lsof -i :5000
   sudo kill -9 <process_id>
   ```

You should now be able to access the app at [http://localhost:5000](http://localhost:5000)! For the full context behind this example app, check out the [tutorial](https://beta.openai.com/docs/quickstart).
