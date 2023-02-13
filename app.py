import os
import openai
from flask import Flask, redirect, render_template, request, url_for
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)
openai.api_key = os.getenv("API_KEY")

@app.route("/", methods=("GET", "POST"))



def index():
    categories = ["divorce", "adoption"]
    if request.method == "POST":
        issue = request.form["issue"]
        relevant_db = categorize_issue(issue, categories)
        if relevant_db is None:
            return redirect(url_for("index", result="Sorry, I do not possess the information that you are looking for yet. As of now, I am only able to answer questions in the area of family law."))
        with open("db\\" + relevant_db + ".txt", "r") as file_db, open("structure\\" + relevant_db + ".txt", "r") as file_st:
            db = file_db.read()
            st = file_st.read()
        prompt = generate_prompt(issue, db, st)
        start_sequence = "\nUser:"
        restart_sequence = "\n\nBob: "
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt= prompt,
            temperature=0,
            max_tokens=100,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=[" User:", " Bob:"]
        )
        return redirect(url_for("index", result=response.choices[0].text))

    result = request.args.get("result")
    return render_template("index.html", result=result)

def categorize_issue(issue, categories):
    prompt = f"Categorize the issue '{issue}' into one of the following categories, and return me with one category. Categories: {', '.join(categories)}."
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=100,
        n=1,
        temperature=0.9,
        top_p=1,
        presence_penalty=0.6
    )
    issue_category = response.choices[0].text.strip().lower()
    if issue_category in categories:
        return issue_category
    return None


def generate_prompt(issue, db, st):

    return """I am an AI assistant that answers legal questions strictly based on the database {0}. I will guide users on the legal procedures, and ask if the user has completed the required step, following the structure of {1}. I will try to answer the question according to the database. However, if I do not know the answer to the question, or if the question is too complicated, or if the question requires input from a lawyer, I will recommend the user to seek legal advice from lawyers, and direct them to 'https://www.probono.sg/'. 

issue: {2}
reply:
""".format(db, st, issue.capitalize())
