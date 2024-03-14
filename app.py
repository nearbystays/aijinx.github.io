from flask import Flask, render_template
import os

app = Flask(__name__)

# Get a list of all HTML files in the templates directory
template_dir = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), 'templates')
html_files = [f for f in os.listdir(template_dir) if f.endswith('.html')]


@app.route('/')
def home():
    # Render a menu with links to all HTML files
    return render_template('index.html', html_files=html_files)


@app.route('/<name>')
def render_template_file(name):
    # Render the requested HTML file
    if name in html_files:
        return render_template(name)
    else:
        return "File not found", 404


if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=8080)
