from flask import Flask, render_template, request, redirect, url_for
from diffusers import DiffusionPipeline
import os

# Initialize Flask app
app = Flask(__name__, static_folder='static')

# Load the pre-trained comic generation model
# This pipeline generates an image from a given text prompt.
pipe = DiffusionPipeline.from_pretrained("ogkalu/Comic-Diffusion")

# Route for the home page
@app.route('/')
def home():
    return render_template('home.html')


# Route for the text input page where the user enters the text
@app.route('/enter_text', methods=['GET', 'POST'])
def enter_text():
    if request.method == 'POST':
        # Fetch the user input from the form
        prompt = request.form['text_prompt']
        
        # Redirect to the comic generation page with the user input
        return redirect(url_for('generate_comic', prompt=prompt))
    
    # Renders the text input page template
    return render_template('enter_text.html')

# Route for generating and displaying the comic based on user input
@app.route('/generate_comic')
def generate_comic():
    # Get the user input (prompt) from the URL query parameters
    prompt = request.args.get('prompt')
    
    # Generate comic image based on the input prompt using the pre-trained model
    image = pipe(prompt).images[0]
    
    # Save the generated image to the static folder
    image_path = os.path.join('static', 'images', 'generated_comic.png')
    image.save(image_path)
    
    # Pass the image path to the display_comic.html template for rendering
    return render_template('display_comic.html', image_path=image_path)

# Run the app when the script is executed
if __name__ == "__main__":
    # Set debug=True for development purposes to get detailed error logs
    app.run(debug=True)
