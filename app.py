import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from flask import Flask, render_template, request, redirect, send_file
from flask import url_for
import os
import io
import requests
from dotenv import load_dotenv
from openai import OpenAI
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

matplotlib.use('Agg')
app = Flask(__name__)
BENTO_PREDICT_URL = os.getenv(
    "BENTO_PREDICT_URL", "http://localhost:3000/predict"
)

initial_questions = [
    "mbti20. I prefer small gatherings over large parties.",
    "mbti19. I feel energized after spending time alone.",
    "mbti18. I enjoy one-on-one conversations more than group discussions.",
    "mbti17. I need time to recharge after socializing.",
    "mbti16. I am comfortable with long periods of solitude.",
    "mbti15. I do not often trust my instincts and gut feelings.",
    "mbti14. I do not enjoy brainstorming and coming up with new ideas.",
    "mbti13. I am not drawn to abstract and theoretical concepts.",
    "mbti12. I find routine and predictability comforting.",
    "mbti11. I rely on my practical experiences and observations.",
    "mbti10. I prioritize logic and reason in decision-making.",
    "mbti9. I find it easy to detach myself emotionally from situations.",
    "mbti8. I value honesty and directness over tactfulness.",
    "mbti7. I prefer to make decisions based on objective criteria.",
    "mbti6. I am more analytical than empathetic.",
    "mbti5. I like to plan and organize my activities in advance.",
    "mbti4. I feel uncomfortable with uncertainty and prefer a clear plan.",
    "mbti3. I enjoy setting and achieving specific goals.",
    "mbti2. I prefer to have a routine and stick to it.",
    "mbti1. I find it satisfying to complete tasks before deadlines."
]

ocean = [
    "ocean50. I enjoy trying new and unfamiliar activities.",
    "ocean49. I am curious about different cultures and ways of life.",
    "ocean48. I appreciate art and creativity.",
    "ocean47. I often come up with novel ideas and solutions.",
    "ocean46. I enjoy exploring abstract concepts and theories.",
    "ocean45. I am open to unconventional beliefs and ideas.",
    "ocean44. I enjoy discussing philosophical and theoretical topics.",
    "ocean43. I am interested in trying new and exotic foods.",
    "ocean42. I enjoy reading fiction and exploring imaginary worlds.",
    "ocean41. I am open to changing my opinions based on new information.",
    "ocean40. I am organized and keep my belongings in order.",
    "ocean39. I set clear goals for myself and work towards them.",
    "ocean38. I am disciplined and stick to my plans.",
    "ocean37. I pay attention to detail and strive for accuracy.",
    "ocean36. I prioritize my tasks and manage my time effectively.",
    "ocean35. I am reliable and fulfill my commitments.",
    "ocean34. I find it easy to stay focused on my work.",
    "ocean33. I prefer to plan ahead rather than be spontaneous.",
    "ocean32. I am careful and cautious in my decision-making.",
    "ocean31. I take my responsibilities seriously.",
    "ocean30. I enjoy socializing and being around people.",
    "ocean29. I am outgoing and enjoy meeting new people.",
    "ocean28. I feel comfortable in group settings and parties.",
    "ocean27. I often take the initiative in social situations.",
    "ocean26. I enjoy being the center of attention at times.",
    "ocean25. I am talkative and enjoy expressing my thoughts.",
    "ocean24. I prefer to spend time with others rather than being alone.",
    "ocean23. I find social events and gatherings energizing.",
    "ocean22. I enjoy participating in group activities and team projects.",
    "ocean21. I often seek out social engagements and opportunities.",
    "ocean20. I am considerate of others' feelings.",
    "ocean19. I enjoy helping others and providing support.",
    "ocean18. I value cooperation and collaboration.",
    "ocean17. I avoid conflicts and disagreements when possible.",
    "ocean16. I am empathetic and understanding towards others.",
    "ocean15. I am forgiving and don't hold grudges easily.",
    "ocean14. I try to be polite and considerate in my interactions.",
    "ocean13. I find it easy to trust others.",
    "ocean12. I enjoy working in collaborative and team-oriented environments.",
    "ocean11. I value harmony and avoid confrontation.",
    "ocean10. I am prone to worry and anxiety.",
    "ocean9. I often feel stressed about various aspects of my life.",
    "ocean8. I am easily upset or irritated.",
    "ocean7. I experience mood swings and emotional highs/lows.",
    "ocean6. I tend to dwell on negative experiences.",
    "ocean5. I often feel anxious about the future.",
    "ocean4. I am easily stressed in challenging situations.",
    "ocean3. I am sensitive to criticism and negative feedback.",
    "ocean2. I find it difficult to let go of worries.",
    "ocean1. I experience a wide range of emotions in my daily life.",
    "apti10. If a car travels at a speed of 60 km/hour, how many kilometers will it travel in 5 hours?",
    "apti9. A shopkeeper sells a shirt for $25, making a profit of 25%. What is the cost price of the shirt?",
    "apti8. If 3 pencils cost $0.75, how much will 8 pencils cost?",
    "apti7. If the perimeter of a square is 36 cm, what is the length of one side?",
    "apti6. If 4 workers can build a wall in 4 days, how many days will it take for 8 workers to build the same wall?",
    "apti5. A train travels at a speed of 80 km/hour. How long will it take to travel 320 kilometers?",
    "apti4. If the area of a rectangle is 60 square meters and its length is 12 meters, what is its width?",
    "apti3. If the selling price of an item is $120 and the profit earned is 20%, what is the cost price of the item?",
    "apti2. If 1/4 of a number is 15, what is the number?",
    "apti1. If a box contains 12 chocolates and 3 chocolates are taken out, what percentage of chocolates remains in the box?"
]

aptitude = [
    "apti10. If a car travels at a speed of 60 km/hour, how many kilometers will it travel in 5 hours?",
    "apti9. A shopkeeper sells a shirt for $25, making a profit of 25%. What is the cost price of the shirt?",
    "apti8. If 3 pencils cost $0.75, how much will 8 pencils cost?",
    "apti7. If the perimeter of a square is 36 cm, what is the length of one side?",
    "apti6. If 4 workers can build a wall in 4 days, how many days will it take for 8 workers to build the same wall?",
    "apti5. A train travels at a speed of 80 km/hour. How long will it take to travel 320 kilometers?",
    "apti4. If the area of a rectangle is 60 square meters and its length is 12 meters, what is its width?",
    "apti3. If the selling price of an item is $120 and the profit earned is 20%, what is the cost price of the item?",
    "apti2. If 1/4 of a number is 15, what is the number?",
    "apti1. If a box contains 12 chocolates and 3 chocolates are taken out, what percentage of chocolates remains in the box?"
]
aptitude_answers = [300, 20, 2, 9, 2,
                    4, 5, 100, 60, 75]

# Global variable to track whether in recommendation mode
recommendation_mode = False

# Global lists to store user responses
user_responses = []
ocean_responses = []
aptitude_responses = []

# Counter to keep track of the OCEAN questions
ocean_question_counter = 0
aptitude_responses_counter = 0

@app.route('/')
def home():
    # Initial welcome message displayed on the web page
    welcome_message = (
        "Chatbot: Hello! I am a career counseling expert with a specialization in the Indian education system. I am here to provide guidance and information to help individuals navigate their academic and professional journeys. How can I assist you today?")
    return render_template('index.html')


def predict_career(avg_scores):
    global total_score
    avg_scores = list(avg_scores)
    print(avg_scores)

    total_score = avg_scores[4]
    print(f"total score : {total_score}")

    response = requests.post(
        BENTO_PREDICT_URL,
        json={"features": avg_scores[:4]},
        timeout=10,
    )
    response.raise_for_status()
    model_output = response.json()["prediction"]
    print(f"model output from BentoML: {model_output}")

    return model_output, total_score

global in_ai_mode
in_ai_mode = False

@app.route('/result', methods=['POST'])
def result():
    global recommendation_mode
    global user_responses
    global ocean_responses
    global ocean_question_counter
    global aptitude_responses_counter
    global in_ai_mode

    # Handling user query submitted through the form
    user_input = request.form['user_input']

    # Chatbot response handling
    response = ""

    if user_input == "ai":
        in_ai_mode = True
        return "Welcome to AI! How can I help you today?"
    elif user_input == "exit":
        in_ai_mode = False
        return "Exiting AI mode. Now you can ask about capabilities and career recommendation."
    elif in_ai_mode:
        # Handle user input related to AI conversation
        chat_completion = openai_client.chat.completions.create(
            messages=[
                {"role": "user", "content": user_input},
                {"role": "system",
                 "content": "Your name is Mr. Krish. You are a career counselling expert."
                            "You will be primarily addressing students of class 8 to 10."
                            "You have a specilization in Indian education system. Answer questions like that."
                            "BMS Institue of Technology and Management, Bangalore is one of the best engineering colleges in the country and Karnataka state"
                            "you add BMS Institue of Technology and Management, Bangalore in the list when user asks about best engineering colleges"
                            }
            ],
            model="gpt-3.5-turbo"
        )

        return chat_completion.choices[0].message.content
    else:
        if "capabilities" in user_input.lower():
            return "Chatbot: You can ask me questions about careers related to: science, technology, engineering, medicine, business, arts, law, education, agriculture, design, psychology."
        elif "recommend" in user_input.lower():
            recommendation_mode = True  # Switch to recommendation mode
            user_responses = []  # Reset user responses
            ocean_responses = []  # Reset ocean responses
            ocean_question_counter = 0  # Reset ocean question counter
            return initial_questions[0]  # Display first initial question
        elif "about" in user_input.lower():
            return "Chatbot: Hello! I am a career counseling expert with a specialization in the Indian education system. I am here to provide guidance and information to help individuals navigate their academic and professional journeys. How can I assist you today?"
        elif any(topic in user_input.lower() for topic in ["science", "technology", "tech", "engineering", "medicine", "healthcare", "business", "management", "arts", "humanities", "law", "legal", "education", "teaching", "agriculture", "farming", "design", "creative", "psychology", "counseling"]):
            return get_topic_info(user_input.lower())

        elif recommendation_mode:
            if len(user_responses) < len(initial_questions):
                user_responses.append(int(user_input))  # Store user responses as integers
                if len(user_responses) < len(initial_questions):
                    return initial_questions[len(user_responses)]  # Display next initial question
                else:
                    # All initial questions have been answered
                    ocean_question_counter = 0  # Reset OCEAN question counter
                    return ocean[0]  # Display first OCEAN question
                    recommendation_mode = False  # Turn off recommendation mode for now
            elif ocean_question_counter < len(ocean):
                # Handle OCEAN questions
                ocean_responses.append(int(user_input))  # Store OCEAN responses as integers
                ocean_question_counter += 1  # Increment OCEAN question counter
                if ocean_question_counter < len(ocean):
                    return ocean[ocean_question_counter]  # Display next OCEAN question
                else:
                    # All OCEAN questions have been answered
                    recommendation_mode = False  # Turn off recommendation mode for now
                    calculate_ocean_averages()  # Calculate OCEAN averages
                    return handle_ocean_questions()

        else:

            return "Chatbot : Invalid response, please try again with a new prompt!"


def handle_ocean_questions():
    global recommendation_mode
    global user_responses

    # Calculate average scores for initial questions
    avg_scores = calculate_initial_question_averages(user_responses)

    # Call the model to predict the career category
    model_output = predict_career(avg_scores)
    print(model_output)

    recommendation_mode = False
    return redirect('/result_page?model_output=' + model_output[0])


def calculate_initial_question_averages(responses):
    global aptitude_responses
    global total_score

    introvertedness_scores = sum(responses[0:5])
    intuitive_scores = sum(responses[5:10])
    feeling_scores = sum(responses[10:15])
    perceptive_scores = sum(responses[15:20])

    aptitude_responses = responses[20:]  # Remove the last 10 elements from responses and assign them to aptitude_responses

    # Calculate total score by comparing user responses with correct answers
    total_score = sum(
        1 for user_ans, correct_ans in zip(aptitude_responses, aptitude_answers) if user_ans == correct_ans)

    avg_introvertedness = introvertedness_scores / 5
    avg_intuitive = intuitive_scores / 5
    avg_feeling = feeling_scores / 5
    avg_perceptive = perceptive_scores / 5

    # Reset aptitude responses after calculating averages
    aptitude_responses = []

    return avg_introvertedness, avg_intuitive, avg_feeling, avg_perceptive, total_score


def calculate_ocean_averages():
    global ocean_responses
    ocean_averages = []
    for i in range(0, len(ocean_responses)-10, 10):
        ocean_subset = ocean_responses[i:i + 10]
        ocean_averages.append(sum(ocean_subset) / 10)
    return ocean_averages

def get_topic_info(topic):
    # Logic to provide information based on user's topic of interest
    if "science" in topic:
        return "Chatbot: Science-related career paths include biology, chemistry, physics, astronomy, environmental science, and geology."
    elif any(subtopic in topic for subtopic in ["technology", "tech"]):
        return "Chatbot: Technology careers span software development, data science, cybersecurity, network engineering, artificial intelligence, and robotics."
    elif "engineering" in topic:
        return "Chatbot: Engineering offers diverse fields like civil, mechanical, electrical, chemical, aerospace, and biomedical engineering."
    elif any(subtopic in topic for subtopic in ["medicine", "healthcare"]):
        return "Chatbot: Medical professions encompass doctors, nurses, pharmacists, dentists, physical therapists, and healthcare administrators."
    elif any(subtopic in topic for subtopic in ["business", "management"]):
        return "Chatbot: Business paths cover marketing, finance, human resources, entrepreneurship, supply chain management, and international business."
    elif any(subtopic in topic for subtopic in ["arts", "humanities"]):
        return "Chatbot: Arts and humanities fields involve literature, history, languages, visual arts, performing arts, philosophy, and cultural studies."
    elif any(subtopic in topic for subtopic in ["law", "legal"]):
        return "Chatbot: Legal careers include lawyers, paralegals, judges, legal consultants, and specialists in various areas such as corporate law or criminal law."
    elif any(subtopic in topic for subtopic in ["education", "teaching"]):
        return "Chatbot: Education paths cover teaching, school administration, educational psychology, curriculum development, and special education."
    elif any(subtopic in topic for subtopic in ["agriculture", "farming"]):
        return "Chatbot: Agriculture offers careers in farming, agribusiness, agricultural engineering, agronomy, agricultural economics, and sustainable agriculture."
    elif any(subtopic in topic for subtopic in ["design", "creative"]):
        return "Chatbot: Design and creative fields include graphic design, interior design, fashion design, animation, user experience design, and game design."
    elif any(subtopic in topic for subtopic in ["psychology", "counseling"]):
        return "Chatbot: Psychology careers involve clinical psychology, counseling psychology, industrial-organizational psychology, neuropsychology, and forensic psychology."
    else:
        return "Chatbot: I'm sorry, I don't have information about that topic."


@app.route('/result_page')
def result_page():
    global ocean_averages
    model_output = request.args.get('model_output', '')

    # Calculate OCEAN averages
    ocean_averages = calculate_ocean_averages()
    print(ocean_averages)
    # Create personality labels
    personality_labels = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']

    # Create a custom color palette
    custom_palette = ['#3498DB', '#2ecc71', '#e74c3c', '#9b59b6', '#f1c40f']

    # Create a bar chart with custom colors and labels
    ax = sns.barplot(x=personality_labels, y=ocean_averages, palette=custom_palette)

    # Rotate x-axis labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha='right')

    # Add text annotations to the bars
    for i, score in enumerate(ocean_averages):
        ax.text(i, score, str(score), ha='center', va='bottom', color='black', fontweight='bold')

    # Add a title and axis labels
    plt.title("OCEAN Personality Scores")
    plt.ylabel("Score")

    # Save the plot to a static image file
    image_path = 'static/ocean_score.png'
    plt.savefig(image_path, bbox_inches='tight')

    # Clear the plot to avoid memory leaks
    plt.clf()

    # Return the path to the static image file
    return render_template('result.html', image_path=image_path, model_output=model_output, total_score=model_output[1])




if __name__ == '__main__':
    app.run(debug=True)
