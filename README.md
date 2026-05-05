# A&D-Fusion
The A&D Fusion Career Architect project represents a technological solution in response to the challenges of the labor market in the Republic of Moldova. This AI-based platform was created to halt the phenomena affecting the country's economy and people's lives, such as massive migration, unemployment, and the acute shortage of skilled workers. By utilizing artificial intelligence, the platform successfully matches user skills with available job offers, facilitating the recruitment and professional orientation process.

The platform combats migration by highlighting local opportunities that match each user's profile, giving young people concrete reasons to build a future at home instead of seeking alternatives abroad due to a lack of information. At the same time, unemployment is significantly reduced through a high-precision matching system that eliminates human error and time lost in simple recruitment processes, ensuring rapid and convenient employment.
Furthermore, the specialist deficit is addressed using a vast dataset of over 100,000 job descriptions, which allows the system to accurately identify job offers where the user's specific skills are required, directing the workforce exactly where the market needs it most. The model analyzes complex job descriptions and guides users exactly in the directions demanded by the market. Thus, A&D Fusion Career Architect transforms a fragmented labor market into a clear one destined for everyone, which not only retains talent in the Republic of Moldova but also directs it to ensure the economic progress of the entire society.








Libraries and Versions:
Package                   Version
------------------------- -----------
joblib                    1.5.3
numpy                     2.4.4
openai                    2.33.0
pandas                    3.0.2
pip                       23.2.1
python-dateutil           2.9.0.post0
python-multipart          0.0.27
scikit-learn              1.8.0
streamlit                 1.57.0



How to use our platform:
1. Setup and Installation
Access the Source Code: Begin by downloading or cloning the project files from the official GitHub repository.
Install Dependencies: Open your terminal or command prompt in the project directory and install all necessary Python libraries (such as Pandas, Streamlit, and Scikit-learn) by running: pip install.
Configure API Access: To enable the AI features, you must provide your own free API key from OpenRouter within the application settings or environment configuration.

2. Running the Application
Launch the Platform: Start the local server by executing the following command: streamlit run .\main.py.
Company Overview: Once the website loads, you can read the detailed company description to understand the mission of A&D Fusion.

3. Personalized Job Evaluation
Select Your Preferences: Navigate through the interactive sidebar or main menu to select options that best fit your profile, including your preferred city of employment.
Skill and Benefit Filters: Input the number of professional skills you possess and select the specific benefits you desire from a workplace.
Result Customization: Choose how many potential job matches you wish to receive after the system evaluates your responses.

4. Data Exploration and AI CV Analysis
Dataset Visualization: You can directly view and interact with the underlying dataset used for professional competency analysis.
AI Recommender: Insert the text from your CV into the "AI Recommender" section.
Future Career Pathing: The AI will analyze your CV to provide specific employment variants and personalized recommendations for future skill development based on your unique professional history.


Why we choose all Alpha:
Owl Alpha was selected as the core engine for our third tab primarily due to its massive 1-million-token context window, which allows for the seamless integration of an entire 5,000-entry dataset into a single prompt. This eliminates the need for complex retrieval systems like RAG and prevents the loss of vital information that occurs when splitting data into smaller fragments. By keeping the a decent chunk of our Moldovan job market context "in-memory," the model can perform highly accurate cross-references between cities and salary benchmarks while strictly adhering to complex instructions like the STAR method for CV optimization. Choosing this model via OpenRouter ensures a high-performance, cost-effective solution that prioritizes data integrity and localized precision without the technical overhead of traditional vector databases.
