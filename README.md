# A&D-Fusion
The A&D Fusion Career Architect project represents a technological solution in response to the challenges of the labor market in the Republic of Moldova. This AI-based platform was created to halt the phenomena affecting the country's economy and people's lives, such as massive migration, unemployment, and the acute shortage of skilled workers. By utilizing artificial intelligence, the platform successfully matches user skills with available job offers, facilitating the recruitment and professional orientation process.

The platform combats migration by highlighting local opportunities that match each user's profile, giving young people concrete reasons to build a future at home instead of seeking alternatives abroad due to a lack of information. At the same time, unemployment is significantly reduced through a high-precision matching system that eliminates human error and time lost in simple recruitment processes, ensuring rapid and convenient employment.
Furthermore, the specialist deficit is addressed using a vast dataset of over 100,000 job descriptions (we sourced the raw dataset from Kaggle: link, then sampled it down to 100,000 entries to maintain high data quality and computational efficiency) , which allows the system to accurately identify job offers where the user's specific skills are required, directing the workforce exactly where the market needs it most. The model analyzes complex job descriptions and guides users exactly in the directions demanded by the market. Thus, A&D Fusion Career Architect transforms a fragmented labor market into a clear one destined for everyone, which not only retains talent in the Republic of Moldova but also directs it to ensure the economic progress of the entire society.Furthermore, aligning salaries with the 1–5 qualification hierarchy provided a logical economic structure, ensuring the model outputs predictable and realistic compensation trends.

  Images:
  
We chose TruncatedSVD with 15 components and a technical weight of 20 to prioritize generalization over raw training accuracy. While other parameters showed higher scores, they risked overfitting by capturing noise within the dataset. Reducing the components to 15 forces the model to focus on the most significant job market patterns, acting as a regularizer. Similarly, the specific weighting ensures technical skills drive the matching logic without overshadowing other critical factors like location. This balanced approach prevents the model from simply memorizing the training data, ensuring more reliable and realistic recommendations for new users. We evaluated the model using a heatmap-based correlation analysis and confusion matrices to visualize how effectively it clusters similar roles. This verified that the model maintains high precision in matching user profiles to the most relevant job categories without overfitting:
<img width="533" height="338" alt="Screenshot 2026-05-05 154315" src="https://github.com/user-attachments/assets/1a123619-e6b3-4fa1-8bf0-f371748ea226" />
<img width="490" height="290" alt="Screenshot 2026-05-05 154348" src="https://github.com/user-attachments/assets/fc969574-2404-4b35-b47c-979fbbe9ed31" />

A&D Fusion description in website and the place when the users can put their API keys:
<img width="248" height="883" alt="Screenshot 2026-05-05 153153" src="https://github.com/user-attachments/assets/67d74e81-2e2c-435a-9ded-8afb31c6a9e6" />

"This initial section of the platform allows users to input their professional preferences, skills, and desired benefits. The collected data serves as the foundation for the AI engine to filter and generate personalized career recommendations. By structuring these user attributes, the system ensures high-accuracy matching between candidates and potential opportunities:
<img width="1919" height="907" alt="Screenshot 2026-05-04 231613" src="https://github.com/user-attachments/assets/62e51da7-137c-49b7-8ee0-513b4cff3104" />
<img width="1910" height="895" alt="Screenshot 2026-05-04 234209" src="https://github.com/user-attachments/assets/d890ce83-b2e7-43da-b058-0ab392c42cb1" />
<img width="1917" height="899" alt="image" src="https://github.com/user-attachments/assets/f47a6413-c4aa-4014-bcc8-5cbf3152647b" />

This section serves as the primary dataset containing structured information on skills and domains. The AI engine processes these specific data points to identify patterns and ensure an accurate match between the user's profile and the most relevant opportunities:
<img width="1644" height="905" alt="Screenshot 2026-05-05 153311" src="https://github.com/user-attachments/assets/b44e3ef7-a81d-4726-8efc-e6b479343d00" />

This final page allows users to input their CV as text, enabling the AI to analyze their professional background in depth. The system then generates tailored job matches and provides actionable recommendations for future career improvements. By comparing the CV against market demands, the AI offers a strategic roadmap for professional growth:
<img width="1549" height="862" alt="Screenshot 2026-05-05 213149" src="https://github.com/user-attachments/assets/2f72a3ce-b5d9-4e35-acdd-7f8bc55d2eda" />
<img width="1849" height="894" alt="Screenshot 2026-05-05 213136" src="https://github.com/user-attachments/assets/e5515cc9-c208-4a4a-b087-b06be7077efa" />
<img width="1910" height="906" alt="Screenshot 2026-05-05 213117" src="https://github.com/user-attachments/assets/9889a627-2888-4dca-ac71-b366fd6beaf7" />

This section displays the personalized results generated from the user's initial inputs, including their skills, experience, and professional preferences. The AI processes these specific attributes to rank and present the most compatible job opportunities with high precision. Additionally, users have the flexibility to customize the output by selecting exactly how many top results they wish to view:
<img width="1468" height="838" alt="Screenshot 2026-05-05 153431" src="https://github.com/user-attachments/assets/e91a633c-9ecc-4ea6-bf48-62281608a0dd" />
<img width="1453" height="836" alt="Screenshot 2026-05-05 153800" src="https://github.com/user-attachments/assets/94b5d874-7095-402f-b1d2-090e879fb9f2" />



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



Why we chose KNN for our job recommendation model: We chose unsupervised K-Nearest Neighbors because it operates on "look-alike" logic, which is more effective for recommendations than standard classification. Unlike "black-box" models, KNN identifies existing data points most similar to the user, providing high interpretability. It was specifically selected over supervised alternatives because it allows for direct retrieval of multiple similar roles without the overhead of a lengthy training process.

Why we choose all Alpha:
Owl Alpha was selected as the core engine for our third tab primarily due to its massive 1-million-token context window, which allows for the seamless integration of an entire 5,000-entry dataset into a single prompt. This eliminates the need for complex retrieval systems like RAG and prevents the loss of vital information that occurs when splitting data into smaller fragments. By keeping the a decent chunk of our Moldovan job market context "in-memory," the model can perform highly accurate cross-references between cities and salary benchmarks while strictly adhering to complex instructions like the STAR method for CV optimization. Choosing this model via OpenRouter ensures a high-performance, cost-effective solution that prioritizes data integrity and localized precision without the technical overhead of traditional vector databases.

Model Architecture:
The architecture utilizes a pipeline that reduces dimensionality via TruncatedSVD (15 components) and applies a 20x weight to technical skills to prioritize core competencies.

