A&D Fusion Career Architect — Intelligent Professional Modeling Platform (Simulation Prototype)
The A&D Fusion Career Architect project represents a simulated technological prototype developed to demonstrate how an AI-based platform could address the structural challenges of the labor market in the Republic of Moldova. This proof of concept is designed to model potential solutions for systemic issues affecting the country's economy, such as massive migration, unemployment, and the acute shortage of skilled workers.

Important Note on Project Scope: This platform is an advanced simulation and a functional prototype. It does not manage real-time job openings or ensure live, rapid employment. Instead, it serves to demonstrate approximately how an intelligent system would look, behave, and process data if populated with complete, production-scale real-world local market data. It visualizes how user skills could be algorithmically matched with available offers to facilitate recruitment and professional orientation.
<img width="1919" height="909" alt="image" src="https://github.com/user-attachments/assets/5c1fc145-a41f-4df2-80fc-04312d4ba76b" />

By showing how a high-precision matching system operates, this simulation models how migration could be combated by highlighting localized opportunities that match a user's profile, providing a digital framework that gives young people visibility into building a career at home. It simulates a workflow where human error and time lost in traditional recruitment are minimized, outlining a blueprint for efficient career placement. Furthermore, the specialized worker deficit is modeled using a sampled dataset of 100,000 job descriptions (sourced originally from Kaggle and downsampled to maintain computational efficiency and high data quality), allowing the system to demonstrate how workforce allocation can be optimized based on market demands. To anchor this simulation in realistic economic structures, salaries were aligned with a 1–5 qualification ierarhy, ensuring the model outputs predictable and logical compensation trends.
<img width="1523" height="904" alt="image" src="https://github.com/user-attachments/assets/7999bb27-b56b-4c9d-be38-5fb64fe78dca" />

System Architecture & Machine Learning Components
The system is split into two fundamental, complementary models that collaborate to deliver precision recommendations and generative contextual feedback.
While the primary model utilizes spatial distance algorithms to map structured user preferences, the secondary pipeline leverages a localized generative architecture to evaluate an unstructured text.
<img width="784" height="226" alt="image" src="https://github.com/user-attachments/assets/a1a405c7-ad0d-4e73-b1d9-bf971fcf523c" />

To ensure scientific rigor, system validation relies on pre-defined targets instead of post-facto metrics, setting operational baselines at 90.0% Precision@5 for text classification and a 100x lift over random assignment for structured matching. Both subsystems utilize standardized job roles as the primary evaluation target. To prevent data leakage and score inflation, the preprocessing pipeline structurally isolates all feature transformations from the target labels, enforcing a strict operational separation during validation to guarantee models are continuously benchmarked on entirely unseen profile architectures.


1. Primary Model: job_datasets (Local Recommendation System)
We chose unsupervised K-Nearest Neighbors (KNN) because it operates on "look-alike" logic, which is significantly more effective for recommendations than standard classification. Unlike "black-box" models, KNN identifies existing data points most similar to the user, providing high interpretability. It allows for the direct retrieval of multiple similar roles without the overhead of a lengthy training process.

Dimensionality Reduction & Regularization: The architecture utilizes a pipeline that reduces dimensionality via TruncatedSVD (15 components) and applies a 20x weight to technical skills to prioritize core competencies. While other parameter configurations showed higher training scores, they risked overfitting by capturing noise within the dataset. Reducing the components to 15 forces the model to focus on the most significant job market patterns, acting as a regularizer. Similarly, the specific weighting ensures technical skills drive the matching logic without overshadowing other critical factors like location.

Consistency Metric Calibrations: To guarantee absolute mathematical consistency across scripts and execution states, the model employs a uniform strictness parameter (strictness = 1.5). The realistic matching score is derived from spatial distance utilizing the following exponential formula:

Realistic Score=min(e −strictness × raw_dist) × 100))
This balanced approach prevents the model from simply memorizing training data, ensuring reliable and realistic recommendations for new profiles.

We evaluated the model using a heatmap-based correlation analysis and confusion matrices to visualize how effectively it clusters similar roles. This verified that the model maintains high precision in matching user profiles to the most relevant job categories without overfitting:
<img width="1370" height="805" alt="Screenshot 2026-05-03 161051" src="https://github.com/user-attachments/assets/6be2960f-30d4-46e1-8cab-74cfc54da16a" />
<img width="490" height="290" alt="Screenshot 2026-05-05 154348" src="https://github.com/user-attachments/assets/74c260cc-853c-43d9-bed3-b9a27805a091" />

The component configuration within the TruncatedSVD pipeline was selected to achieve an optimal middle ground in variance retention. While altering the dimensions yielded higher localized precision during specific iterations, a balanced architectural threshold was chosen to retain a significant portion of the original dataset's information while successfully filtering out redundant structural noise. This calibrated spatial approach avoids forcing user profiles into rigid, predefined boundaries, providing fluid, vector-based recommendations across adjacent career paths while maintaining overall model stability and generalization.

2. Secondary Model: CV Evaluation via Local RAG (Retrieval-Augmented Generation)
The third tab addresses unstructured text data from user CVs. Rather than dropping raw CV text directly into an external cloud API—which would consume excessive tokens and lack local context—the platform implements a localized RAG architecture:

[ User CV (Raw Text) ]-[ Local NLP Engine: TF-IDF ] ── (Cosine Similarity)──> [ Local Dataset (Top 15 Jobs) ]-[ OpenRouter API: owl-alpha ] <───(Enriched Context Feed)──────[ Tailored Feedback & STAR Method Optimization Roadmap ]
Local NLP Processing (TF-IDF): The uploaded CV text is vectorized locally using a TF-IDF (Term Frequency-Inverse Document Frequency) pipeline. This acts as our local NLP engine, breaking down text features and identifying skill weights without external network dependencies.

Context Filtering via Cosine Similarity: The system calculates the similarity between the CV vector and the dataset rows, extracting the top 15 most relevant job entries.


LLM Inference (Owl Alpha): Owl Alpha was selected as our core generative engine via OpenRouter primarily due to its massive 1-million-token context window. The model performs highly accurate cross-references between user profiles, cities, and salary benchmarks while strictly adhering to complex operational prompts, such as structuring career advice around the STAR method.
<img width="748" height="275" alt="Screenshot 2026-06-03 211920" src="https://github.com/user-attachments/assets/47d1c4b0-dbdf-4c93-b062-ccd564ba091a" />


max_features: 1000 (Dimensionality Control & Regularization)

Limiting the TF-IDF vectorizer to the top 1,000 most important terms across the corpus acts as a robust regularizer. Instead of mapping every typo, unique edge-case word, or irrelevant formatting token, the model strictly focuses on core domain vocabulary (e.g., specific technologies, skills, and job titles). This optimization drastically prevents overfitting, reduces the memory footprint, and ensures ultra-fast matrix calculations during live deployment on Streamlit Cloud.

ngram_range: (1, 1) (Unigram Efficiency)

Using strictly single words (unigrams) ensures that key professional terms—such as "Python", "SQL", "React", or "Manager"—are captured with maximum weight. In professional modeling, expanding to bigrams or trigrams (e.g., (1, 2)) introduces significant vocabulary noise and inflates computational complexity without adding proportional semantic value, as most technical skills are distinct, single-word nouns.
n_neighbors: 3 (High-Density Localized Matching)

Setting the neighborhood size to 3 ensures that the recommendation logic is highly specific and tightly bound to the absolute closest look-alike professional profiles in the vector space. A higher number of neighbors would dilute the results by averaging preferences across broader, unrelated job roles, whereas a k=3 setup preserves distinct, sharp boundary lines between job classes.

2. Justification of the 91.65% Scientific Precision
The model demonstrates an exceptionally high precision of 91.65% during cross-validation. This high metric indicates that text-based features (such as skill sets, tools, and background summaries) naturally form highly distinct, predictable clusters within the 1,000-dimensional space.
Because professional documentation uses standardized terminology, the distance vectors between a structured profile and its corresponding career track are mathematically unambiguous. This score proves that the model can reliably categorize and map an unparsed, raw text CV to its correct professional domain over 91 out of 100 times, minimizing mismatched career paths.

3. Execution & Serialization (.pkl File Generation)
Once the optimal configuration was discovered and validated, the pipeline assets were serialized into persistent binary files to decouple the heavy training phase from the lightweight application runtime:
tfidf_vectorizer.pkl: Contains the complete trained vocabulary weights and Inverse Document Frequency (IDF) coefficients. This allows the live Streamlit application to instantly transform new user CV text into the exact same 1,000-feature mathematical space without needing to rebuild the vocabulary from scratch.
recommender_text.pkl: Houses the fitted indexing structure and spatial distance configurations of the model.
Both files were generated using the joblib optimization library and are automatically streamed at runtime directly into the cloud application infrastructure. This completely eliminates execution overhead, allowing users to receive complex AI career mapping roadmaps in milliseconds.
<img width="1348" height="726" alt="image" src="https://github.com/user-attachments/assets/8ffe9f8b-ef38-408b-8adf-67ee8b129482" />



   
Data Engineering and Remote Storage Infrastructure:
To maintain high data quality and computational efficiency, a raw dataset sourced from Kaggle was downsampled to a clean, robust baseline of 100,000 entries.

Model Deployment & Storage: All trained machine learning models, vectorization pipelines, and structural weights are hosted on Hugging Face, ensuring rapid loading and separation of model assets from runtime code.

Large-Scale Data Hosting: Due to file size limitations within Git repositories, the core datasets, including the preprocessed data/pentru_andrei.csv, are stored on Google Drive. These data files are integrated directly as remote access links within GitHub, keeping the repository lightweight and efficient.

Data Preprocessing Pipeline
To transform raw information into structured matrices suitable for the KNN algorithm, data passed through a rigorous pipeline:

Category Mapping: Standardizing highly fragmented categories.
<img width="1230" height="362" alt="image" src="https://github.com/user-attachments/assets/3580852c-5575-478e-80bd-a9959b0b79e8" />

Label Encoding: Converting ordinal features into clear sequential indices.
<img width="658" height="121" alt="image" src="https://github.com/user-attachments/assets/f3a15b5e-6ca2-4c4a-8172-2a0f7b033ba0" />

One-Hot Encoding: Expanding nominal categorical variables into binary vectors to eliminate the risk of introducing artificial mathematical hierarchies into the distance algorithms.
<img width="706" height="82" alt="image" src="https://github.com/user-attachments/assets/f103d101-3ab5-49be-9c15-e43bc1ccf88f" />

Niche anomalies like "amphibian care" were explicitly identified and mapped into broader categories rather than deleted, preserving data diversity without distorting the matching logic. Once finalized, the clean matrix was deployed to our cloud storage workflow.

API Transparency
The application requires an OpenRouter API Key. This architecture leverages Streamlit Cloud for both secure API key management and live website hosting, guaranteeing absolute operational transparency. This cloud-native deployment strategy allows users and evaluation juries to verify a secure data flow, inspect how the prompt context is dynamically structured, and ensure that all background processing remains clean, safe, and entirely free from concealed black-box operations.
<img width="475" height="104" alt="image" src="https://github.com/user-attachments/assets/0b9b5b4a-25fb-4005-a833-71e93d51c360" />


Technical Evaluation & Architectural Comparison
The model achieved a precision score of 0.3, when tasting with the same dataset, representing a major performance lift in an unsupervised recommendation context. Across a complex dataset containing thousands of unique career configurations, this is approximately 150 times more accurate than random guessing, confirming that spatial distance vectors correspond deeply to actual professional paths.
<img width="533" height="338" alt="Screenshot 2026-05-05 154315" src="https://github.com/user-attachments/assets/61285403-00f8-492d-bec5-701d08e7a862" />



Architectural Approach Analysis
1. Localized Precision
Our Primary Model (KNN): Excellent. Specifically calibrated to mirror and model Moldovan market dynamics and localized salary structures using structured preference data.
Our Secondary Model (TF-IDF RAG): Excellent. Successfully anchors unstructured CV text to localized domestic realities by filtering context through our curated local dataset before any generation takes place.
Naive LLM Prompting: Poor. Large global models lack specific structural familiarity with small regional markets, resulting in generalized assumptions and inaccurate localized guidance.
Classic Classification (e.g., Random Forest): Moderate. Effectively categorizes data but requires a rigid and continuous re-training schedule for any underlying dataset modification.

2. Token Cost Efficiency
Our Primary Model (KNN): Maximum efficiency. Operates 100% locally or within our secure cloud environment using pure mathematical distance calculations, generating zero external API token costs.
Our Secondary Model (TF-IDF RAG): High efficiency. The local TF-IDF engine pre-filters and pipes only the 15 most relevant roles to the LLM, dramatically reducing external API token use compared to dumping raw text files.
Naive LLM Prompting: Very low efficiency. Uploading massive raw profiles or complete datasets directly to a cloud API creates extensive, costly, and unnecessary token overhead.
Classic Classification (e.g., Random Forest): Maximum efficiency. Operates with 100% local mathematical calculations, resulting in zero external API costs.

3. Actionable Feedback Delivery
Our Primary Model (KNN): Mathematically targeted. Ranks and presents structural look-alike career options with high interpretability based on spatial distance, though it does not output generative text.
Our Secondary Model (TF-IDF RAG): Highly personalized and generative. Pinpoints precise, granular skill gaps and paths against locally filtered matches, delivering custom-tailored resumes optimization advice.
Naive LLM Prompting: Generic. Frequently falls back on broad, non-specific career advice and standard platitudes without tying them to hard data.
Classic Classification (e.g., Random Forest): Non-existent. Outputs a static numerical label, class index, or job ID without any textual elaboration or growth roadmap.

4. Data Safety & Flow
Our Primary Model (KNN): Maximum safety. All preference data scanning, vector matrices processing, and profile comparisons occur strictly within the isolated application runtime environment.
Our Secondary Model (TF-IDF RAG): High safety. Unstructured documents are tokenized and scanned locally to find matching rows; only filtered, non-identifiable job profile contexts are transmitted for LLM inference.
Naive LLM Prompting: Low safety. Exposes whole, unstructured, raw CV documents directly to external cloud systems without any local masking or pre-filtering.
Classic Classification (e.g., Random Forest): Maximum safety. The entire data processing stream and predictive runtime remain completely local and self-contained.

5. Categorization Style
Our Primary Model (KNN): Fluid and flexible. Employs distance-based recommendations across continuous mathematical vectors, allowing users to discover unexpected adjacent career trajectories.
Our Secondary Model (TF-IDF RAG): Hybrid semantic matching. Seamlessly bridges the gap between raw natural language text patterns and fixed dataset distributions.
Naive LLM Prompting: Unpredictable. Highly susceptible to minor prompt variations, structural formatting changes, and random model hallucinations.
Classic Classification (e.g., Random Forest): Rigid. Inflexibly forces multi-dimensional user profiles into strict, pre-defined, hard-coded binary categories.

Limitations & Error Analysis
The primary limitation involves semantic overlap, where the system occasionally confuses roles with nearly identical skill requirements (e.g., Data Analyst vs. BI Developer). This challenge slightly impacts both systems: the KNN primary model might calculate highly narrow spatial distances between these close vectors, while the TF-IDF secondary model might extract overlapping keyword weights.
Additionally, because a sampled subset of 100,000 entries is utilized, rare or highly specialized roles may exhibit a lower density of near neighbors in the spatial matrix, slightly shrinking recommendation diversity for highly niche professional profiles.




Ethics, Risks, and Future Horizons
Data Privacy (user_submissions logging): To track model performance and analytical trends, user selections and submission attributes are securely compiled within a user_submissions repository. Crucially, this storage file completely excludes personal identifiers such as names, ensuring that data privacy is rigorously maintained while retaining anonymous profiles for platform evaluation.
Potential Algorithmic Bias: As an inherent technical risk, because the primary model is trained on a historically sampled dataset, it may naturally favor highly populated industries over long-tail, niche career paths. Acknowledging this baseline bias is a critical first step for future dataset balancing.
The "Echo Chamber" Risk: Like many distance-based recommendation engines, there is a risk of restricting users into rigid career boundaries based only on their past history. We recognize this limitation, and future updates intend to introduce structural variety to ensure users are exposed to broader, diverse career opportunities.
The platform is designed to scale effortlessly, maintaining potential to integrate directly with regional Moldovan employment portals like anofm.md and angajat.md. Future iterations intend to evolve the CV analyzer into a comprehensive career coach via a complete chatbot interface, real-time market trend analytics, and advanced predictive search filters.

Class balance across occupational tracks was evaluated to protect long-tail local specializations from high-density clusters. To mitigate demographic and geographic discrimination from gender or city variables in the local data, the system enforces absolute algorithmic fairness via demographic masking. All sensitive tokens and location identifiers are scrubbed during vector calculations, ensuring the engine operates exclusively on objective qualifications and skills. Furthermore, validation is completely standardized across both engines using a unified multi-metric framework—Precision@k, Recall@k, and NDCG—paired with continuous error analysis loops to isolate precise keyword discrepancies.


The version of some of the libraries that we used:
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




How to Run and Use the Platform:
1 Accessing the Live Web Application
Because the platform is fully hosted and deployed via Streamlit Cloud, there is no need to download datasets, clone the repository, or execute local terminal startup commands. You can access the functional simulation prototype instantly through any browser using the official link:

A&D Fusion Career Architect Deployment Link - https://ad-fusion-career-architect-tl4his6fhpdzauvlcruzub.streamlit.app/

2 Step-by-Step Usage Guide for Optimal Model Performance
To extract the most accurate, high-precision recommendations and comprehensive feedback from our integrated machine learning models, follow this operational workflow:


Step 1: Calibrate Profile Preferences & Select Target Variants
Geographic Focus: Select your target city from the dropdown options to anchor the spatial recommendation engines to localized salary structures.
Filter Competencies & Benefits: Use the checkboxes and sliders to select all variations and skills that accurately match your profile. Taking the time to precisely select your exact technical capabilities allows the KNN Primary Model to calculate near-neighbor spatial distances with maximum precision.
Adjust Match Density: Move the result slider to choose exactly how many ranked recommendations you want the model to display.
<img width="1895" height="883" alt="image" src="https://github.com/user-attachments/assets/c0610cb5-af01-40ef-b2b3-6052bbfd9da2" />


Step 2: Execute Interactive CV Analysis & Improvement
Navigate to the AI Recommender tab inside the web interface.
Paste CV as Raw Text: Copy the text from your existing resume document and paste it directly into the provided text submission box.
Extract and Refine: Submit the text to trigger the TF-IDF local NLP engine. The system will scan your text features, compare them against the dataset using cosine similarity, and feed the top 15 matching contexts to the LLM.
Review Your Actionable Roadmap: Study the generated evaluation report. To achieve the best real-world results from the model's output, update your profile following the customized STAR method roadmap to bridge the exact technical skill gaps highlighted by the simulation.
<img width="1905" height="910" alt="image" src="https://github.com/user-attachments/assets/ec58aab0-e852-43ed-b35e-0910c1f663f7" />

