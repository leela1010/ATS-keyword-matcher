import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string

nltk.download('stopwords')
from nltk.corpus import stopwords

# Read file
def load_file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return f.read().lower()

resume_text = load_file('resume.txt')
jd_text = load_file('job_description.txt')

# Clean text
def clean_text(text):
    stop_words = set(stopwords.words('english'))
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = [word for word in text.split() if word not in stop_words]
    return words

resume_words = clean_text(resume_text)
jd_words = clean_text(jd_text)

# For similarity
resume_clean = " ".join(resume_words)
jd_clean = " ".join(jd_words)

# Cosine similarity
vectorizer = CountVectorizer().fit_transform([resume_clean, jd_clean])
similarity = cosine_similarity(vectorizer[0:1], vectorizer[1:2])
match_percentage = round(float(similarity[0][0]) * 100, 2)

# Matching keywords
resume_set = set(resume_words)
jd_set = set(jd_words)

matched_keywords = resume_set & jd_set
missing_keywords = jd_set - resume_set

# Output
print(f"\n🔍 Resume matches Job Description by: {match_percentage}%")
print("\n✅ Matched Keywords:")
print(", ".join(sorted(matched_keywords)) or "None")

print("\n❌ Missing Keywords from Resume:")
print(", ".join(sorted(missing_keywords)) or "None")
