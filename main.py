import os
import sys
import csv
import json
from src.models.chatgpt_model import ChatGPTScorer
from src.config.scorer_config import ScorerConfig

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

def load_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def load_json_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            if not content.strip():
                print(f"Warning: File '{file_path}' is empty.")
                return {}
            return json.loads(content)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON in file '{file_path}': {str(e)}")
        print(f"File content: {content[:100]}...")  # Print first 100 characters
        return {}
    except Exception as e:
        print(f"Error reading file '{file_path}': {str(e)}")
        return {}

def load_cv_file(file_path):
    if file_path.endswith('.json'):
        json_data = load_json_file(file_path)
        if not json_data:
            print(f"Warning: Unable to load JSON data from '{file_path}'")
        return json.dumps(json_data)  # Convert back to string for consistency
    elif file_path.endswith('.txt'):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            print(f"Error reading text file '{file_path}': {str(e)}")
            return ""
    else:
        print(f"Unsupported file format: {file_path}")
        return ""

def load_job_descriptions(folder_path):
    job_descriptions = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            job_description = load_text_file(file_path)
            job_descriptions.append((filename, job_description))
    return job_descriptions

def save_results(results, output_file):
    with open(output_file, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['CV', 'Job Description', 'Model', 'Score', 'Top Skills'])
        for result in results:
            writer.writerow([
                result['cv'],
                result['job_description'],
                result['score'],
                ', '.join([f"{skill['compétence']} ({skill['score']:.2f})" for skill in result['top_skills']])
            ])

def main():
    config = ScorerConfig()

    # Load models
    model = ChatGPTScorer()

    # Load CV data
    cv_folder = 'data/cvs'
    cv_files = [f for f in os.listdir(cv_folder) if f.endswith('.txt') or f.endswith('.json')]

    # Load job descriptions
    job_descriptions_folder = 'data/job_descriptions'
    job_descriptions = load_job_descriptions(job_descriptions_folder)

    results = []

    for cv_file in cv_files:
        cv_path = os.path.join(cv_folder, cv_file)
        cv_text = load_cv_file(cv_path)

        for job_desc_file, job_description in job_descriptions:
            print(f"\nProcessing CV: {cv_file}")
            print(f"Job Description: {job_desc_file}")

            print(f"\nScoring with ChatGPT model:")
            score = model.score_cv(cv_text, job_description)
            top_skills = model.extract_top_skills(cv_text, job_description)

            print(f"Score: {score:.4f}")
            print("Top Skills:")
            for skill in top_skills:
                print(
                    f"  - {skill['compétence']}: Score: {skill['score']:.4f}, Années: {skill.get('années', 'N/A')}")

            results.append({
                'cv': cv_file,
                'job_description': job_desc_file,
                'score': score,
                'top_skills': top_skills
            })

    # Save results to CSV
    save_results(results, 'output/scoring_results.csv')
    print("\nResults have been saved to output/scoring_results.csv")

    # Perform analysis on the results
    analyze_results(results)

def analyze_results(results):
    print("\nModel Performance Analysis:")
    avg_score = sum(result['score'] for result in results) / len(results)
    print(f"ChatGPT - Average Score: {avg_score:.4f}")

if __name__ == "__main__":
    main()