import openai
import os
import json
import time
from src.scoring.cv_scorer import BaseScorer
from src.utils.skill_extractor import extract_skills_certifications_projects
from src.config.scorer_config import ScorerConfig


class ChatGPTScorer(BaseScorer):
    def __init__(self):
        self.config = ScorerConfig()
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if not openai.api_key:
            raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")

    def _get_completion(self, prompt, max_retries=3):
        for attempt in range(max_retries):
            try:
                response = openai.ChatCompletion.create(
                    model=self.config.get('models.chatgpt.model_name', 'gpt-3.5-turbo'),
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.config.get('models.chatgpt.temperature', 0.2),
                    max_tokens=self.config.get('models.chatgpt.max_tokens', 500)
                )
                return response.choices[0].message.content.strip()
            except openai.error.OpenAIError as e:
                print(f"OpenAI API error on attempt {attempt + 1}: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    print("Max retries reached. Falling back to default scoring.")
                    return None
            except Exception as e:
                print(f"Unexpected error on attempt {attempt + 1}: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    print("Max retries reached. Falling back to default scoring.")
                    return None
        return None

    def score_cv(self, cv_text, job_description):
        skills, certifications, projects = extract_skills_certifications_projects(cv_text)

        prompt = f"""
        Given the following CV and job description, provide a relevance score between 0 and 1, 
        where 1 is a perfect match and 0 is completely irrelevant. Consider the following factors with their respective weights:

        1. Overall match between CV and job requirements (Weight: {self.config.get('global.skill_weight', 0.4)})
        2. Relevance of skills mentioned in the CV to the job requirements (Weight: {self.config.get('global.skill_weight', 0.4)})
        3. Years of experience for each skill (Weight: {self.config.get('global.experience_weight', 0.3)})
        4. Relevance of certifications to the job (Weight: {self.config.get('global.certification_weight', 0.2)})
        5. Relevance of projects to the job requirements (Weight: {self.config.get('global.project_weight', 0.1)})

        CV:
        {cv_text}

        Job Description:
        {job_description}

        Skills with experience: {skills}
        Certifications: {certifications}
        Projects: {projects}

        Provide your response in the following format:
        Score: [Your score between 0 and 1]
        Explanation: [Brief explanation of your scoring]
        """

        response = self._get_completion(prompt)
        if response is None:
            # Fallback scoring method
            return self._fallback_scoring(cv_text, job_description)

        try:
            score_line = response.split('\n')[0]
            score = float(score_line.split(':')[1].strip())
            return min(max(score, 0), 1)  # Ensure score is between 0 and 1
        except Exception as e:
            print(f"Error parsing score: {e}")
            return self._fallback_scoring(cv_text, job_description)

    def extract_top_skills(self, cv_text, job_description):
        skills, certifications, projects = extract_skills_certifications_projects(cv_text)

        prompt = f"""
        Given the following CV and job description, identify the top {self.config.get('global.top_skills_count', 3)} most relevant skills 
        mentioned in the CV that match the job requirements. For each skill, provide a relevance score 
        between 0 and 1, considering:

        1. The skill's importance to the job description (Weight: {self.config.get('global.skill_weight', 0.4)})
        2. The candidate's years of experience with the skill (Weight: {self.config.get('global.experience_weight', 0.3)})
        3. Related certifications mentioned in the CV (Weight: {self.config.get('global.certification_weight', 0.2)})
        4. Relevant projects that demonstrate the skill (Weight: {self.config.get('global.project_weight', 0.1)})

        CV:
        {cv_text}

        Job Description:
        {job_description}

        Skills with experience: {skills}
        Certifications: {certifications}
        Projects: {projects}

        Provide the result in the following JSON format:
        [
            {{"compétence": "Skill Name", "score": 0.95, "années": 5}},
            {{"compétence": "Another Skill", "score": 0.8, "années": 3}},
            {{"compétence": "Third Skill", "score": 0.7, "années": 2}}
        ]
        """

        response = self._get_completion(prompt)
        if response is None:
            # Fallback skill extraction method
            return self._fallback_skill_extraction(cv_text, job_description)

        try:
            skills = json.loads(response)
            return skills[:self.config.get('global.top_skills_count', 3)]
        except json.JSONDecodeError:
            print(f"Invalid JSON returned: {response}")
            return self._fallback_skill_extraction(cv_text, job_description)

    def _fallback_scoring(self, cv_text, job_description):
        # Implement a simple fallback scoring method
        # This is a very basic method and should be improved for better results
        common_words = set(cv_text.lower().split()) & set(job_description.lower().split())
        score = len(common_words) / len(set(job_description.lower().split()))
        return min(score, 1.0)  # Ensure score is not above 1

    def _fallback_skill_extraction(self, cv_text, job_description):
        # Implement a simple fallback skill extraction method
        # This is a very basic method and should be improved for better results
        skills, _, _ = extract_skills_certifications_projects(cv_text)
        job_words = set(job_description.lower().split())

        skill_scores = []
        for skill, years in skills.items():
            if skill.lower() in job_words:
                skill_scores.append({
                    "compétence": skill,
                    "score": 0.5,  # Default score
                    "années": years
                })

        return sorted(skill_scores, key=lambda x: x["score"], reverse=True)[
               :self.config.get('global.top_skills_count', 3)]