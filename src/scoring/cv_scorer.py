from abc import ABC, abstractmethod


class BaseScorer(ABC):
    @abstractmethod
    def score_cv(self, cv_text, job_description):
        pass

    @abstractmethod
    def extract_top_skills(self, cv_text, job_description):
        pass


class CVScorer:
    def __init__(self, model):
        self.model = model

    def score(self, cv_text, job_description):
        global_score = self.model.score_cv(cv_text, job_description)
        top_skills = self.model.extract_top_skills(cv_text, job_description)

        return {
            "global_score": global_score,
            "top_skills": top_skills
        }
