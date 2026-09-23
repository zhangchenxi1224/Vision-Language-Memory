Exact upstream utils/utils_mcq.py and LICENSE, copied without edits from
amazon-science/PrefEval commit 50795054b5ff5f418d2b768a331d71e480f93331.
Only pure MCQ formatting and parsing functions are loaded through AST so API client
dependencies are not imported. Official data are in the existing alignment export.

Also preserves generation_task/llm_based_evaluation_errortypes.py,
generation_task/get_preference_following_accuracy_generation_task.py and error_type
prompts from the same revision, unchanged. Judge parsing and aggregation are loaded
from these functions; API transport/model substitutions must be reported separately.
