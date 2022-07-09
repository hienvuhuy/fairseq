
from fairseq.tasks.translation_coref_probing import TranslationCorefProbingTask

from fairseq import checkpoint_utils
import argparse
from fairseq.tasks import register_task

@register_task("ProbingTranslationTask")
class ProbingTranslationTask(TranslationCorefProbingTask):
    """
        Rescore the t k candidates from each beam using coref task and translation task
    """
