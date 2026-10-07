from deepeval.contextvars import get_current_golden
from deepeval.metrics import ContextualPrecisionMetric, ContextualRecallMetric, AnswerRelevancyMetric, \
    FaithfulnessMetric
from deepeval.tracing import update_current_trace, observe

from rag_agent import rag_support_agent as _rag_support_agent
from deepeval.dataset import EvaluationDataset, Golden


#Tracing
@observe(name="rag_support_agent")
def rag_support_agent (user_input: str) -> str:
    golden = get_current_golden()
    if golden:
        if golden.expected_output:
            update_current_trace( expected_output=golden.expected_output)
    return _rag_support_agent(user_input)

# DataSet
dataset=EvaluationDataset(
    goldens = [
    Golden(
        input="What is the return policy for electronics?",
        expected_output=(
            "Electronics can be returned within 15 days of delivery if unopened "
            "and in original packaging. Refunds take 5–7 business days."
        ), ),

    Golden(
        input="How long does express shipping take and what does it cost?",
        expected_output=(
            "Express shipping takes 1–2 business days and costs $15. "
            "Orders placed before 2 PM are dispatched the same day."
        ),
    ), ]
)

#Metrics
precisionMetric = ContextualPrecisionMetric(
    model="gpt-4o-mini",
    threshold= 0.7,
    include_reason=True
)
recallMetric = ContextualRecallMetric(
    model="gpt-4o-mini",
    threshold=0.7,
    include_reason=True
)
ansRelevancyMetric = AnswerRelevancyMetric(
    model="gpt-4o-mini",
    threshold=0.7,
    include_reason=True
)
faithfulMetric = FaithfulnessMetric(
    model="gpt-4o-mini",
    threshold=0.7,
    include_reason=True
)

#Evaluation
for golden in dataset.evals_iterator(metrics=[precisionMetric, recallMetric, ansRelevancyMetric, faithfulMetric]):
    rag_support_agent(golden.input)