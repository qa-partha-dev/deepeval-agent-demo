#Project Dir
import os, sys
from deepeval.dataset import dataset, Golden
from deepeval.metrics import PromptAlignmentMetric, StepEfficiencyMetric, GEval, AnswerRelevancyMetric
from deepeval.test_case import SingleTurnParams
from deepeval.tracing import observe

sys.path.insert( 0, os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) ) )
from agent_instrumented import support_agent as _support_agent

@observe(name="support_agent")
def support_agent (user_input: str) -> str:
    return _support_agent(user_input)

#Metrics
promptAlignmentMetric = PromptAlignmentMetric(
    prompt_instructions=[
        "You are a friendly customer-support agent. "
        "Use the available tools to answer order and refund questions. "
        "Keep replies short and helpful."
    ],
    threshold=0.7,
    model = "gpt-4o-mini"
)

stepEfficiency = StepEfficiencyMetric(threshold=0.7, model = "gpt-4o-mini")
factCorrectness = GEval(
    name="factCorrectness",
    model = "gpt-4o-mini",
    threshold=0.7,
    evaluation_params=[
        SingleTurnParams.INPUT,
        SingleTurnParams.EXPECTED_OUTPUT,
        SingleTurnParams.ACTUAL_OUTPUT
    ]

)

answerRelevancyMetric = AnswerRelevancyMetric(threshold=0.7, model = "gpt-4o-mini")

#Goldens
dataset = dataset.EvaluationDataset(
    goldens=[
        Golden(input="Where is my order ORD-1043?"),
        Golden(input="What is the refund policy for electronics?"),
        Golden(input="I want to return ORD-2099, what should I do?"),

    ]
)

#Execution
for golden in dataset.evals_iterator(metrics=[promptAlignmentMetric, stepEfficiency, factCorrectness, answerRelevancyMetric ]):
    support_agent(golden.input)


