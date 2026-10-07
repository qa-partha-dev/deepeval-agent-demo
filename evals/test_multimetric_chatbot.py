import os
import sys

from deepeval.contextvars import get_current_golden
from deepeval.dataset import dataset, Golden
from deepeval.evaluate import evaluate
from deepeval.metrics import TaskCompletionMetric, ToolCorrectnessMetric
from deepeval.test_case import ToolCall, LLMTestCase
from deepeval.tracing import observe, update_current_trace

sys.path.insert( 0, os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) ) )
from agent_instrumented import support_agent as _support_agent

###################################################################################################################
# TEST CASE: using LLMTestCase(input, actual_output, tools_called[ToolCall(name)], expected_tools)          #
# or                                                                                                               #
# TEST CASE: using dataset.EvaluationDataset(goldens[Golden(input, expected_tools)])                         #
# Add metrics TaskCompletionMetric(threshold, model)                                                              #
# Add evaluation by dataSet.evals_iterator(ToolCorrectnessMetric(), TaskCompletionMetric(threshold, model)])      #
# or                                                                                                               #
# evaluate(test_cases[], metrics[])                                                                                #
###################################################################################################################

#Metrics
tool_correctness = ToolCorrectnessMetric()
task_completion_metric = TaskCompletionMetric(threshold=0.7,model = "gpt-4o-mini")

#Golden
dataSet= dataset.EvaluationDataset(goldens= [
    Golden(input="Where is my order ORD-1043?", expected_tools=[ToolCall(name="get_order_status")]),
    Golden(input="What is the refund policy for electronics ?", expected_tools=[ToolCall(name="get_refund_policy")] ),
])

#Test Case normal way
test_case = LLMTestCase(
    input = "Where is my order ORD-1042?",
    actual_output = _support_agent("Where is my order ORD-1042?"),
    tools_called = [ToolCall(name="get_order_status"), ToolCall(name="get_refund_policy")],
    expected_tools=[ToolCall(name="get_order_status")],
)

#Tracing
@observe
def support_agent(user_input: str) -> str:
    golden = get_current_golden()
    if golden:
        if golden.expected_tools:
            update_current_trace(expected_tools=golden.expected_tools)
        if golden.expected_output:
            update_current_trace(expected_output=golden.expected_output)

    return _support_agent(user_input)

#Execution
for golden in dataSet.evals_iterator(metrics=[tool_correctness, task_completion_metric]):
    support_agent(golden.input)

evaluate(test_cases=[test_case], metrics=[tool_correctness, task_completion_metric])

