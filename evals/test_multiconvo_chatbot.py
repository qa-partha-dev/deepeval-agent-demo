import os, sys
from deepeval.evaluate import evaluate
from deepeval.metrics import TurnRelevancyMetric, KnowledgeRetentionMetric, ConversationCompletenessMetric
from deepeval.test_case import ConversationalTestCase, Turn
sys.path.insert( 0, os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) ) )
from chatbot import chat

#Goldens
turns = []
history = []

for user_msg in [
        "Hi! I placed an order last week, the order ID is ORD-1042.",
        "Is it going to arrive on time?",
        "What was the ETA you just mentioned?",   # tests memory retention
        " Can I upgrade to express shipping?",
    ]:
        reply, history, tools_called = chat(user_msg, history)
        turns.append(Turn(role="user", content=user_msg))
        turns.append(Turn(role="assistant", content=reply))

test_cases= ConversationalTestCase(
    turns= turns
)

#Metric
turnRelMetric = TurnRelevancyMetric(threshold=0.7, model="gpt-4o-mini")
knowledgeRetMetric = KnowledgeRetentionMetric(threshold=0.7, model="gpt-4o-mini")
convCompleteMetric = ConversationCompletenessMetric(threshold=0.7, model="gpt-4o-mini")


#Evaluation
evaluate(test_cases=[test_cases], metrics=[turnRelMetric, knowledgeRetMetric, convCompleteMetric])