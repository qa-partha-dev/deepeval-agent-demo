#turns -> history
#Evaluation
from deepeval.test_case import MultiTurnParams
from deepeval.evaluate import evaluate
from deepeval.metrics import ConversationalGEval
from deepeval.test_case import ConversationalTestCase, Turn

from chatbot import chat

turns=[]
history=[]

# 1. Create turns [require: reply and history] by sending msg
for user_msg in [
        "Hi! I placed an order last week, the order ID is ORD-1042.",
        "Is it going to arrive on time?",
        "What was the ETA you just mentioned?",   # tests memory retention
        " Can I upgrade to express shipping?",
    ]:
        #Get the *reply,history,tools_called* from AI by calling chat()
        reply, history, tools_called = chat(user_msg, history)
        #Create and push the turn[user_msg and reply] to turns[]
        turns.append(Turn(role= "user", content=user_msg))
        turns.append(Turn(role="assistant", content=reply))

# 2. Prepare test cases
test_cases = ConversationalTestCase(
    turns=turns
)

#3. Create Custom Metrics
correctness= ConversationalGEval(
    name="Correctness",
    criteria=(
        "Did the chatbot fully resolve the customer's issue?"
        "It should use tools when needed and provide accurate answers."
    ),
    model="gpt-4o-mini",
    threshold=0.8,
    evaluation_params = [
        MultiTurnParams.ROLE,
        MultiTurnParams.CONTENT,
    ])


#4. Evaluate
evaluate (test_cases = [test_cases], metrics = [correctness])
