# Notes
### Metrics List
1. ToolCorrectnessMetric()
2. TaskCompletionMetric(threshold=0.7,model = "gpt-4o-mini")
3. promptAlignmentMetric()
4. StepEfficiencyMetric(threshold=0.7, model = "gpt-4o-mini")
5. Custom Metric

## Steps
1. TEST CASE: using `LLMTestCase(input, actual_output, tools_called[ToolCall(name)], expected_tools)`          
or                                                                                                               
TEST CASE: using `dataset.EvaluationDataset(goldens[Golden(input, expected_tools)]) `                        
2. Add metrics `TaskCompletionMetric(threshold, model)  `                                                            
3. Add evaluation by `for golden in dataSet.evals_iterator(metrics=[tool_correctness, task_completion_metric]):
    support_agent(golden.input)`      
or                                                                                                              
`evaluate(test_cases[], metrics[]) `                                                                               
