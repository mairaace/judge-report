from llm_judge_reporting import point_estimator, confidence_interval
from config import EVALUATED_MODELS, DATA_BASE_PATH, JUDGES_OUTPUT_PATH, PILOT_CSV_PATH

'''
en outputs deberíamos guardar las evaluaciones humanas . 

'''


'''
Aquí hay que calcular las métricas 

'''

'''
data falsa
'''
p_test = 0.7977      
n_test = 1600        # Total de preguntas
m0_final = 38        # Total de Incorrectas Reales del set 
m1_final = 62        # Total de Correctas Reales en el set 
q0_final = 0.55      # % de acierto del juez en las m0
q1_final = 0.92      # % de acierto del juez en las m1

# calcular precición corregida con la estimación puntal 
theta_hat = point_estimator(p_test, q0_final, q1_final)

# calcular intervalo de 
ci_lower, ci_upper = confidence_interval(
    p=p_test, 
    q0=q0_final, 
    q1=q1_final, 
    n=n_test, 
    m0=m0_final, 
    m1=m1_final, 
    alpha=0.05
)

'''
guardar los resultados en un csv:
'''