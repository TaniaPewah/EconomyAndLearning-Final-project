import pandas as pd
import numpy as np
import random
tasks = pd.read_csv('tasks.csv')

class Task(object):
    def __init__(self ):
        self.id = None
        self.num_buttons = None
        self.result_A1 = None
        self.p_A1 = None
        self.result_A2 = None
        self.standart_dev_SA = None
        self.result_B1 = None
        self.p_B1 = None
        self.result_B2 = None
        self.standart_dev_SB = None
        self.result_C1 = None
        self.p_C1 = None
        self.result_C2 = None
        self.standart_dev_SC = None
        self.correlation_AB = None
        self.correlation_BC = None
        self.to_add_mean_C = None
        self.added_value_to_C = None
        self.history_click_results =[]
        self.num_of_choices = 100

    def load_task(self, row):
        # load from df to properties of Task
        self.id = row['task_n']
        self.num_buttons = row['n']
        self.result_A1 = row['a1']
        self.p_A1 = row['pa1']
        self.result_A2 = row['a2']
        self.standart_dev_SA = row['sa']
        self.result_B1 = row['b1']
        self.p_B1 = row['pb1']
        self.result_B2 = row['b2']
        self.standart_dev_SB = row['sb']
        self.result_C1 = row['c1']
        self.p_C1 = row['pc1']
        self.result_C2 = row['c2']
        self.standart_dev_SC = row['sc']
        self.correlation_AB = row['rab']
        self.correlation_BC = row['rbc']
        self.to_add_mean_C = row['ADD']

        # calc the ADD value if to_add_mean_C == 1
        if self.to_add_mean_C == '1':
            exp_valueA = self.result_A1 * self.p_A1 + self.result_A2 * (1-self.p_A1)
            exp_valueB = self.result_B1 * self.p_B1 + self.result_B2 * (1-self.p_B1)
            self.added_value_to_C = np.mean([exp_valueA, exp_valueB])

    def get_buttons_results(self):
        sample_numA = random.random(0, 1)
        if sample_numA < self.p_A1:
            result_A = self.result_A1
        else:
            result_A = self.result_A2

        sample_numB = random.random(0, 1)
        if sample_numB < self.p_B1:
            result_B = self.result_B1
        else:
            result_B = self.result_B2

        result_C = None
        if self.num_buttons == 3:
            sample_numC = random.random(0,1)
            if sample_numC < self.p_C1:
                result_C = self.result_C1
            else:
                result_C = self.result_C2

        #history_click_results.append(-1)
        return ( result_A , result_B, result_C )


    def run_task(self, choice_func):
        for trail in range(0, self.num_of_choices):
            choice = choice_func( self.num_buttons )
            (resA, resB, resC) = self.get_buttons_results()
            print("choice: ", choice, "results:", (resA, resB, resC))

Tasks = []

for index, row in tasks.iterrows():
    current_task = Task()
    current_task.load_task(row)
    Tasks.append(current_task)

#num of clicks per task
T = 100

def choice_rule( num ):
    return 'A'



for task in Tasks:
    # TODO create decision rule
    task.run_task( choice_rule )


print("hello")