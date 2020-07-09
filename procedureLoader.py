import pandas as pd
import numpy as np
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

    def click_A(self):
        x=4


        #history_click_results.append(-1)

    def click_B(self):
        x=4

    def click_C(self):
        x=5


Tasks = []

for index, row in tasks.iterrows():
    current_task = Task()
    current_task.load_task(row)
    Tasks.append(current_task)

print("hello")