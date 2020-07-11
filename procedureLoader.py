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
        self.choices = []
        self.num_blocks = 5
        self.percentage_sum = []
        self.a_block_percentage = []
        self.b_block_percentage = []
        self.c_block_percentage = []

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
        sample_numA = random.random()

        if sample_numA < self.p_A1:
            result_A = self.result_A1
        else:
            result_A = self.result_A2

        sample_numB = random.random()
        if sample_numB < self.p_B1:
            result_B = self.result_B1
        else:
            result_B = self.result_B2

        result_C = None
        if self.num_buttons == 3:
            sample_numC = random.random()
            if sample_numC < float(self.p_C1):
                result_C = self.result_C1
            else:
                result_C = self.result_C2

        #history_click_results.append(-1)
        return ( result_A , result_B, result_C )


    def run_task(self, choice_func):

        print("~~~ Task number :", self.id, "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        for trail in range(0, self.num_of_choices):
            choice = choice_func( self.num_buttons )
            self.choices.append(choice)
            (resA, resB, resC) = self.get_buttons_results()
            print("choice: ", choice, "results:", (resA, resB, resC))
        print("choices:", self.choices)
        self.calc_decision_blocks()
        self.save_csv()


    def calc_decision_blocks(self):

        block_size = self.num_of_choices/self.num_blocks

        blocks = np.array(self.choices).reshape(self.num_blocks, int(block_size))

        for block in blocks:
            num_of_A = 0
            num_of_B = 0
            num_of_C = 0
            choices, counts = np.unique(block, return_counts=True)

            # for every choice in a block calc count of choice
            for indx, count in enumerate(counts):
                if choices[indx] == 'A':
                    num_of_A = count
                if choices[indx] == 'B':
                    num_of_B = count
                if choices[indx] == 'C':
                    num_of_C = count

            self.a_block_percentage.append(num_of_A / block_size)
            self.b_block_percentage.append(num_of_B / block_size)
            self.c_block_percentage.append(num_of_C / block_size)

        print('A', self.a_block_percentage)
        print('B', self.b_block_percentage)
        print('C', self.c_block_percentage)


    def save_csv(self ):
        # id of the trial, self.percentage_sum

        df = pd.DataFrame([self.id] + self.a_block_percentage + self.b_block_percentage + self.c_block_percentage)
        print(df)
        df.to_csv("output.csv", mode ='a', index=False)


Tasks = []

for index, row in tasks.iterrows():
    current_task = Task()
    current_task.load_task(row)
    Tasks.append(current_task)

#num of clicks per task
T = 100

def choice_rule( num ):

    return 'A' if random.random() < 0.5 else 'B'



for task in Tasks:
    # TODO create decision rule
    task.run_task( choice_rule )


print("hello")