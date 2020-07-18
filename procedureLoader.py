import pandas as pd
import numpy as np
import random
tasks = pd.read_csv('tasks44.csv')


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
        self.actual_A = [0,0,0,0,0]
        self.actual_B = [0,0,0,0,0]
        self.actual_C = [0,0,0,0,0]


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
        self.actual_A[0] = row['block_1_A']
        self.actual_A[1] = row['block_2_A']
        self.actual_A[2] = row['block_3_A']
        self.actual_A[3] = row['block_4_A']
        self.actual_A[4] = row['block_5_A']
        self.actual_B[0] = row['block_1_B']
        self.actual_B[1] = row['block_2_B']
        self.actual_B[2] = row['block_3_B']
        self.actual_B[3] = row['block_4_B']
        self.actual_B[4] = row['block_5_B']
        self.actual_C[0] = 1 - (row['block_1_A'] + row['block_1_B'])
        self.actual_C[1] = 1 - (row['block_2_A'] + row['block_2_B'])
        self.actual_C[2] = 1 - (row['block_3_A'] + row['block_3_B'])
        self.actual_C[3] = 1 - (row['block_4_A'] + row['block_4_B'])
        self.actual_C[4] = 1 - (row['block_5_A'] + row['block_5_B'])

        # calc the ADD value if to_add_mean_C == 1
        if self.to_add_mean_C == '1':
            exp_valueA = self.result_A1 * self.p_A1 + self.result_A2 * (1-self.p_A1)
            exp_valueB = self.result_B1 * self.p_B1 + self.result_B2 * (1-self.p_B1)
            self.added_value_to_C = np.mean([exp_valueA, exp_valueB])

    def get_buttons_results(self):
        sample_numA = random.random()
        sample_numB = random.random()
        sample_numC = random.random()

        # ea is drawn from N(0, SA)
        Ea = np.random.normal(0, self.standart_dev_SA)
        Eb = np.random.normal(0, self.standart_dev_SB)
        Ec = 0

        if self.correlation_AB == 1:
            sample_numB = sample_numA
        if self.correlation_AB == -1:
            sample_numB = 1 - sample_numA

        if sample_numA < self.p_A1:
            result_A = self.result_A1
        else:
            result_A = self.result_A2

        if sample_numB < self.p_B1:
            result_B = self.result_B1
        else:
            result_B = self.result_B2

        result_C = None
        if self.num_buttons == 3:
            Ec = np.random.normal(0, float(self.standart_dev_SC))

            if self.correlation_BC == 1:
                sample_numC = sample_numB
            if self.correlation_BC == -1:
                sample_numC = 1 - sample_numB

            if sample_numC < float(self.p_C1):
                result_C = float(self.result_C1) + Ec
            else:
                result_C = float(self.result_C2) + Ec

        return ( result_A + Ea , result_B + Eb, result_C)


    def run_task(self, choice_func, sample_size, random_choice_for_turns):

        for trail in range(0, self.num_of_choices):
            choice = choice_func( self.num_buttons, self.choices, self.num_buttons, sample_size, random_choice_for_turns)

            (resA, resB, resC) = self.get_buttons_results()
            #print("choice: ", choice, "results:", (resA, resB, resC))

            self.get_chosen_result(choice, resA, resB, resC)

        #print("choices:", self.choices)
        self.calc_decision_blocks()
        csv_name = str(sample_size) + '_' + str(random_choice_for_turns)
        self.save_csv(csv_name)

    def get_chosen_result(self, choice, resA, resB, resC):
        chosen_reward = -99999
        if choice == 'A':
            chosen_reward = resA
        if choice == 'B':
            chosen_reward = resB
        if choice == 'C':
            chosen_reward = resC

        self.choices.append({'choice': choice, 'chosen_reward': chosen_reward,
                             'result_A': resA, 'result_B': resB, 'result_C': resC})


    def calc_decision_blocks(self):

        block_size = self.num_of_choices/self.num_blocks

        choices = [choice['choice'] for choice in self.choices]

        blocks = np.array(choices).reshape(self.num_blocks, int(block_size))

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


    def save_csv(self, params_name ):
        # id of the trial, self.percentage_sum
        accuracy = 0
        if self.id < 45:
            accuracy = self.calc_accuracy_per_row()
        #print('accuracy:', accuracy)
        df = pd.DataFrame([self.id] + self.a_block_percentage + self.b_block_percentage + self.c_block_percentage + [accuracy])
        df = df.transpose()
        #print(df)

        df.to_csv("output" + params_name + ".csv", mode ='a', index=False, header=False)
        self.choices = []
        self.a_block_percentage = []
        self.b_block_percentage = []
        self.c_block_percentage = []

    def calc_accuracy_per_row(self):

        result = 0
        for i in range(0, 5):
            result += (self.a_block_percentage[i] - self.actual_A[i])**2
            result += (self.b_block_percentage[i] - self.actual_B[i])**2

            if self.num_buttons == 3:
                result += (self.c_block_percentage[i] - self.actual_C[i])**2

        if self.num_buttons == 3:
            result /= 15
        elif self.num_buttons == 2:
            result /= 10

        return result

Tasks = []

for index, row in tasks.iterrows():
    current_task = Task()
    current_task.load_task(row)
    Tasks.append(current_task)

#num of clicks per task
T = 100

def identify_rare_dis(choices, num_buttons):
    results = [(choices[index]['result_A'], choices[index]['result_B'], choices[index]['result_C']) for index in range(0, len(choices) - 1)]
    meanA = np.mean([result[0] for result in results])
    stdA = np.std([result[0] for result in results])

    meanB = np.mean([result[1] for result in results])
    stdB = np.std([result[1] for result in results])

    meanC = 0
    stdC = 0
    if num_buttons == 3:
        meanC = np.mean([result[2] for result in results])
        stdC = np.std([result[2] for result in results])

    print(meanA , " " , meanB , " " , meanC, " ", stdA, " ", stdB, " ", stdC)

    for choice in choices:
        resA = choice['result_A']
        resB = choice['result_B']
        resC = choice['result_C']
    return 0

def choice_rule( num, choices, num_buttons, sample_size, random_choice_for_turns):
    # small samples random, try diffrent sample size 5

    # check choices and seen history for rare disaster:
    if len(choices) > random_choice_for_turns:
        rare_disaster_choice = identify_rare_dis(choices, num_buttons)

    if len(choices) < random_choice_for_turns:
        if num_buttons == 3:
            return random.sample(['A','B','C'], 1)[0]
        return 'A' if random.random() < 0.5 else 'B'

    # TODO should sample_size be bigger than the random_choice_for_turns?
    # should sample size stay the same for all rounds of choice?
    if len(choices) > sample_size:
        small_samples = random.sample(choices, sample_size)
    else:
        small_samples = random.sample(choices, len(choices))

    return get_choice_for_buttons(num_buttons, small_samples, sample_size)


def get_choice_for_buttons ( num_buttons, small_samples, sample_size):

    small_sample_results = {}

    if num_buttons == 2:
        small_sample_results = {'A': 0, 'B': 0}
    elif num_buttons == 3:
        small_sample_results = {'A': 0, 'B': 0, 'C': 0}

    # find mean result for every button
    for sample in small_samples:
        small_sample_results['A'] += sample['result_A']
        small_sample_results['B'] += sample['result_B']

        if num_buttons == 3:
            small_sample_results['C'] += sample['result_C']


    small_sample_results['A'] /= sample_size
    small_sample_results['B'] /= sample_size
    if num_buttons == 3:
        small_sample_results['C'] /= sample_size

    # choose button with highest mean
    max_choice = max(small_sample_results,  key=small_sample_results.get)

    #print("max choice is: ", max_choice, "max mean:", max(small_sample_results),
    #      "small sample mean: ", small_sample_results)

    return(max_choice)
    # rare treasures + rare disasters ? stove



# Create empty pandas DataFrame add column names
data = []
df = pd.DataFrame(data, columns = ["a1", "a2", "a3", "a4", "a5", "b1", "b2", "b3", "b4", "b5", "c1", "c2", "c3", "c4", "c5", "Briars"])


def calc_briers_avg(csv_name):
    results = pd.read_csv(csv_name)
    average = results["Briars"].mean()
    return average


def find_params( tasks, list_of_sample_size, list_random_choice_for_turns ):
    min_briers_score = 1
    best_sample_size = -1
    best_random_choice = -1

    for random_choice in list_random_choice_for_turns:
        for sample_size in list_of_sample_size:
            csv_name = "output" + str(sample_size) + '_' + str(random_choice) + ".csv"
            df.to_csv(csv_name)
            for task in tasks:
                task.run_task(choice_rule, sample_size, random_choice)
                task.run_task(choice_rule, sample_size, random_choice)
                task.run_task(choice_rule, sample_size, random_choice)
                task.run_task(choice_rule, sample_size, random_choice)
                task.run_task(choice_rule, sample_size, random_choice)
                task.run_task(choice_rule, sample_size, random_choice)
                task.run_task(choice_rule, sample_size, random_choice)
                task.run_task(choice_rule, sample_size, random_choice)
                task.run_task(choice_rule, sample_size, random_choice)
                task.run_task(choice_rule, sample_size, random_choice)
            # compare each briars score and return the params for the minimum best
            briers_score_avg = calc_briers_avg(csv_name)

            if briers_score_avg < min_briers_score:
                min_briers_score = briers_score_avg
                best_sample_size = sample_size
                best_random_choice = random_choice

    print("min_briers_score: ", min_briers_score, "  best sample size: ", best_sample_size, "  best_random_choice: ", best_random_choice)

    return 0


def find_random_choice_for_turns():
    return 0

# TODO
#  refine decision rule based on worst accuracy,
#  decide on two phenomena - small samples - underweighting of rare events, hot stove
#  terrible disaster happends- hot stove effect
#  add Rab correlation


find_params(Tasks, range(4,6), range(2,7))
#find_params(Tasks, range(3,7), range(3,9))
#find_params(Tasks, range(3,7), range(3,9))
print("hello")




