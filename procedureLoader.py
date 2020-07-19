import pandas as pd
import numpy as np
import random
import statistics
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
        self.num_of_choices = 100
        self.choices = []
        self.num_blocks = 5
        self.a_block_percentage = []
        self.b_block_percentage = []
        self.c_block_percentage = []
        self.actual_A = [0,0,0,0,0]
        self.actual_B = [0,0,0,0,0]
        self.actual_C = [0, 0, 0, 0, 0]
        self.chance_to_chose_risky = 1
        self.danger_button = 'X'


    def load_task(self, row, unknown):
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
        if unknown != "unknown":
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


    def run_task(self, choice_func, sample_size, random_choice_for_turns, csv_name):

        for trail in range(0, self.num_of_choices):
            choice, self.chance_to_chose_risky, self.danger_button = \
                choice_func(self.num_buttons, self.choices, self.num_buttons, sample_size, random_choice_for_turns, self.chance_to_chose_risky, self.danger_button)

            (resA, resB, resC) = self.get_buttons_results()
            #print("choice: ", choice, "results:", (resA, resB, resC))

            self.get_chosen_result(choice, resA, resB, resC)



        #print("choices:", self.choices)
        self.calc_decision_blocks()
        if csv_name == "output_unknown.csv":
            self.save_csv(csv_name, "unknown")
        else:
            self.save_csv(csv_name, "known")

        self.choices = []
        self.danger_button = 'X'
        self.chance_to_chose_risky = 1

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


    def save_csv(self, csv_name, unknown ):
        accuracy = 0
        if self.id < 45:
            accuracy = self.calc_accuracy_per_row()
        #print('accuracy:', accuracy)
        if unknown == "known":
            df = pd.DataFrame([self.id] + self.a_block_percentage + self.b_block_percentage + self.c_block_percentage + [accuracy])
        elif unknown == "unknown":
            df = pd.DataFrame([self.id] + self.a_block_percentage + self.b_block_percentage)
        df = df.transpose()
        #print(df)

        df.to_csv(csv_name, mode ='a', index=False, header=False)
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
                result += (self.c_block_percentage[i] - self.actual_C[i]) ** 2

        if self.num_buttons == 3:
            result /= 15
        elif self.num_buttons == 2:
            result /= 10

        return result

Tasks = []

for index, row in tasks.iterrows():
    current_task = Task()
    current_task.load_task(row, "known")
    Tasks.append(current_task)

#num of clicks per task
T = 100

def identify_rare_dis(choices, num_buttons):
    results_before_last = [(choices[index]['result_A'], choices[index]['result_B'], choices[index]['result_C']) for index in range(0, len(choices) - 1)]
    stdA = statistics.stdev([result[0] for result in results_before_last])
    stdB = statistics.stdev([result[1] for result in results_before_last])

    stdC = 0
    if num_buttons == 3:
        stdC = statistics.stdev([result[2] for result in results_before_last])

    results_with_last = [(choices[index]['result_A'], choices[index]['result_B'], choices[index]['result_C']) for index in range(0, len(choices))]
    stdA_last = statistics.stdev([result[0] for result in results_with_last])
    stdB_last = statistics.stdev([result[1] for result in results_with_last])

    stdC_last = 0
    if num_buttons == 3:
        stdC_last = statistics.stdev([result[2] for result in results_with_last])

    danger_button = 'X'
    chance_to_chose_risky = 1

    chance_to_chose_riskyA, danger_buttonA = check_disaster(stdA, stdA_last, results_with_last, choices, 'A', 0)
    chance_to_chose_riskyB, danger_buttonB = check_disaster(stdB, stdB_last, results_with_last, choices, 'B', 1)

    if chance_to_chose_riskyA == chance_to_chose_riskyB == 1:
        danger_button = 'X'
    elif chance_to_chose_riskyA < chance_to_chose_riskyB:
        chance_to_chose_risky = chance_to_chose_riskyA
        danger_button = 'A'
    else:
        chance_to_chose_risky = chance_to_chose_riskyB
        danger_button = 'B'

    if num_buttons == 3:
        chance_to_chose_riskyC, danger_buttonC = check_disaster(stdC, stdC_last, results_with_last, choices, 'C', 2)

        if chance_to_chose_riskyC < chance_to_chose_riskyB:
            chance_to_chose_risky = chance_to_chose_riskyC
            danger_button = 'C'

    return chance_to_chose_risky, danger_button

def check_disaster(std_X, std_X_last, results_with_last_X, choices, danger_button, index ):

    chance_to_chose_risky = 1
    danger_button = danger_button
    if std_X < 3 and std_X_last > 5:

        last_X = results_with_last_X[len(choices)-1][index]
        if last_X < 0 and (np.sum(np.array(results_with_last_X) == last_X) / len(results_with_last_X)) < 0.1:

            # was this option chosen by user?
            my_choice = choices[len(choices)-1]['choice']
            if my_choice == danger_button:
                chance_to_chose_risky = 0.7
            else:
                chance_to_chose_risky = 0.8

            # was extreme minimum value
            if std_X_last > 7:
                chance_to_chose_risky *= 0.9

    return chance_to_chose_risky, danger_button


def choice_rule( num, choices, num_buttons, sample_size, random_choice_for_turns, prev_chance_to_chose_risky, prev_danger_button):
    # small samples random, try diffrent sample size 5

    if len(choices) < random_choice_for_turns:
        if num_buttons == 3:
            return random.sample(['A','B','C'], 1)[0], prev_chance_to_chose_risky, prev_danger_button
        return 'A' if random.random() < 0.5 else 'B', prev_chance_to_chose_risky, prev_danger_button

    chance_to_chose_risky = 0
    danger_button = 'X'

    # check choices and seen history for rare disaster:
    if len(choices) > random_choice_for_turns:
        chance_to_chose_risky, danger_button = identify_rare_dis(choices, num_buttons)

    # we want to preserve previous dangerous button if the new button is less risky or no new danger identified
    if danger_button == 'X' or prev_chance_to_chose_risky < chance_to_chose_risky:
        danger_button = prev_danger_button
        chance_to_chose_risky = prev_chance_to_chose_risky

    # TODO should sample_size be bigger than the random_choice_for_turns?
    # should sample size stay the same for all rounds of choice?
    if len(choices) > sample_size:
        small_samples = random.sample(choices, sample_size)
    else:
        small_samples = random.sample(choices, len(choices))

    return get_choice_for_buttons(num_buttons, small_samples, sample_size, chance_to_chose_risky, danger_button)


def get_choice_for_buttons ( num_buttons, small_samples, sample_size, chance_to_chose_risky, danger_button):

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
    max_choice = max(small_sample_results, key=small_sample_results.get)
    if max_choice == danger_button and random.random() > chance_to_chose_risky:
        # change choice to second max
        small_sample_results.pop(danger_button)
        max_choice = max(small_sample_results, key=small_sample_results.get)

    #print("max choice is: ", max_choice, "max mean:", max(small_sample_results),
    #      "small sample mean: ", small_sample_results)

    return max_choice, chance_to_chose_risky, danger_button
    # rare treasures + rare disasters ? stove


# Create empty pandas DataFrame add column names
data = []
df = pd.DataFrame(data, columns = ["a1", "a2", "a3", "a4", "a5", "b1", "b2", "b3", "b4", "b5", "c1", "c2", "c3", "c4", "c5", "Briars"])
#df = pd.DataFrame(data, columns = ["a1", "a2", "a3", "a4", "a5", "b1", "b2", "b3", "b4", "b5", "Briars"])


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
                task.run_task(choice_rule, sample_size, random_choice, csv_name)

            # compare each briars score and return the params for the minimum best
            briers_score_avg = calc_briers_avg(csv_name)

            if briers_score_avg < min_briers_score:
                min_briers_score = briers_score_avg
                best_sample_size = sample_size
                best_random_choice = random_choice

    print("min_briers_score: ", min_briers_score, "  best sample size: ", best_sample_size, "  best_random_choice: ", best_random_choice)

    return 0



# TODO
#  refine decision rule based on worst accuracy,
#  decide on two phenomena - small samples - underweighting of rare events, hot stove
#  terrible disaster happends- hot stove effect



find_params(Tasks, range(3,8), range(3,7))

def calc_unknown( filename ):
    # Create empty pandas DataFrame add column names
    unknown_tasks = pd.read_csv(filename)
    Tasks = []

    for index, row in unknown_tasks.iterrows():
        current_task = Task()
        current_task.load_task(row, "unknown")
        Tasks.append(current_task)

    data = []
    df = pd.DataFrame(data,
                      columns=["a1", "a2", "a3", "a4", "a5", "b1", "b2", "b3", "b4", "b5"])
    csv_name = "output_unknown.csv"
    df.to_csv(csv_name)
    for task in Tasks:
        # sample_size = # random_choice
        task.run_task(choice_rule, 5, 6, csv_name)
        task.run_task(choice_rule, 5, 6, csv_name)
        task.run_task(choice_rule, 5, 6, csv_name)
        task.run_task(choice_rule, 5, 6, csv_name)
        task.run_task(choice_rule, 5, 6, csv_name)


#calc_unknown( "test_data.csv" )


print("hello")




