

import random
import argparse
import codecs
import os
import sys

import numpy as np


def call_forward(file_name):
    file = open(file_name)

    count = 0
    word_count = 0
    while True:
        line = file.readline()
        if not line:
            break
        if count % 2 != 0:
            words_array = line.split()
            obs = Observation([], words_array)
            model.forward(obs)
            # Create a new file and write the obs object to the file
            output_file = open('ambiguous_sents.output.obs', 'a')
            output_file.write(str(obs))
            output_file.close()
        count += 1
    # print('No of words', word_count)
    file.close()

def call_viterbi(file_name):
    file = open(file_name)

    count = 0
    word_count = 0
    while True:
        line = file.readline()
        if not line:
            break
        if count % 2 != 0:
            words_array = line.split()
            obs = Observation([], words_array)
            model.viterbi(obs)
            # Create a new file and write the obs object to the file
            output_file = open('ambiguous_sents.viterbi.output.obs', 'a')
            output_file.write(str(obs))
            output_file.close()
        count += 1
    # print('No of words', word_count)
    file.close()
# observations
class Observation:
    def __init__(self, stateseq, outputseq):
        self.stateseq  = stateseq   # sequence of states
        self.outputseq = outputseq  # sequence of outputs
    def __str__(self):
        return ' '.join(self.stateseq)+'\n'+' '.join(self.outputseq)+'\n'
    def __repr__(self):
        return self.__str__()
    def __len__(self):
        return len(self.outputseq)

# hmm model
class HMM:
    def __init__(self, transitions={}, emissions={}):
        """creates a model from transition and emission probabilities"""
        ## Both of these are dictionaries of dictionaries. e.g. :
        # {'#': {'C': 0.814506898514, 'V': 0.185493101486},
        #  'C': {'C': 0.625840873591, 'V': 0.374159126409},
        #  'V': {'C': 0.603126993184, 'V': 0.396873006816}}

        self.transitions = transitions
        self.emissions = emissions

    ## part 1 - you do this.
    def load(self, basename):
        """reads HMM structure from transition (basename.trans),
        and emission (basename.emit) files,
        as well as the probabilities."""

        trans_file = open(basename+'.trans', 'r')
        emission_file = open(basename+'.emit', 'r')

        while True :
            line = trans_file.readline()
            line_array = line.split()
            if not line:
                break

            if line_array[0] in self.transitions:
                inner_dict = self.transitions.get(line_array[0])
                inner_dict[line_array[1]] = float(line_array[2])
            else:
                inner_dict = {}
                inner_dict[line_array[1]] = float(line_array[2])

                self.transitions[line_array[0]] = inner_dict

        trans_file.close()

        while True :
            line = emission_file.readline()
            line_array = line.split()
            if not line:
                break

            if line_array[0] in self.emissions:
                inner_dict = self.emissions.get(line_array[0])
                inner_dict[line_array[1]] = float(line_array[2])
            else:
                inner_dict = {}
                inner_dict[line_array[1]] = float(line_array[2])

                self.emissions[line_array[0]] = inner_dict

        emission_file.close()

   ## you do this.
    def generate(self, n):
        """return an n-length observation by randomly sampling from this HMM."""
        obs = Observation([], [])
        start_state = '#'
        count = 0


        list_trans = []
        list_emission = []
        while count!=n:
            if start_state in self.transitions:
                inner_dict = self.transitions[start_state]
                list_of_values = list(inner_dict.keys())
                list_of_probabilites = list(inner_dict.values())
                # print(list_of_probabilites)
                random_value = random.choices(list_of_values, weights=list_of_probabilites, k=1)
                # list_trans.append(random_value[0]) #Taking 0th element as random.choices return a list and we have k=1 ie one value we get it from random_value[0]
                obs.stateseq.append(random_value[0])
                start_state = random_value[0]
            count+=1

        for item in obs.stateseq:
            if item in self.emissions:
                inner_dict = self.emissions[item]
                list_of_values = list(inner_dict.keys())
                list_of_probabilites = list(inner_dict.values())
                random_value = random.choices(list_of_values, weights=list_of_probabilites, k=1)
                # list_emission.append(random_value[0])
                obs.outputseq.append(random_value[0])
        print(obs.stateseq)
        print(obs.outputseq)
    def forward(self, observation):
        #create a matrix with columns as observations and rows would be the states
        col_list = observation.outputseq #observations
        row_list = list(self.transitions.keys()) #states


        for element in row_list:
            index_1 = row_list.index(row_list[0])
            index_2 = row_list.index('#')
            if(element == '#'):
                row_list[index_1], row_list[index_2] = row_list[index_2], row_list[index_1]

        col = len(col_list)
        row = len(row_list)
        matrix = [[0 for _ in range(col)] for _ in range(row)]

        for i in range(1, row):

            key = row_list[i]
            if key in self.emissions:
                inner_dict_emit = self.emissions[key]
                first_column = col_list[0]
                if(first_column in inner_dict_emit):
                    inner_dict_trans = self.transitions[row_list[0]]
                    tran_probability = inner_dict_trans[row_list[i]]
                    emit_probability = inner_dict_emit[first_column]
                    probability = tran_probability * emit_probability
                    matrix[i][0] = probability
                else:
                    matrix[i][0] = 0
        probability = 0
        for j in range(1, col):
            for i in range(1, row):

                probability = 0
                key = row_list[i]#states
                if key in self.emissions:
                    inner_dict_emit = self.emissions[key]
                    if col_list[j] in inner_dict_emit:
                        p1 = inner_dict_emit[col_list[j]]
                        inner_dict_trans = self.transitions[row_list[i]]
                        for k in range(1, row):

                            p2 = inner_dict_trans[row_list[k]]
                            probability += p1*p2*matrix[k][j-1]
                        matrix[i][j] = probability

        numpy_matrix = np.array(matrix)
        max_row = numpy_matrix.argmax(axis=0)
        max_element = [row_list[element] for element in max_row]
        observation.stateseq = max_element


    ## you do this: Implement the Viterbi alborithm. Given an Observation (a list of outputs or emissions)
    ## determine the most likely sequence of states.

    def viterbi(self, observation):
        """given an observation,
        find and return the state sequence that generated
        the output sequence, using the Viterbi algorithm.
        """
        # create a matrix with columns as observations and rows would be the states
        col_list = observation.outputseq  # observations
        row_list = list(self.transitions.keys())  # states

        for element in row_list:
            index_1 = row_list.index(row_list[0])
            index_2 = row_list.index('#')
            if (element == '#'):
                row_list[index_1], row_list[index_2] = row_list[index_2], row_list[index_1]
        col = len(col_list)
        row = len(row_list)
        matrix = [[0 for _ in range(col)] for _ in range(row)]
        backpointers = [[None for _ in range(col)] for _ in range(row)]


        backpointers[0][0] = 0
        for i in range(1, row):

            key = row_list[i]

            if key in self.emissions:
                inner_dict_emit = self.emissions[key]
                first_column = col_list[0]
                if (first_column in inner_dict_emit):
                    inner_dict_trans = self.transitions[row_list[0]]
                    tran_probability = inner_dict_trans[row_list[i]]
                    emit_probability = inner_dict_emit[first_column]
                    probability = tran_probability * emit_probability
                    matrix[i][0] = probability
                else:
                    matrix[i][0] = 0
            backpointers[i][0] = 0

        max_value = -1.0
        local_max_state = -sys.maxsize -1
        max_state = -sys.maxsize -1

        for j in range(1, col):#observations
            for i in range(1, row):#states

                probability = 0
                key = row_list[i]
                if key in self.emissions:
                    inner_dict_emit = self.emissions[key]
                    if col_list[j] in inner_dict_emit:
                        p1 = inner_dict_emit[col_list[j]]
                        inner_dict_trans = self.transitions[row_list[i]]
                        for k in range(1, row):#states

                            p2 = inner_dict_trans[row_list[k]]
                            probability = p1*p2*matrix[k][j-1]
                            if(probability>max_value):
                                max_value = probability
                                max_state = k
                        matrix[i][j] = max_value
                        max_value = -1.0
                        backpointers[i][j] = max_state

        numpy_matrix = np.array(matrix)
        max_row = numpy_matrix.argmax(axis=0)
        max_element = [row_list[element] for element in max_row]
        observation.stateseq = max_element

if __name__ == '__main__':
    model = HMM()

    obs_words = ["i",  "shot", "the",  "elephant",  "."]
    obs = Observation([], obs_words)


    parser = argparse.ArgumentParser(description='Reading from terminal')
    parser.add_argument('--generate', type=int, help='run generate method')
    parser.add_argument('filename', type=str, help='file')
    parser.add_argument('--forward', type=str, help='obs file')
    parser.add_argument('--viterbi', type=str, help='obs file')

    args = parser.parse_args()
    model.load(args.filename)


    call_forward(args.forward)

    call_viterbi(args.viterbi)

