

import random
import argparse
import codecs
import os
import numpy

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

        # print(self.transitions)
   ## you do this.
    def generate(self, n):
        """return an n-length observation by randomly sampling from this HMM."""
        start_state = '#'
        count = 0
        list_trans = []
        list_emission = []
        while count!=n:
            if start_state in self.transitions:
                inner_dict = self.transitions[start_state]
                list_of_values = list(inner_dict)
                random_value = random.choices(list_of_values, k=1)
                list_trans.append(random_value[0]) #Taking 0th element as random.choices return a list and we have k=1 ie one value we get it from random_value[0]
                start_state = random_value[0]

            count+=1

        for item in list_trans:
            if item in self.emissions:
                inner_dict = self.emissions[item]
                list_of_values = list(inner_dict)
                random_value = random.choices(list_of_values, k=1)
                list_emission.append(random_value[0])


    ## you do this: Implement the Viterbi alborithm. Given an Observation (a list of outputs or emissions)
    ## determine the most likely sequence of states.

    def viterbi(self, observation):
        """given an observation,
        find and return the state sequence that generated
        the output sequence, using the Viterbi algorithm.
        """

model = HMM()
model.load('partofspeech.browntags.trained')

model.generate(10)
