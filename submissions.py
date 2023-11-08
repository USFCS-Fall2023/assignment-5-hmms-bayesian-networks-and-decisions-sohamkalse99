import HMM as hmm_class
from HMM import HMM
from HMM import Observation

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
            print(str(obs))
            output_file.write(str(obs))
            output_file.close()
        count += 1
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
            print(str(obs))
            output_file.write(str(obs))
            output_file.close()
        count += 1
    file.close()

if __name__ == '__main__':

    filename = 'partofspeech.browntags.trained'
    obs_filename = 'ambiguous_sents.obs'
    model = HMM()
    model.load(filename)
    print('==========Call Generate Method===========')
    model.generate(10)
    print('===============Call Forward=============')
    call_forward(obs_filename)
    print('===============Call Viterbi=============')
    call_viterbi(obs_filename)
