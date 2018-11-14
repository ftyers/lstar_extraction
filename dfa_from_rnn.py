from LSTM import LSTMNetwork
from GRU import GRUNetwork
from RNNClassifier import RNNClassifier
from Training_Functions import make_train_set_for_target,mixed_curriculum_train
from Tomita_Grammars import tomita_1, tomita_2, tomita_3, tomita_4, tomita_5, tomita_6, tomita_7
from Extraction import extract
from math import pow

target = tomita_3
alphabet = "01"

print(target("00011000"))
print(target("10011000"))

print('========================================================================')

train_set = make_train_set_for_target(target,alphabet)

print(len(train_set))
print(list(train_set.items())[:10])

rnn = RNNClassifier(alphabet,num_layers=1,hidden_dim=10,RNNClass = LSTMNetwork)

mixed_curriculum_train(rnn,train_set,stop_threshold = 0.0005)

all_words = sorted(list(train_set.keys()),key=lambda x:len(x))
pos = next((w for w in all_words if rnn.classify_word(w)==True),None)
neg = next((w for w in all_words if rnn.classify_word(w)==False),None)
starting_examples = [w for w in [pos,neg] if not None == w]
print(starting_examples)

rnn.renew()

dfa = extract(rnn,time_limit = 50,initial_split_depth = 10,starting_examples=starting_examples)

print('========================================================================')

def percent(num,digits=2):
	tens = pow(10,digits)
	return str(int(100*num*tens)/tens)+"%"

dfa.draw_nicely(maximum=30) #max size willing to draw\n",
test_set = train_set
print("testing on train set, i.e. test set is train set")
# we're printing stats on the train set for now, but you can define other test sets by using\n",
# make_train_set_for_target\n",

n = len(test_set)
print("test set size:", n)
pos = len([w for w in test_set if target(w)])
print("of which positive:",pos,"("+percent(pos/n)+")")
rnn_target = len([w for w in test_set if rnn.classify_word(w)==target(w)])
print("rnn score against target on test set:",rnn_target,"("+percent(rnn_target/n)+")")
dfa_rnn = len([w for w in test_set if rnn.classify_word(w)==dfa.classify_word(w)])
print("extracted dfa score against rnn on test set:",dfa_rnn,"("+percent(dfa_rnn/n)+")")
dfa_target = len([w for w in test_set if dfa.classify_word(w)==target(w)])
print("extracted dfa score against target on rnn's test set:",dfa_target,"("+percent(dfa_target/n)+")")

print('========================================================================')
